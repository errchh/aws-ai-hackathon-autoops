"""Data masking and security utilities for Langfuse traces."""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Union, Pattern
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MaskingStrategy(Enum):
    """Strategies for masking sensitive data."""

    REPLACE = "replace"
    HASH = "hash"
    REDACT = "redact"
    PARTIAL = "partial"


@dataclass
class MaskingRule:
    """Configuration for a data masking rule."""

    field_path: (
        str  # JSON path to the field (e.g., "user.email", "payment.card_number")
    )
    strategy: MaskingStrategy
    replacement: Optional[str] = None  # For REPLACE strategy
    preserve_length: bool = False  # For PARTIAL strategy
    pattern: Optional[Pattern] = None  # Regex pattern to match
    description: str = ""


class DataMasker:
    """Handles masking of sensitive data in traces and logs."""

    def __init__(self, rules: Optional[List[MaskingRule]] = None):
        """Initialize the data masker with masking rules.

        Args:
            rules: List of masking rules to apply
        """
        self.rules = rules or self._get_default_rules()
        self._compiled_patterns: Dict[str, Pattern] = {}

        # Compile regex patterns for efficiency
        for rule in self.rules:
            if rule.pattern:
                self._compiled_patterns[rule.field_path] = rule.pattern

    def _get_default_rules(self) -> List[MaskingRule]:
        """Get default masking rules for common sensitive data."""
        return [
            # PII - Personally Identifiable Information
            MaskingRule(
                field_path="user.email",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=False,
                pattern=re.compile(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                ),
                description="Email addresses",
            ),
            MaskingRule(
                field_path="user.phone",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=True,
                pattern=re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
                description="Phone numbers",
            ),
            MaskingRule(
                field_path="user.ssn",
                strategy=MaskingStrategy.REPLACE,
                replacement="***-**-****",
                pattern=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
                description="Social Security Numbers",
            ),
            MaskingRule(
                field_path="user.name",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=False,
                description="Full names",
            ),
            # Financial Information
            MaskingRule(
                field_path="payment.card_number",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=True,
                pattern=re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
                description="Credit card numbers",
            ),
            MaskingRule(
                field_path="payment.cvv",
                strategy=MaskingStrategy.REPLACE,
                replacement="***",
                pattern=re.compile(r"\b\d{3,4}\b"),
                description="CVV codes",
            ),
            MaskingRule(
                field_path="payment.bank_account",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=True,
                pattern=re.compile(r"\b\d{8,12}\b"),
                description="Bank account numbers",
            ),
            # Authentication
            MaskingRule(
                field_path="auth.password",
                strategy=MaskingStrategy.REPLACE,
                replacement="********",
                description="Passwords",
            ),
            MaskingRule(
                field_path="auth.api_key",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=False,
                description="API keys",
            ),
            MaskingRule(
                field_path="auth.token",
                strategy=MaskingStrategy.REPLACE,
                replacement="[TOKEN_MASKED]",
                description="Authentication tokens",
            ),
            # Location Data
            MaskingRule(
                field_path="location.address",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=False,
                description="Street addresses",
            ),
            MaskingRule(
                field_path="location.coordinates",
                strategy=MaskingStrategy.REPLACE,
                replacement="[COORDINATES_MASKED]",
                description="GPS coordinates",
            ),
            # Business Sensitive Data
            MaskingRule(
                field_path="business.profit_margin",
                strategy=MaskingStrategy.REPLACE,
                replacement="[PROFIT_MASKED]",
                description="Profit margins",
            ),
            MaskingRule(
                field_path="business.revenue",
                strategy=MaskingStrategy.PARTIAL,
                preserve_length=False,
                description="Revenue figures",
            ),
        ]

    def add_rule(self, rule: MaskingRule) -> None:
        """Add a new masking rule.

        Args:
            rule: The masking rule to add
        """
        self.rules.append(rule)
        if rule.pattern:
            self._compiled_patterns[rule.field_path] = rule.pattern

    def remove_rule(self, field_path: str) -> bool:
        """Remove a masking rule by field path.

        Args:
            field_path: The field path of the rule to remove

        Returns:
            True if rule was removed, False if not found
        """
        for i, rule in enumerate(self.rules):
            if rule.field_path == field_path:
                del self.rules[i]
                if field_path in self._compiled_patterns:
                    del self._compiled_patterns[field_path]
                return True
        return False

    def mask_data(
        self, data: Union[Dict[str, Any], List[Any], str, Any]
    ) -> Union[Dict[str, Any], List[Any], str, Any]:
        """Apply masking rules to the given data.

        Args:
            data: The data to mask (dict, list, string, or other)

        Returns:
            The masked data with the same structure
        """
        return self._mask_data_recursive(data)

    def _mask_data_recursive(
        self, data: Union[Dict[str, Any], List[Any], str, Any], path_prefix: str = ""
    ) -> Union[Dict[str, Any], List[Any], str, Any]:
        """Apply masking rules to the given data with path awareness.

        Args:
            data: The data to mask
            path_prefix: Current path prefix for nested structures

        Returns:
            The masked data with the same structure
        """
        if isinstance(data, dict):
            return self._mask_dict(data, path_prefix)
        elif isinstance(data, list):
            return [self._mask_data_recursive(item, path_prefix) for item in data]
        elif isinstance(data, str):
            return self._mask_string(data)
        else:
            return data

    def _mask_dict(self, data: Dict[str, Any], path_prefix: str = "") -> Dict[str, Any]:
        """Mask a dictionary recursively with field path awareness."""
        masked = {}
        for key, value in data.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            # Check if this path matches any rule
            matched_rule = None
            for rule in self.rules:
                if rule.field_path == current_path or current_path.endswith(
                    "." + rule.field_path.split(".")[-1]
                ):
                    matched_rule = rule
                    break

            if matched_rule:
                masked[key] = self._apply_rule_to_value(value, matched_rule)
            else:
                masked[key] = self._mask_data_recursive(value, current_path)
        return masked

    def mask_data(
        self, data: Union[Dict[str, Any], List[Any], str, Any], path_prefix: str = ""
    ) -> Union[Dict[str, Any], List[Any], str, Any]:
        """Apply masking rules to the given data with path awareness.

        Args:
            data: The data to mask
            path_prefix: Current path prefix for nested structures

        Returns:
            The masked data with the same structure
        """
        if isinstance(data, dict):
            return self._mask_dict(data, path_prefix)
        elif isinstance(data, list):
            return [self.mask_data(item, path_prefix) for item in data]
        elif isinstance(data, str):
            return self._mask_string(data)
        else:
            return data

    def _apply_rule_to_value(self, value: Any, rule: MaskingRule) -> Any:
        """Apply a specific masking rule to a value."""
        if rule.strategy == MaskingStrategy.REPLACE and rule.replacement:
            return rule.replacement
        elif (
            rule.strategy == MaskingStrategy.PARTIAL
            and rule.pattern
            and isinstance(value, str)
        ):

            def mask_match(match):
                original = match.group(0)
                if rule.preserve_length:
                    return original[0] + "*" * (len(original) - 2) + original[-1]
                else:
                    return f"[MASKED_{rule.description.upper()}]"

            return rule.pattern.sub(mask_match, value)
        elif rule.strategy == MaskingStrategy.REDACT:
            return "[REDACTED]"
        else:
            return value

    def _mask_string(self, text: str) -> str:
        """Mask sensitive information in a string."""
        masked_text = text

        for rule in self.rules:
            if rule.strategy == MaskingStrategy.REPLACE and rule.replacement:
                # Simple replacement for exact matches
                if rule.field_path in text:  # This is a simplified check
                    masked_text = masked_text.replace(text, rule.replacement)
            elif rule.strategy == MaskingStrategy.PARTIAL and rule.pattern:
                # Use regex to find and mask patterns
                def mask_match(match):
                    original = match.group(0)
                    if rule.preserve_length:
                        return original[0] + "*" * (len(original) - 2) + original[-1]
                    else:
                        return f"[MASKED_{rule.description.upper()}]"

                masked_text = rule.pattern.sub(mask_match, masked_text)

        return masked_text

    def mask_trace_data(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in trace input/output/metadata.

        Args:
            trace_data: The trace data to mask

        Returns:
            Masked trace data
        """
        masked_data = {
            "input": self.mask_data(trace_data.get("input", {})),
            "output": self.mask_data(trace_data.get("output", {})),
            "metadata": self.mask_data(trace_data.get("metadata", {})),
        }

        # Preserve non-sensitive fields
        for key in [
            "id",
            "name",
            "start_time",
            "end_time",
            "level",
            "status",
            "status_message",
        ]:
            if key in trace_data:
                masked_data[key] = trace_data[key]

        return masked_data

    def should_mask_field(self, field_path: str) -> bool:
        """Check if a field should be masked based on rules.

        Args:
            field_path: The field path to check

        Returns:
            True if the field should be masked
        """
        for rule in self.rules:
            if rule.field_path == field_path or field_path.startswith(
                rule.field_path + "."
            ):
                return True
        return False

    def get_masking_summary(self) -> Dict[str, Any]:
        """Get a summary of masking rules and statistics.

        Returns:
            Dictionary with masking configuration summary
        """
        return {
            "total_rules": len(self.rules),
            "rules": [
                {
                    "field_path": rule.field_path,
                    "strategy": rule.strategy.value,
                    "description": rule.description,
                    "has_pattern": rule.pattern is not None,
                }
                for rule in self.rules
            ],
            "compiled_patterns": len(self._compiled_patterns),
        }


class SecureTraceManager:
    """Manages secure trace creation and data handling."""

    def __init__(self, masker: Optional[DataMasker] = None):
        """Initialize the secure trace manager.

        Args:
            masker: Optional DataMasker instance. If None, creates default.
        """
        self.masker = masker or DataMasker()
        self._retention_policies: Dict[str, int] = {}  # field_path -> days
        self._access_controls: Dict[str, List[str]] = {}  # trace_type -> allowed_roles

    def set_retention_policy(self, field_path: str, days: int) -> None:
        """Set data retention policy for a field.

        Args:
            field_path: The field path
            days: Number of days to retain data
        """
        self._retention_policies[field_path] = days
        logger.info(f"Set retention policy for {field_path}: {days} days")

    def set_access_control(self, trace_type: str, allowed_roles: List[str]) -> None:
        """Set access control for trace types.

        Args:
            trace_type: Type of trace (e.g., "simulation", "agent_decision")
            allowed_roles: List of roles that can access this trace type
        """
        self._access_controls[trace_type] = allowed_roles
        logger.info(f"Set access control for {trace_type}: {allowed_roles}")

    def create_secure_trace(
        self,
        trace_type: str,
        data: Dict[str, Any],
        user_roles: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a secure trace with masking and access control.

        Args:
            trace_type: Type of the trace
            data: The trace data
            user_roles: Roles of the user creating the trace

        Returns:
            Secure trace data or None if access denied
        """
        # Check access control
        if not self._check_access(trace_type, user_roles):
            logger.warning(f"Access denied for trace type: {trace_type}")
            return None

        # Apply data masking
        masked_data = self.masker.mask_trace_data(data)

        # Apply retention policies
        masked_data = self._apply_retention_policies(masked_data)

        return masked_data

    def _check_access(
        self, trace_type: str, user_roles: Optional[List[str]] = None
    ) -> bool:
        """Check if user has access to create this trace type."""
        if trace_type not in self._access_controls:
            return True  # No restrictions

        if not user_roles:
            return False  # No roles provided

        allowed_roles = self._access_controls[trace_type]
        return any(role in allowed_roles for role in user_roles)

    def _apply_retention_policies(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply retention policies to remove expired data."""
        # This is a simplified implementation
        # In a real system, you'd check timestamps and remove data accordingly
        return data

    def validate_credentials(self, credentials: Dict[str, str]) -> bool:
        """Validate that credentials are properly secured.

        Args:
            credentials: Dictionary of credentials to validate

        Returns:
            True if credentials are secure
        """
        required_fields = ["public_key", "secret_key"]

        for field in required_fields:
            if field in credentials:
                value = credentials[field]
                if not value or len(value) < 10:  # Basic validation
                    logger.warning(f"Invalid credential format for {field}")
                    return False

        return True

    def audit_trace_access(self, trace_id: str, user_id: str, action: str) -> None:
        """Audit trace access for compliance.

        Args:
            trace_id: ID of the trace
            user_id: ID of the user
            action: Action performed (e.g., "create", "view", "delete")
        """
        # In a real implementation, this would log to an audit system
        logger.info(f"Audit: User {user_id} performed {action} on trace {trace_id}")


# Global instances
_default_masker: Optional[DataMasker] = None
_secure_manager: Optional[SecureTraceManager] = None


def get_data_masker() -> DataMasker:
    """Get the global data masker instance."""
    global _default_masker
    if _default_masker is None:
        _default_masker = DataMasker()
    return _default_masker


def get_secure_trace_manager() -> SecureTraceManager:
    """Get the global secure trace manager instance."""
    global _secure_manager
    if _secure_manager is None:
        _secure_manager = SecureTraceManager()
    return _secure_manager


def initialize_security_features(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize security features with configuration.

    Args:
        config: Optional security configuration
    """
    global _default_masker, _secure_manager

    if config and "masking_rules" in config:
        # Load custom masking rules from config
        custom_rules = config.get("masking_rules", [])
        masker = DataMasker([MaskingRule(**rule) for rule in custom_rules])
    else:
        masker = DataMasker()

    _default_masker = masker
    _secure_manager = SecureTraceManager(masker)

    # Set up retention policies from config
    if config:
        retention_policies = config.get("retention_policies", {})
        for field_path, days in retention_policies.items():
            _secure_manager.set_retention_policy(field_path, days)

        # Set up access controls from config
        access_controls = config.get("access_controls", {})
        for trace_type, roles in access_controls.items():
            _secure_manager.set_access_control(trace_type, roles)

    logger.info("Security features initialized")
