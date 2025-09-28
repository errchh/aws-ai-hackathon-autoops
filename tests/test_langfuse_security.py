"""Tests for Langfuse security and data masking features."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from config.langfuse_data_masking import (
    DataMasker,
    SecureTraceManager,
    MaskingRule,
    MaskingStrategy,
)
from config.langfuse_config import LangfuseConfig


class TestDataMasker:
    """Test cases for DataMasker functionality."""

    def test_default_rules_creation(self):
        """Test that default masking rules are created correctly."""
        masker = DataMasker()

        assert len(masker.rules) > 0
        assert any(rule.field_path == "user.email" for rule in masker.rules)
        assert any(rule.field_path == "payment.card_number" for rule in masker.rules)

    def test_custom_rule_addition(self):
        """Test adding custom masking rules."""
        masker = DataMasker()

        initial_count = len(masker.rules)
        custom_rule = MaskingRule(
            field_path="custom.sensitive_field",
            strategy=MaskingStrategy.REPLACE,
            replacement="[CUSTOM_MASKED]",
            description="Custom sensitive field",
        )

        masker.add_rule(custom_rule)

        assert len(masker.rules) == initial_count + 1
        assert custom_rule in masker.rules

    def test_rule_removal(self):
        """Test removing masking rules."""
        masker = DataMasker()

        initial_count = len(masker.rules)
        rule_to_remove = masker.rules[0]

        result = masker.remove_rule(rule_to_remove.field_path)

        assert result is True
        assert len(masker.rules) == initial_count - 1
        assert rule_to_remove not in masker.rules

    def test_email_masking(self):
        """Test masking of email addresses."""
        masker = DataMasker()

        test_data = {
            "user": {"email": "test@example.com", "name": "John Doe"},
            "message": "Contact test@example.com for more info",
        }

        masked = masker.mask_data(test_data)

        assert "test@example.com" not in str(masked)
        assert "[MASKED_EMAIL ADDRESSES]" in str(masked)

    def test_credit_card_masking(self):
        """Test masking of credit card numbers."""
        masker = DataMasker()

        # Test direct field masking
        test_data = {"card_number": "4532-1234-5678-9012", "cvv": "123"}

        masked = masker.mask_data(test_data)

        assert "4532-1234-5678-9012" not in str(masked)
        assert "123" not in str(masked)
        assert "4532" in str(masked)  # First 4 digits should remain
        assert "9012" in str(masked)  # Last 4 digits should remain
        assert "***" in str(masked)

    def test_password_masking(self):
        """Test masking of passwords."""
        masker = DataMasker()

        test_data = {"auth": {"password": "secretpassword123"}}

        masked = masker.mask_data(test_data)

        assert "secretpassword123" not in str(masked)
        assert "********" in str(masked)

    def test_phone_number_masking(self):
        """Test masking of phone numbers."""
        masker = DataMasker()

        test_data = {"phone": "555-123-4567"}

        masked = masker.mask_data(test_data)

        assert "555-123-4567" not in str(masked)
        assert "555" in str(masked)  # First 3 digits should remain
        assert "4567" in str(masked)  # Last 4 digits should remain

    def test_nested_data_masking(self):
        """Test masking in nested data structures."""
        masker = DataMasker()

        test_data = {
            "users": [
                {"email": "user1@test.com", "phone": "555-000-1111"},
                {"email": "user2@test.com", "phone": "555-000-2222"},
            ],
            "metadata": {"description": "Test with user1@test.com contact info"},
        }

        masked = masker.mask_data(test_data)

        assert "user1@test.com" not in str(masked)
        assert "user2@test.com" not in str(masked)
        assert "555-000-1111" not in str(masked)
        assert "555-000-2222" not in str(masked)

    def test_masking_summary(self):
        """Test getting masking summary."""
        masker = DataMasker()

        summary = masker.get_masking_summary()

        assert "total_rules" in summary
        assert "rules" in summary
        assert len(summary["rules"]) == len(masker.rules)


class TestSecureTraceManager:
    """Test cases for SecureTraceManager functionality."""

    def test_retention_policy_setting(self):
        """Test setting retention policies."""
        manager = SecureTraceManager()

        manager.set_retention_policy("user.email", 30)

        assert "user.email" in manager._retention_policies
        assert manager._retention_policies["user.email"] == 30

    def test_access_control_setting(self):
        """Test setting access controls."""
        manager = SecureTraceManager()

        manager.set_access_control("simulation", ["admin", "analyst"])

        assert "simulation" in manager._access_controls
        assert manager._access_controls["simulation"] == ["admin", "analyst"]

    def test_access_check_with_roles(self):
        """Test access control checking with user roles."""
        manager = SecureTraceManager()
        manager.set_access_control("agent_decision", ["admin"])

        # Should allow access
        assert manager._check_access("agent_decision", ["admin"]) is True

        # Should deny access
        assert manager._check_access("agent_decision", ["user"]) is False

        # Should allow access for unrestricted trace types
        assert manager._check_access("unrestricted_type", ["user"]) is True

    def test_access_check_without_roles(self):
        """Test access control checking without user roles."""
        manager = SecureTraceManager()
        manager.set_access_control("restricted_type", ["admin"])

        # Should deny access without roles
        assert manager._check_access("restricted_type", None) is False

    def test_secure_trace_creation_allowed(self):
        """Test creating secure trace with proper access."""
        masker = DataMasker()
        manager = SecureTraceManager(masker)

        manager.set_access_control("simulation", ["admin"])

        trace_data = {
            "name": "test_trace",
            "input": {"user": {"email": "test@example.com"}},
            "metadata": {"timestamp": datetime.now().isoformat()},
        }

        secure_trace = manager.create_secure_trace(
            "simulation", trace_data, user_roles=["admin"]
        )

        assert secure_trace is not None
        assert "test@example.com" not in str(secure_trace)

    def test_secure_trace_creation_denied(self):
        """Test creating secure trace with denied access."""
        masker = DataMasker()
        manager = SecureTraceManager(masker)

        manager.set_access_control("simulation", ["admin"])

        trace_data = {
            "name": "test_trace",
            "input": {"user": {"email": "test@example.com"}},
            "metadata": {"timestamp": datetime.now().isoformat()},
        }

        secure_trace = manager.create_secure_trace(
            "simulation", trace_data, user_roles=["user"]
        )

        assert secure_trace is None

    def test_credential_validation(self):
        """Test credential validation."""
        manager = SecureTraceManager()

        # Valid credentials
        valid_creds = {
            "public_key": "pk-lf-12345678901234567890123456789012",
            "secret_key": "sk-lf-12345678901234567890123456789012",
        }
        assert manager.validate_credentials(valid_creds) is True

        # Invalid credentials
        invalid_creds = {"public_key": "short", "secret_key": ""}
        assert manager.validate_credentials(invalid_creds) is False


class TestSecurityIntegration:
    """Test integration of security features."""

    def test_security_initialization(self):
        """Test security feature initialization."""
        masker = DataMasker()
        manager = SecureTraceManager(masker)

        # Simulate initialization with custom config
        custom_rule = MaskingRule(
            field_path="custom.field",
            strategy=MaskingStrategy.REPLACE,
            replacement="[CUSTOM]",
            description="Custom field",
        )
        masker.add_rule(custom_rule)
        manager.set_retention_policy("user.email", 60)
        manager.set_access_control("simulation", ["admin"])

        # Check custom rule was added
        assert any(rule.field_path == "custom.field" for rule in masker.rules)

        # Check retention policy was set
        assert "user.email" in manager._retention_policies

        # Check access control was set
        assert "simulation" in manager._access_controls


if __name__ == "__main__":
    pytest.main([__file__])
