#!/usr/bin/env python3
"""
Langfuse Configuration Validation Script

This script validates Langfuse configuration settings, credentials, and integration
for the AutoOps Retail Optimization System.

Usage:
    python scripts/validate_langfuse_config.py [OPTIONS]

Options:
    --env-file FILE    Path to .env file (default: .env)
    --check-credentials    Validate Langfuse credentials
    --check-connection     Test Langfuse connection
    --check-config         Validate configuration settings
    --check-all            Run all validation checks
    --verbose              Enable verbose output
    --help                 Show this help message
"""

import argparse
import os
import sys
import json
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from urllib.parse import urlparse

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import get_settings
from config.langfuse_config import LangfuseClient, LangfuseConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Custom exception for configuration errors."""

    pass


class LangfuseConfigValidator:
    """Validates Langfuse configuration and credentials."""

    def __init__(self, env_file: Optional[Path] = None):
        self.env_file = env_file or (project_root / ".env")
        self.validation_results = {}

    def load_env_file(self) -> Dict[str, str]:
        """Load environment variables from .env file."""
        if not self.env_file.exists():
            raise ConfigurationError(f"Environment file not found: {self.env_file}")

        env_vars = {}
        with open(self.env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

        return env_vars

    def validate_credentials(self, verbose: bool = False) -> Dict[str, Any]:
        """Validate Langfuse credentials."""
        logger.info("Validating Langfuse credentials...")

        try:
            env_vars = self.load_env_file()
        except ConfigurationError as e:
            return {
                "valid": False,
                "error": str(e),
                "details": "Cannot load environment file",
            }

        issues = []
        warnings = []

        # Check required credentials
        required_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
        for key in required_keys:
            if key not in env_vars:
                issues.append(f"Missing required environment variable: {key}")
            elif (
                not env_vars[key]
                or env_vars[key] == f"{key.lower()}=your_{key.lower()}_here"
            ):
                issues.append(f"Invalid or placeholder value for {key}")
            elif len(env_vars[key]) < 10:  # Basic length check
                warnings.append(f"Suspiciously short value for {key}")

        # Validate key formats
        if "LANGFUSE_PUBLIC_KEY" in env_vars and env_vars["LANGFUSE_PUBLIC_KEY"]:
            if not env_vars["LANGFUSE_PUBLIC_KEY"].startswith("pk-lf-"):
                warnings.append(
                    "LANGFUSE_PUBLIC_KEY does not start with expected prefix 'pk-lf-'"
                )

        if "LANGFUSE_SECRET_KEY" in env_vars and env_vars["LANGFUSE_SECRET_KEY"]:
            if not env_vars["LANGFUSE_SECRET_KEY"].startswith("sk-lf-"):
                warnings.append(
                    "LANGFUSE_SECRET_KEY does not start with expected prefix 'sk-lf-'"
                )

        result = {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}

        if verbose:
            logger.info(
                f"Credentials validation: {'PASSED' if result['valid'] else 'FAILED'}"
            )
            if issues:
                for issue in issues:
                    logger.error(f"  ❌ {issue}")
            if warnings:
                for warning in warnings:
                    logger.warning(f"  ⚠️  {warning}")

        return result

    def validate_connection(self, verbose: bool = False) -> Dict[str, Any]:
        """Test Langfuse connection."""
        logger.info("Testing Langfuse connection...")

        try:
            # Try to initialize client and perform health check
            client = LangfuseClient()
            health = client.health_check()

            result = {
                "valid": health["available"],
                "connection_status": health.get("connection_status", "unknown"),
                "details": health,
            }

            if verbose:
                if result["valid"]:
                    logger.info("✅ Connection test PASSED")
                    logger.info(f"   Status: {result['connection_status']}")
                else:
                    logger.error("❌ Connection test FAILED")
                    logger.error(f"   Status: {result['connection_status']}")

            return result

        except Exception as e:
            result = {"valid": False, "error": str(e), "connection_status": "error"}

            if verbose:
                logger.error(f"❌ Connection test FAILED: {e}")

            return result

    def validate_configuration(self, verbose: bool = False) -> Dict[str, Any]:
        """Validate Langfuse configuration settings."""
        logger.info("Validating Langfuse configuration...")

        try:
            settings = get_settings()
            langfuse_settings = settings.langfuse

            issues = []
            warnings = []

            # Validate numeric ranges
            if not (0.0 <= langfuse_settings.sample_rate <= 1.0):
                issues.append(
                    f"Invalid sample_rate: {langfuse_settings.sample_rate} (must be 0.0-1.0)"
                )

            if langfuse_settings.flush_interval <= 0:
                issues.append(
                    f"Invalid flush_interval: {langfuse_settings.flush_interval} (must be > 0)"
                )

            if langfuse_settings.flush_at <= 0:
                issues.append(
                    f"Invalid flush_at: {langfuse_settings.flush_at} (must be > 0)"
                )

            if langfuse_settings.timeout <= 0:
                issues.append(
                    f"Invalid timeout: {langfuse_settings.timeout} (must be > 0)"
                )

            if langfuse_settings.max_retries < 0:
                issues.append(
                    f"Invalid max_retries: {langfuse_settings.max_retries} (must be >= 0)"
                )

            if langfuse_settings.max_latency_ms < 0:
                issues.append(
                    f"Invalid max_latency_ms: {langfuse_settings.max_latency_ms} (must be >= 0)"
                )

            if langfuse_settings.buffer_size <= 0:
                issues.append(
                    f"Invalid buffer_size: {langfuse_settings.buffer_size} (must be > 0)"
                )

            # Validate host URL
            if langfuse_settings.host:
                try:
                    parsed = urlparse(langfuse_settings.host)
                    if not parsed.scheme or not parsed.netloc:
                        warnings.append(
                            f"Invalid host URL format: {langfuse_settings.host}"
                        )
                except Exception:
                    warnings.append(f"Cannot parse host URL: {langfuse_settings.host}")

            # Validate blocked scopes format
            if langfuse_settings.blocked_instrumentation_scopes:
                try:
                    scopes = [
                        s.strip()
                        for s in langfuse_settings.blocked_instrumentation_scopes.split(
                            ","
                        )
                        if s.strip()
                    ]
                    if not scopes:
                        warnings.append(
                            "blocked_instrumentation_scopes is empty after parsing"
                        )
                except Exception as e:
                    issues.append(f"Invalid blocked_instrumentation_scopes format: {e}")

            # Validate additional headers format
            if langfuse_settings.additional_headers:
                try:
                    headers = json.loads(langfuse_settings.additional_headers)
                    if not isinstance(headers, dict):
                        issues.append("additional_headers must be a JSON object")
                except json.JSONDecodeError as e:
                    issues.append(f"Invalid additional_headers JSON: {e}")

            result = {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "config": {
                    "enabled": langfuse_settings.enabled,
                    "host": langfuse_settings.host,
                    "sample_rate": langfuse_settings.sample_rate,
                    "environment": langfuse_settings.environment,
                    "release": langfuse_settings.release,
                },
            }

            if verbose:
                logger.info(
                    f"Configuration validation: {'PASSED' if result['valid'] else 'FAILED'}"
                )
                if issues:
                    for issue in issues:
                        logger.error(f"  ❌ {issue}")
                if warnings:
                    for warning in warnings:
                        logger.warning(f"  ⚠️  {warning}")

                logger.info("Configuration summary:")
                logger.info(f"  Enabled: {result['config']['enabled']}")
                logger.info(f"  Host: {result['config']['host']}")
                logger.info(f"  Sample Rate: {result['config']['sample_rate']}")
                logger.info(f"  Environment: {result['config']['environment']}")
                logger.info(f"  Release: {result['config']['release']}")

            return result

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "details": "Failed to load configuration",
            }

    def run_all_validations(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("Running all Langfuse validation checks...")

        results = {
            "credentials": self.validate_credentials(verbose),
            "connection": self.validate_connection(verbose),
            "configuration": self.validate_configuration(verbose),
        }

        # Overall status
        overall_valid = all(result["valid"] for result in results.values())

        if verbose:
            logger.info(
                f"\nOverall validation: {'PASSED' if overall_valid else 'FAILED'}"
            )

            if overall_valid:
                logger.info("✅ All checks passed!")
            else:
                logger.error("❌ Some checks failed. Please review the issues above.")

        return {"overall_valid": overall_valid, "results": results}


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Langfuse configuration validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=project_root / ".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "--check-credentials", action="store_true", help="Validate Langfuse credentials"
    )
    parser.add_argument(
        "--check-connection", action="store_true", help="Test Langfuse connection"
    )
    parser.add_argument(
        "--check-config", action="store_true", help="Validate configuration settings"
    )
    parser.add_argument(
        "--check-all", action="store_true", help="Run all validation checks"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if not any(
        [
            args.check_credentials,
            args.check_connection,
            args.check_config,
            args.check_all,
        ]
    ):
        parser.print_help()
        return 1

    try:
        validator = LangfuseConfigValidator(args.env_file)

        if args.check_credentials:
            result = validator.validate_credentials(args.verbose)
            return 0 if result["valid"] else 1

        elif args.check_connection:
            result = validator.validate_connection(args.verbose)
            return 0 if result["valid"] else 1

        elif args.check_config:
            result = validator.validate_configuration(args.verbose)
            return 0 if result["valid"] else 1

        elif args.check_all:
            results = validator.run_all_validations(args.verbose)
            return 0 if results["overall_valid"] else 1

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
