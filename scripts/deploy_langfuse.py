#!/usr/bin/env python3
"""
Langfuse Deployment Script for AutoOps Retail Optimization System

This script handles the setup, configuration validation, and deployment of Langfuse
observability integration for the multi-agent retail optimization system.

Usage:
    python scripts/deploy_langfuse.py [OPTIONS]

Options:
    --validate-only    Only validate configuration without making changes
    --setup-langfuse   Set up Langfuse integration (install dependencies, configure)
    --create-env       Create .env file from .env.example if it doesn't exist
    --health-check     Perform health check on Langfuse integration
    --help             Show this help message

Environment Variables:
    LANGFUSE_PUBLIC_KEY    Langfuse public key (required for setup)
    LANGFUSE_SECRET_KEY    Langfuse secret key (required for setup)
    LANGFUSE_HOST          Langfuse host URL (default: https://cloud.langfuse.com)
    LANGFUSE_ENVIRONMENT   Environment name (e.g., production, staging)
    LANGFUSE_RELEASE       Release version
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.langfuse_config import LangfuseClient, LangfuseConfig, get_langfuse_client
from config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LangfuseDeploymentError(Exception):
    """Custom exception for deployment errors."""

    pass


class LangfuseDeployer:
    """Handles Langfuse deployment and configuration."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.env_file = project_root / ".env"
        self.env_example = project_root / ".env.example"
        self.requirements_file = project_root / "pyproject.toml"

    def validate_environment(self) -> Dict[str, Any]:
        """Validate the current environment and configuration."""
        logger.info("Validating environment and configuration...")

        issues = []
        warnings = []

        # Check if .env file exists
        if not self.env_file.exists():
            issues.append(".env file not found. Run with --create-env to create one.")
        else:
            # Check for required Langfuse variables
            required_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
            missing_vars = []

            with open(self.env_file, "r") as f:
                env_content = f.read()

            for var in required_vars:
                if var not in env_content or f"{var}=" not in env_content:
                    missing_vars.append(var)

            if missing_vars:
                issues.append(
                    f"Missing required environment variables: {', '.join(missing_vars)}"
                )

        # Check if Langfuse SDK is installed
        try:
            import langfuse

            logger.info("Langfuse SDK is installed")
        except ImportError:
            issues.append(
                "Langfuse SDK not installed. Install with: pip install langfuse"
            )

        # Validate configuration if possible
        try:
            settings = get_settings()
            if settings.langfuse.enabled:
                client = get_langfuse_client()
                health = client.health_check()

                if not health["available"]:
                    issues.append(
                        f"Langfuse client not available: {health.get('connection_status', 'unknown')}"
                    )
                else:
                    logger.info("Langfuse client is healthy")
            else:
                warnings.append("Langfuse tracing is disabled in configuration")
        except Exception as e:
            warnings.append(f"Could not validate Langfuse configuration: {e}")

        return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings}

    def create_env_file(self) -> None:
        """Create .env file from .env.example if it doesn't exist."""
        if self.env_file.exists():
            logger.warning(".env file already exists. Skipping creation.")
            return

        if not self.env_example.exists():
            raise LangfuseDeploymentError(".env.example file not found")

        # Copy .env.example to .env
        import shutil

        shutil.copy(self.env_example, self.env_file)
        logger.info("Created .env file from .env.example")

        # Provide instructions for customization
        logger.info("Please edit .env file and set your Langfuse credentials:")
        logger.info("  - LANGFUSE_PUBLIC_KEY")
        logger.info("  - LANGFUSE_SECRET_KEY")
        logger.info("  - LANGFUSE_HOST (if using self-hosted instance)")

    def setup_langfuse_integration(self) -> None:
        """Set up Langfuse integration by installing dependencies and validating setup."""
        logger.info("Setting up Langfuse integration...")

        # Install/update Langfuse SDK
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "langfuse"],
                check=True,
                capture_output=True,
                text=True,
            )
            logger.info("Langfuse SDK installed/updated successfully")
        except subprocess.CalledProcessError as e:
            raise LangfuseDeploymentError(f"Failed to install Langfuse SDK: {e.stderr}")

        # Validate setup
        validation = self.validate_environment()
        if not validation["valid"]:
            error_msg = "Setup validation failed:\n" + "\n".join(
                f"  - {issue}" for issue in validation["issues"]
            )
            raise LangfuseDeploymentError(error_msg)

        logger.info("Langfuse integration setup completed successfully")

    def perform_health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check of the Langfuse integration."""
        logger.info("Performing Langfuse health check...")

        try:
            client = get_langfuse_client()
            health = client.health_check()

            # Additional checks
            if health["available"]:
                # Test trace creation
                try:
                    test_trace = client.client.start_span(  # type: ignore
                        name="deployment_health_check",
                        metadata={"test": True, "timestamp": "2024-01-01T00:00:00Z"},
                    )
                    client.flush()
                    logger.info("Trace creation test successful")
                except Exception as e:
                    health["warnings"] = health.get("warnings", []) + [
                        f"Trace creation test failed: {e}"
                    ]

            return health

        except Exception as e:
            return {"available": False, "error": str(e), "connection_status": "error"}

    def generate_documentation(self) -> None:
        """Generate configuration documentation."""
        doc_content = self._create_config_documentation()
        doc_file = self.project_root / "docs" / "langfuse_configuration.md"

        # Create docs directory if it doesn't exist
        doc_file.parent.mkdir(exist_ok=True)

        with open(doc_file, "w") as f:
            f.write(doc_content)

        logger.info(f"Configuration documentation generated: {doc_file}")

    def _create_config_documentation(self) -> str:
        """Create comprehensive configuration documentation."""
        return """# Langfuse Configuration Guide

## Overview

This guide provides detailed information on configuring Langfuse observability integration for the AutoOps Retail Optimization System.

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LANGFUSE_PUBLIC_KEY` | Your Langfuse public key | `pk-lf-...` |
| `LANGFUSE_SECRET_KEY` | Your Langfuse secret key | `sk-lf-...` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `LANGFUSE_HOST` | Langfuse host URL | `https://cloud.langfuse.com` | `https://your-instance.langfuse.com` |
| `LANGFUSE_ENABLED` | Enable/disable tracing | `true` | `false` |
| `LANGFUSE_SAMPLE_RATE` | Sampling rate (0.0-1.0) | `1.0` | `0.1` |
| `LANGFUSE_DEBUG` | Enable debug mode | `false` | `true` |
| `LANGFUSE_FLUSH_INTERVAL` | Flush interval in seconds | `5.0` | `10.0` |
| `LANGFUSE_FLUSH_AT` | Events to trigger flush | `15` | `10` |
| `LANGFUSE_TIMEOUT` | HTTP timeout in seconds | `60` | `30` |
| `LANGFUSE_MAX_RETRIES` | Maximum retry attempts | `3` | `5` |
| `LANGFUSE_MAX_LATENCY_MS` | Max acceptable latency | `100` | `50` |
| `LANGFUSE_ENABLE_SAMPLING` | Enable intelligent sampling | `true` | `false` |
| `LANGFUSE_BUFFER_SIZE` | Buffer size for traces | `1000` | `500` |
| `LANGFUSE_ENVIRONMENT` | Environment name | `development` | `production` |
| `LANGFUSE_RELEASE` | Release version | `0.1.0` | `1.0.0` |
| `LANGFUSE_TRACING_ENABLED` | Enable OpenTelemetry tracing | `true` | `false` |
| `LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT` | Media upload threads | `4` | `2` |

### Advanced Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LANGFUSE_BLOCKED_INSTRUMENTATION_SCOPES` | Comma-separated blocked scopes | `scope1,scope2` |
| `LANGFUSE_ADDITIONAL_HEADERS` | Additional HTTP headers (JSON) | `{"X-Custom": "value"}` |

## Setup Instructions

### 1. Get Langfuse Credentials

1. Sign up at [Langfuse](https://langfuse.com)
2. Create a new project
3. Copy your public and secret keys from the project settings

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and set your Langfuse credentials:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_ENVIRONMENT=production
LANGFUSE_RELEASE=1.0.0
```

### 3. Validate Configuration

Run the deployment script to validate your setup:

```bash
python scripts/deploy_langfuse.py --validate-only
```

### 4. Deploy Integration

Deploy the Langfuse integration:

```bash
python scripts/deploy_langfuse.py --setup-langfuse
```

## Deployment Scripts

### Validation Only

```bash
python scripts/deploy_langfuse.py --validate-only
```

### Full Setup

```bash
python scripts/deploy_langfuse.py --setup-langfuse
```

### Create Environment File

```bash
python scripts/deploy_langfuse.py --create-env
```

### Health Check

```bash
python scripts/deploy_langfuse.py --health-check
```

## Troubleshooting

### Common Issues

1. **Missing Credentials**: Ensure `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
2. **Connection Errors**: Check `LANGFUSE_HOST` and network connectivity
3. **Permission Errors**: Verify your Langfuse keys have the correct permissions
4. **High Latency**: Adjust `LANGFUSE_MAX_LATENCY_MS` or enable sampling

### Debug Mode

Enable debug mode for detailed logging:

```bash
LANGFUSE_DEBUG=true
```

### Health Check

Perform a health check:

```bash
python scripts/deploy_langfuse.py --health-check
```

## Security Considerations

1. **Credential Storage**: Store keys in environment variables, not in code
2. **Access Control**: Configure appropriate permissions in Langfuse
3. **Data Privacy**: Review what data is being traced and consider sampling
4. **Network Security**: Use HTTPS for all Langfuse communications

## Performance Tuning

### Sampling

Reduce tracing overhead with sampling:

```bash
LANGFUSE_SAMPLE_RATE=0.1  # Trace 10% of events
LANGFUSE_ENABLE_SAMPLING=true
```

### Buffer Management

Adjust buffer settings for your workload:

```bash
LANGFUSE_BUFFER_SIZE=500
LANGFUSE_FLUSH_AT=10
LANGFUSE_FLUSH_INTERVAL=10.0
```

### Latency Control

Set maximum acceptable latency:

```bash
LANGFUSE_MAX_LATENCY_MS=50
```

## Monitoring

Monitor the integration using the built-in health check:

```bash
python scripts/deploy_langfuse.py --health-check
```

Check the application logs for Langfuse-related messages.

## Support

For issues with Langfuse integration:

1. Check the [Langfuse Documentation](https://langfuse.com/docs)
2. Review application logs for error messages
3. Run the health check script
4. Contact the development team
"""


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(
        description="Langfuse deployment script for AutoOps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without making changes",
    )
    parser.add_argument(
        "--setup-langfuse",
        action="store_true",
        help="Set up Langfuse integration (install dependencies, configure)",
    )
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create .env file from .env.example if it does not exist",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform health check on Langfuse integration",
    )
    parser.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate configuration documentation",
    )

    args = parser.parse_args()

    if not any(
        [
            args.validate_only,
            args.setup_langfuse,
            args.create_env,
            args.health_check,
            args.generate_docs,
        ]
    ):
        parser.print_help()
        return 1

    try:
        deployer = LangfuseDeployer(project_root)

        if args.validate_only:
            validation = deployer.validate_environment()
            if validation["valid"]:
                logger.info("✅ Environment validation successful")
                if validation["warnings"]:
                    logger.warning("Warnings found:")
                    for warning in validation["warnings"]:
                        logger.warning(f"  - {warning}")
            else:
                logger.error("❌ Environment validation failed:")
                for issue in validation["issues"]:
                    logger.error(f"  - {issue}")
                return 1

        elif args.create_env:
            deployer.create_env_file()

        elif args.setup_langfuse:
            deployer.setup_langfuse_integration()

        elif args.health_check:
            health = deployer.perform_health_check()
            if health["available"]:
                logger.info("✅ Langfuse health check passed")
                logger.info(
                    f"Connection status: {health.get('connection_status', 'unknown')}"
                )
            else:
                logger.error("❌ Langfuse health check failed")
                logger.error(f"Error: {health.get('error', 'unknown')}")
                return 1

        elif args.generate_docs:
            deployer.generate_documentation()

        return 0

    except LangfuseDeploymentError as e:
        logger.error(f"Deployment error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
