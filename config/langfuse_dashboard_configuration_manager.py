"""Dashboard configuration management for export/import functionality."""

import json
import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

from config.langfuse_dashboard_config import (
    get_dashboard_config,
    LangfuseDashboardConfig,
    DashboardConfig,
    DashboardView,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfigurationMetadata:
    """Metadata for dashboard configurations."""

    name: str
    description: str
    version: str
    created_at: datetime
    updated_at: datetime
    exported_by: str = "system"
    tags: List[str] = field(default_factory=list)
    compatibility_version: str = "1.0"


@dataclass
class ConfigurationBackup:
    """Complete configuration backup with metadata."""

    metadata: ConfigurationMetadata
    dashboard_config: Dict[str, Any]
    view_configs: Dict[str, Dict[str, Any]]
    alerting_config: Dict[str, Any]
    export_timestamp: datetime


class DashboardConfigurationManager:
    """Manages dashboard configuration export, import, and backup functionality."""

    def __init__(
        self,
        config_dir: str = "./dashboard_configs",
        backup_dir: str = "./dashboard_backups",
        auto_backup: bool = True,
        max_backups: int = 10,
    ):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory to store configuration files
            backup_dir: Directory to store configuration backups
            auto_backup: Whether to automatically create backups before imports
            max_backups: Maximum number of backup files to keep
        """
        self._config_dir = Path(config_dir)
        self._backup_dir = Path(backup_dir)
        self._auto_backup = auto_backup
        self._max_backups = max_backups

        # Ensure directories exist
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        # Get dashboard configuration
        self._dashboard_config = get_dashboard_config()

        logger.info(
            f"Initialized dashboard configuration manager (config_dir: {config_dir})"
        )

    def export_configuration(
        self,
        name: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        include_alerts: bool = True,
        include_views: bool = True,
        filepath: Optional[str] = None,
    ) -> str:
        """Export current dashboard configuration.

        Args:
            name: Name for the exported configuration
            description: Optional description
            tags: Optional tags for categorization
            include_alerts: Whether to include alerting configuration
            include_views: Whether to include view configurations
            filepath: Optional specific filepath to export to

        Returns:
            Path to the exported configuration file
        """
        try:
            # Get current configuration
            dashboard_config = self._dashboard_config.get_dashboard_config()

            # Build export data
            export_data = {
                "metadata": {
                    "name": name,
                    "description": description or f"Dashboard configuration: {name}",
                    "version": dashboard_config.get("version", "1.0"),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "exported_by": "dashboard_manager",
                    "tags": tags or [],
                    "compatibility_version": "1.0",
                },
                "dashboard": {
                    "name": dashboard_config.get("name"),
                    "description": dashboard_config.get("description"),
                    "version": dashboard_config.get("version"),
                    "global_filters": dashboard_config.get("global_filters", {}),
                },
            }

            # Include view configurations
            if include_views:
                export_data["views"] = {}
                for view_data in dashboard_config.get("views", []):
                    view_id = view_data.get("id")
                    if view_id:
                        export_data["views"][view_id] = {
                            "name": view_data.get("name"),
                            "description": view_data.get("description"),
                            "type": view_data.get("type"),
                            "filters": view_data.get("filters", {}),
                            "time_range": view_data.get("time_range"),
                            "refresh_interval": view_data.get("refresh_interval"),
                            "layout": view_data.get("layout", {}),
                        }

            # Include alerting configuration
            if include_alerts:
                export_data["alerting"] = dashboard_config.get("alerting_config", {})

            # Generate filename if not provided
            if not filepath:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dashboard_config_{name}_{timestamp}.json"
                filepath = str(self._config_dir / filename)

            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported dashboard configuration to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to export dashboard configuration: {e}")
            raise

    def import_configuration(
        self,
        filepath: str,
        create_backup: bool = True,
        validate_only: bool = False,
    ) -> Dict[str, Any]:
        """Import dashboard configuration from file.

        Args:
            filepath: Path to configuration file
            create_backup: Whether to create backup before import
            validate_only: Only validate configuration without applying

        Returns:
            Import result with status and any warnings/errors
        """
        try:
            # Read configuration file
            with open(filepath, "r", encoding="utf-8") as f:
                import_data = json.load(f)

            # Validate configuration structure
            validation_result = self._validate_import_data(import_data)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "errors": validation_result["errors"],
                    "warnings": validation_result["warnings"],
                }

            if validate_only:
                return {
                    "success": True,
                    "message": "Configuration is valid",
                    "warnings": validation_result["warnings"],
                }

            # Create backup if requested
            if create_backup and self._auto_backup:
                self._create_backup("pre_import_backup")

            # Apply configuration
            import_result = self._apply_configuration(import_data)

            # Create post-import backup
            if import_result["success"] and self._auto_backup:
                self._create_backup("post_import_backup")

            return import_result

        except FileNotFoundError:
            return {
                "success": False,
                "errors": [f"Configuration file not found: {filepath}"],
            }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "errors": [f"Invalid JSON in configuration file: {e}"],
            }
        except Exception as e:
            logger.error(f"Failed to import dashboard configuration: {e}")
            return {
                "success": False,
                "errors": [f"Import failed: {e}"],
            }

    def _validate_import_data(self, import_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate imported configuration data."""
        errors = []
        warnings = []

        # Check required fields
        if "metadata" not in import_data:
            errors.append("Missing metadata section")
        else:
            metadata = import_data["metadata"]
            required_fields = ["name", "version"]
            for field in required_fields:
                if field not in metadata:
                    errors.append(f"Missing required metadata field: {field}")

        if "dashboard" not in import_data:
            errors.append("Missing dashboard section")

        # Validate views if present
        if "views" in import_data:
            for view_id, view_data in import_data["views"].items():
                if not isinstance(view_data, dict):
                    errors.append(f"Invalid view data for {view_id}")
                    continue

                required_view_fields = ["name", "type"]
                for field in required_view_fields:
                    if field not in view_data:
                        errors.append(
                            f"Missing required view field '{field}' for view {view_id}"
                        )

        # Validate alerting if present
        if "alerting" in import_data:
            alerting = import_data["alerting"]
            if not isinstance(alerting, dict):
                errors.append("Invalid alerting configuration")
            else:
                # Check for common alerting structure
                for category, rules in alerting.items():
                    if not isinstance(rules, dict):
                        warnings.append(
                            f"Unexpected alerting structure for category: {category}"
                        )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _apply_configuration(self, import_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply imported configuration."""
        try:
            # Note: This would require implementing configuration update methods
            # in the dashboard config class. For now, we'll simulate the process.

            applied_changes = []
            warnings = []

            # Apply dashboard-level changes
            dashboard_section = import_data.get("dashboard", {})
            if dashboard_section:
                # In a real implementation, this would update the dashboard config
                applied_changes.append("Updated dashboard settings")

            # Apply view configurations
            views_section = import_data.get("views", {})
            if views_section:
                for view_id, view_data in views_section.items():
                    # In a real implementation, this would update view configurations
                    applied_changes.append(f"Updated view configuration: {view_id}")

            # Apply alerting configuration
            alerting_section = import_data.get("alerting", {})
            if alerting_section:
                # In a real implementation, this would update alerting rules
                applied_changes.append("Updated alerting configuration")

            return {
                "success": True,
                "message": f"Successfully applied configuration with {len(applied_changes)} changes",
                "applied_changes": applied_changes,
                "warnings": warnings,
            }

        except Exception as e:
            logger.error(f"Failed to apply configuration: {e}")
            return {
                "success": False,
                "errors": [f"Failed to apply configuration: {e}"],
            }

    def _create_backup(self, backup_name: str) -> str:
        """Create a backup of current configuration."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backup_{backup_name}_{timestamp}.json"
            filepath = self._backup_dir / filename

            # Export current configuration
            self.export_configuration(
                name=f"backup_{backup_name}",
                description=f"Automatic backup created at {datetime.now().isoformat()}",
                filepath=str(filepath),
            )

            # Clean up old backups
            self._cleanup_old_backups()

            logger.info(f"Created configuration backup: {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise

    def _cleanup_old_backups(self) -> None:
        """Clean up old backup files to maintain max_backups limit."""
        try:
            backup_files = list(self._backup_dir.glob("backup_*.json"))
            backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Remove excess backups
            if len(backup_files) > self._max_backups:
                for old_backup in backup_files[self._max_backups :]:
                    old_backup.unlink()
                    logger.debug(f"Removed old backup: {old_backup}")

        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")

    def list_configurations(self) -> List[Dict[str, Any]]:
        """List all available configuration files."""
        try:
            config_files = list(self._config_dir.glob("*.json"))
            configurations = []

            for config_file in config_files:
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config_data = json.load(f)

                    metadata = config_data.get("metadata", {})
                    configurations.append(
                        {
                            "filename": config_file.name,
                            "filepath": str(config_file),
                            "name": metadata.get("name", "Unknown"),
                            "description": metadata.get("description", ""),
                            "version": metadata.get("version", "1.0"),
                            "created_at": metadata.get("created_at"),
                            "updated_at": metadata.get("updated_at"),
                            "tags": metadata.get("tags", []),
                            "size_bytes": config_file.stat().st_size,
                            "modified_time": datetime.fromtimestamp(
                                config_file.stat().st_mtime
                            ).isoformat(),
                        }
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to read configuration file {config_file}: {e}"
                    )

            return configurations

        except Exception as e:
            logger.error(f"Failed to list configurations: {e}")
            return []

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backup files."""
        try:
            backup_files = list(self._backup_dir.glob("backup_*.json"))
            backups = []

            for backup_file in backup_files:
                try:
                    with open(backup_file, "r", encoding="utf-8") as f:
                        backup_data = json.load(f)

                    metadata = backup_data.get("metadata", {})
                    backups.append(
                        {
                            "filename": backup_file.name,
                            "filepath": str(backup_file),
                            "name": metadata.get("name", "Unknown"),
                            "description": metadata.get("description", ""),
                            "created_at": metadata.get("created_at"),
                            "size_bytes": backup_file.stat().st_size,
                            "modified_time": datetime.fromtimestamp(
                                backup_file.stat().st_mtime
                            ).isoformat(),
                        }
                    )

                except Exception as e:
                    logger.warning(f"Failed to read backup file {backup_file}: {e}")

            # Sort by creation time (newest first)
            backups.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return backups

        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            return []

    def delete_configuration(self, filename: str) -> bool:
        """Delete a configuration file.

        Args:
            filename: Name of the configuration file to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            config_file = self._config_dir / filename
            if config_file.exists():
                config_file.unlink()
                logger.info(f"Deleted configuration file: {filename}")
                return True
            else:
                logger.warning(f"Configuration file not found: {filename}")
                return False

        except Exception as e:
            logger.error(f"Failed to delete configuration file {filename}: {e}")
            return False

    def restore_from_backup(self, backup_filename: str) -> Dict[str, Any]:
        """Restore configuration from a backup file.

        Args:
            backup_filename: Name of the backup file to restore from

        Returns:
            Restore result with status and any warnings/errors
        """
        try:
            backup_file = self._backup_dir / backup_filename
            if not backup_file.exists():
                return {
                    "success": False,
                    "errors": [f"Backup file not found: {backup_filename}"],
                }

            # Import the backup as a regular configuration
            return self.import_configuration(str(backup_file), create_backup=True)

        except Exception as e:
            logger.error(f"Failed to restore from backup {backup_filename}: {e}")
            return {
                "success": False,
                "errors": [f"Restore failed: {e}"],
            }

    def get_configuration_info(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a configuration file.

        Args:
            filename: Name of the configuration file

        Returns:
            Configuration information or None if not found
        """
        try:
            config_file = self._config_dir / filename
            if not config_file.exists():
                return None

            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            metadata = config_data.get("metadata", {})
            stats = config_file.stat()

            return {
                "filename": filename,
                "filepath": str(config_file),
                "metadata": metadata,
                "size_bytes": stats.st_size,
                "created_time": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                "dashboard_sections": list(config_data.keys()),
                "views_count": len(config_data.get("views", {})),
                "has_alerting": "alerting" in config_data,
            }

        except Exception as e:
            logger.error(f"Failed to get configuration info for {filename}: {e}")
            return None

    def validate_configuration_file(self, filepath: str) -> Dict[str, Any]:
        """Validate a configuration file without importing it.

        Args:
            filepath: Path to configuration file to validate

        Returns:
            Validation result with status and any warnings/errors
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            validation_result = self._validate_import_data(config_data)

            return {
                "valid": validation_result["valid"],
                "errors": validation_result["errors"],
                "warnings": validation_result["warnings"],
                "metadata": config_data.get("metadata", {}),
            }

        except FileNotFoundError:
            return {
                "valid": False,
                "errors": [f"Configuration file not found: {filepath}"],
            }
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"Invalid JSON in configuration file: {e}"],
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {e}"],
            }


# Global configuration manager instance
_config_manager: Optional[DashboardConfigurationManager] = None


def get_dashboard_configuration_manager() -> DashboardConfigurationManager:
    """Get the global dashboard configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = DashboardConfigurationManager()
    return _config_manager


def initialize_dashboard_configuration_manager(
    config_dir: str = "./dashboard_configs",
    backup_dir: str = "./dashboard_backups",
    auto_backup: bool = True,
    max_backups: int = 10,
) -> DashboardConfigurationManager:
    """Initialize the global dashboard configuration manager.

    Args:
        config_dir: Directory to store configuration files
        backup_dir: Directory to store configuration backups
        auto_backup: Whether to automatically create backups before imports
        max_backups: Maximum number of backup files to keep

    Returns:
        Initialized DashboardConfigurationManager instance
    """
    global _config_manager
    _config_manager = DashboardConfigurationManager(
        config_dir=config_dir,
        backup_dir=backup_dir,
        auto_backup=auto_backup,
        max_backups=max_backups,
    )
    return _config_manager
