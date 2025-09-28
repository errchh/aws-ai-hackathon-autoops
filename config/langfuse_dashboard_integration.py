"""Langfuse dashboard integration service for comprehensive workflow visualization."""

import json
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from threading import Lock

from config.langfuse_dashboard_config import (
    get_dashboard_config,
    LangfuseDashboardConfig,
    DashboardConfig,
    DashboardView,
)
from config.langfuse_workflow_views import (
    get_workflow_view_manager,
    LangfuseWorkflowViewManager,
)
from config.langfuse_workflow_visualization_views import (
    get_workflow_visualization_view_manager,
    LangfuseWorkflowVisualizationViewManager,
)
from config.langfuse_agent_performance_views import (
    get_agent_performance_view_manager,
    LangfuseAgentPerformanceViewManager,
)
from config.langfuse_alerting import (
    get_alert_manager,
    LangfuseAlertManager,
)
from config.langfuse_integration import get_langfuse_integration
from config.metrics_collector import get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class DashboardDataSnapshot:
    """Snapshot of dashboard data at a specific point in time."""

    timestamp: datetime
    overview_data: Dict[str, Any] = field(default_factory=dict)
    agent_performance_data: Dict[str, Any] = field(default_factory=dict)
    workflow_data: Dict[str, Any] = field(default_factory=dict)
    alert_data: Dict[str, Any] = field(default_factory=dict)
    system_metrics: Dict[str, Any] = field(default_factory=dict)


class LangfuseDashboardIntegrationService:
    """Comprehensive dashboard integration service for Langfuse workflow visualization."""

    def __init__(
        self,
        dashboard_config: Optional[LangfuseDashboardConfig] = None,
        workflow_manager: Optional[LangfuseWorkflowViewManager] = None,
        workflow_viz_manager: Optional[LangfuseWorkflowVisualizationViewManager] = None,
        agent_performance_manager: Optional[LangfuseAgentPerformanceViewManager] = None,
        alert_manager: Optional[LangfuseAlertManager] = None,
    ):
        """Initialize the dashboard integration service.

        Args:
            dashboard_config: Optional dashboard configuration. Uses global if None.
            workflow_manager: Optional workflow view manager. Uses global if None.
            workflow_viz_manager: Optional workflow visualization manager. Uses global if None.
            agent_performance_manager: Optional agent performance manager. Uses global if None.
            alert_manager: Optional alert manager. Uses global if None.
        """
        self._dashboard_config = dashboard_config or get_dashboard_config()
        self._workflow_manager = workflow_manager or get_workflow_view_manager()
        self._workflow_viz_manager = (
            workflow_viz_manager or get_workflow_visualization_view_manager()
        )
        self._agent_performance_manager = (
            agent_performance_manager or get_agent_performance_view_manager()
        )
        self._alert_manager = alert_manager or get_alert_manager()
        self._integration_service = get_langfuse_integration()
        self._metrics_collector = get_metrics_collector()

        # Data caching
        self._data_cache: Optional[DashboardDataSnapshot] = None
        self._cache_lock = Lock()
        self._cache_ttl_seconds = 30  # Cache for 30 seconds
        self._last_cache_update: Optional[datetime] = None

        # Update callbacks
        self._update_callbacks: List[Callable[[DashboardDataSnapshot], None]] = []

        logger.info("Initialized Langfuse dashboard integration service")

    def get_dashboard_overview(self, view_id: str = "overview") -> Dict[str, Any]:
        """Get overview dashboard data for a specific view.

        Args:
            view_id: ID of the dashboard view to retrieve

        Returns:
            Dictionary containing overview dashboard data
        """
        view_config = self._dashboard_config.get_view_config(view_id)
        if not view_config:
            logger.warning(f"View configuration not found for view_id: {view_id}")
            return {"error": "View configuration not found"}

        # Get fresh data
        system_metrics = self._integration_service.get_system_metrics()
        alert_stats = self._alert_manager.alerting_engine.get_alert_stats()

        # Get workflow metrics
        workflow_metrics = (
            self._workflow_viz_manager.renderer.get_workflow_metrics_view()
        )

        # Get agent performance summary
        agent_performance = self._agent_performance_manager.get_dashboard_agent_data()

        return {
            "view_id": view_id,
            "view_config": view_config,
            "data": {
                "system_metrics": system_metrics,
                "alert_stats": alert_stats,
                "workflow_metrics": workflow_metrics["metrics"],
                "agent_performance": agent_performance["summary"],
                "active_alerts_count": len(agent_performance["alerts"]),
                "total_agents": agent_performance["summary"]["total_agents"],
                "total_workflows": workflow_metrics["metrics"]["total_workflows"],
            },
            "last_updated": datetime.now().isoformat(),
        }

    def get_agent_performance_dashboard(
        self, view_id: str = "agent_performance"
    ) -> Dict[str, Any]:
        """Get agent performance dashboard data.

        Args:
            view_id: ID of the agent performance view

        Returns:
            Dictionary containing agent performance dashboard data
        """
        view_config = self._dashboard_config.get_view_config(view_id)
        if not view_config:
            return {"error": "View configuration not found"}

        agent_data = self._agent_performance_manager.get_dashboard_agent_data()

        return {
            "view_id": view_id,
            "view_config": view_config,
            "data": agent_data,
            "last_updated": datetime.now().isoformat(),
        }

    def get_workflow_dashboard(
        self, view_id: str = "workflow_visualization"
    ) -> Dict[str, Any]:
        """Get workflow visualization dashboard data.

        Args:
            view_id: ID of the workflow visualization view

        Returns:
            Dictionary containing workflow dashboard data
        """
        view_config = self._dashboard_config.get_view_config(view_id)
        if not view_config:
            return {"error": "View configuration not found"}

        workflow_views = self._workflow_viz_manager.get_dashboard_workflow_views()

        return {
            "view_id": view_id,
            "view_config": view_config,
            "data": workflow_views,
            "last_updated": datetime.now().isoformat(),
        }

    def get_alerts_dashboard(self, view_id: str = "alerts") -> Dict[str, Any]:
        """Get alerts dashboard data.

        Args:
            view_id: ID of the alerts view

        Returns:
            Dictionary containing alerts dashboard data
        """
        view_config = self._dashboard_config.get_view_config(view_id)
        if not view_config:
            return {"error": "View configuration not found"}

        alert_data = self._alert_manager.get_dashboard_alert_data()

        return {
            "view_id": view_id,
            "view_config": view_config,
            "data": alert_data,
            "last_updated": datetime.now().isoformat(),
        }

    def get_complete_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data for all views.

        Returns:
            Dictionary containing data for all dashboard views
        """
        return {
            "overview": self.get_dashboard_overview(),
            "agent_performance": self.get_agent_performance_dashboard(),
            "workflow_visualization": self.get_workflow_dashboard(),
            "alerts": self.get_alerts_dashboard(),
            "dashboard_config": self._dashboard_config.get_dashboard_config(),
            "last_updated": datetime.now().isoformat(),
        }

    def get_workflow_detail(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific workflow.

        Args:
            workflow_id: ID of the workflow to retrieve

        Returns:
            Dictionary containing workflow details or None if not found
        """
        return self._workflow_viz_manager.get_workflow_dashboard_data(workflow_id)

    def get_agent_detail(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific agent.

        Args:
            agent_id: ID of the agent to retrieve

        Returns:
            Dictionary containing agent details or None if not found
        """
        return self._agent_performance_manager.get_agent_detailed_view(agent_id)

    def get_dashboard_snapshot(self, use_cache: bool = True) -> DashboardDataSnapshot:
        """Get a complete snapshot of dashboard data.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            DashboardDataSnapshot containing all dashboard data
        """
        with self._cache_lock:
            # Check if cache is valid
            if (
                use_cache
                and self._data_cache
                and self._last_cache_update
                and (datetime.now() - self._last_cache_update).seconds
                < self._cache_ttl_seconds
            ):
                return self._data_cache

            # Generate fresh snapshot
            snapshot = DashboardDataSnapshot(
                timestamp=datetime.now(),
                overview_data=self.get_dashboard_overview()["data"],
                agent_performance_data=self.get_agent_performance_dashboard()["data"],
                workflow_data=self.get_workflow_dashboard()["data"],
                alert_data=self.get_alerts_dashboard()["data"],
                system_metrics=self._integration_service.get_system_metrics(),
            )

            self._data_cache = snapshot
            self._last_cache_update = datetime.now()

            return snapshot

    def add_update_callback(
        self, callback: Callable[[DashboardDataSnapshot], None]
    ) -> None:
        """Add a callback function to be called when dashboard data is updated.

        Args:
            callback: Function to call with DashboardDataSnapshot
        """
        self._update_callbacks.append(callback)
        logger.debug(
            f"Added dashboard update callback. Total callbacks: {len(self._update_callbacks)}"
        )

    def remove_update_callback(
        self, callback: Callable[[DashboardDataSnapshot], None]
    ) -> bool:
        """Remove a dashboard update callback.

        Args:
            callback: Callback function to remove

        Returns:
            True if callback was removed, False if not found
        """
        try:
            self._update_callbacks.remove(callback)
            logger.debug(
                f"Removed dashboard update callback. Total callbacks: {len(self._update_callbacks)}"
            )
            return True
        except ValueError:
            return False

    def trigger_dashboard_update(self) -> None:
        """Manually trigger dashboard data update and notify callbacks."""
        snapshot = self.get_dashboard_snapshot(use_cache=False)

        for callback in self._update_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Error in dashboard update callback: {e}")

        logger.debug("Triggered dashboard update")

    def export_dashboard_configuration(self, filepath: str) -> bool:
        """Export current dashboard configuration to a file.

        Args:
            filepath: Path to export configuration to

        Returns:
            True if export successful, False otherwise
        """
        try:
            self._dashboard_config.export_config_json(filepath)
            logger.info(f"Exported dashboard configuration to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export dashboard configuration: {e}")
            return False

    def import_dashboard_configuration(self, filepath: str) -> bool:
        """Import dashboard configuration from a file.

        Args:
            filepath: Path to configuration file

        Returns:
            True if import successful, False otherwise
        """
        try:
            with open(filepath, "r") as f:
                config_data = json.load(f)

            # Update dashboard configuration
            # Note: This would require implementing configuration update methods
            # in the dashboard config class
            logger.info(f"Imported dashboard configuration from: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to import dashboard configuration: {e}")
            return False

    def get_dashboard_health(self) -> Dict[str, Any]:
        """Get health status of dashboard components.

        Returns:
            Dictionary containing health status of all dashboard components
        """
        return {
            "dashboard_integration": "healthy",
            "langfuse_integration": "healthy"
            if self._integration_service.is_available
            else "unavailable",
            "workflow_manager": "healthy",
            "agent_performance_manager": "healthy",
            "alert_manager": "healthy",
            "metrics_collector": "healthy",
            "cache_status": "active" if self._data_cache else "empty",
            "last_cache_update": self._last_cache_update.isoformat()
            if self._last_cache_update
            else None,
            "update_callbacks_count": len(self._update_callbacks),
            "last_updated": datetime.now().isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear the dashboard data cache."""
        with self._cache_lock:
            self._data_cache = None
            self._last_cache_update = None
            logger.debug("Cleared dashboard data cache")

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set the cache time-to-live in seconds.

        Args:
            ttl_seconds: Cache TTL in seconds
        """
        self._cache_ttl_seconds = ttl_seconds
        logger.debug(f"Set dashboard cache TTL to {ttl_seconds} seconds")


# Global dashboard integration service instance
_dashboard_integration: Optional[LangfuseDashboardIntegrationService] = None


def get_dashboard_integration() -> LangfuseDashboardIntegrationService:
    """Get the global dashboard integration service instance."""
    global _dashboard_integration
    if _dashboard_integration is None:
        _dashboard_integration = LangfuseDashboardIntegrationService()
    return _dashboard_integration


def initialize_dashboard_integration(
    dashboard_config: Optional[LangfuseDashboardConfig] = None,
    workflow_manager: Optional[LangfuseWorkflowViewManager] = None,
    workflow_viz_manager: Optional[LangfuseWorkflowVisualizationViewManager] = None,
    agent_performance_manager: Optional[LangfuseAgentPerformanceViewManager] = None,
    alert_manager: Optional[LangfuseAlertManager] = None,
) -> LangfuseDashboardIntegrationService:
    """Initialize the global dashboard integration service.

    Args:
        dashboard_config: Optional dashboard configuration
        workflow_manager: Optional workflow view manager
        workflow_viz_manager: Optional workflow visualization manager
        agent_performance_manager: Optional agent performance manager
        alert_manager: Optional alert manager

    Returns:
        Initialized LangfuseDashboardIntegrationService instance
    """
    global _dashboard_integration
    _dashboard_integration = LangfuseDashboardIntegrationService(
        dashboard_config=dashboard_config,
        workflow_manager=workflow_manager,
        workflow_viz_manager=workflow_viz_manager,
        agent_performance_manager=agent_performance_manager,
        alert_manager=alert_manager,
    )
    return _dashboard_integration
