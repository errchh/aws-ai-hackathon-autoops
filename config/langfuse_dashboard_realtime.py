"""Real-time dashboard update system for Langfuse integration."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Set
from datetime import datetime, timedelta
from threading import Thread, Event
from dataclasses import dataclass, field

try:
    from fastapi import WebSocket

    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from config.langfuse_dashboard_integration import get_dashboard_integration
from config.langfuse_dashboard_data_aggregator import get_dashboard_data_aggregator
from config.langfuse_integration import get_langfuse_integration
from config.langfuse_alerting import get_alert_manager

logger = logging.getLogger(__name__)


@dataclass
class DashboardUpdateEvent:
    """Represents a dashboard update event."""

    event_type: str  # 'data_update', 'alert', 'metric_change', 'system_status'
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"  # 'low', 'normal', 'high', 'critical'


class RealTimeDashboardManager:
    """Manages real-time dashboard updates via polling and WebSocket connections."""

    def __init__(
        self,
        update_interval_seconds: int = 5,
        enable_websocket: bool = True,
        max_connections: int = 100,
    ):
        """Initialize the real-time dashboard manager.

        Args:
            update_interval_seconds: Interval for polling-based updates
            enable_websocket: Whether to enable WebSocket support
            max_connections: Maximum number of concurrent WebSocket connections
        """
        self._update_interval = update_interval_seconds
        self._enable_websocket = enable_websocket and WEBSOCKET_AVAILABLE
        self._max_connections = max_connections

        # Services
        self._dashboard_integration = get_dashboard_integration()
        self._data_aggregator = get_dashboard_data_aggregator()
        self._integration_service = get_langfuse_integration()
        self._alert_manager = get_alert_manager()

        # Update tracking
        self._last_update_time: Optional[datetime] = None
        self._update_count = 0
        self._is_running = False
        self._stop_event = Event()

        # WebSocket connections
        self._websocket_connections: Set[WebSocket] = set()
        self._connection_lock = asyncio.Lock() if WEBSOCKET_AVAILABLE else None

        # Update callbacks
        self._update_callbacks: List[Callable[[DashboardUpdateEvent], None]] = []

        # Background thread for polling
        self._background_thread: Optional[Thread] = None

        logger.info(
            f"Initialized real-time dashboard manager (WebSocket: {self._enable_websocket})"
        )

    def start(self) -> None:
        """Start the real-time dashboard update system."""
        if self._is_running:
            logger.warning("Real-time dashboard manager already running")
            return

        self._is_running = True
        self._stop_event.clear()

        # Start background polling thread
        self._background_thread = Thread(target=self._polling_loop, daemon=True)
        self._background_thread.start()

        # Start data aggregator if not already running
        if (
            not hasattr(self._data_aggregator, "_is_running")
            or not self._data_aggregator._is_running
        ):
            self._data_aggregator.start_auto_aggregation()

        logger.info("Started real-time dashboard update system")

    def stop(self) -> None:
        """Stop the real-time dashboard update system."""
        if not self._is_running:
            return

        self._is_running = False
        self._stop_event.set()

        # Stop data aggregator
        self._data_aggregator.stop_auto_aggregation()

        # Close all WebSocket connections
        if self._enable_websocket:
            asyncio.run(self._close_all_connections())

        # Wait for background thread to finish
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)

        logger.info("Stopped real-time dashboard update system")

    def _polling_loop(self) -> None:
        """Main polling loop for dashboard updates."""
        logger.debug("Started dashboard polling loop")

        while not self._stop_event.is_set():
            try:
                # Check for updates
                current_time = datetime.now()

                # Check if we need to trigger an update
                should_update = (
                    self._last_update_time is None
                    or (current_time - self._last_update_time).seconds
                    >= self._update_interval
                )

                if should_update:
                    self._trigger_dashboard_update()
                    self._last_update_time = current_time
                    self._update_count += 1

                # Check for new alerts
                self._check_for_alerts()

                # Check system health
                self._check_system_health()

            except Exception as e:
                logger.error(f"Error in dashboard polling loop: {e}")

            # Wait for next iteration
            self._stop_event.wait(self._update_interval)

        logger.debug("Dashboard polling loop stopped")

    def _trigger_dashboard_update(self) -> None:
        """Trigger a dashboard data update."""
        try:
            # Get fresh dashboard data
            snapshot = self._dashboard_integration.get_dashboard_snapshot(
                use_cache=False
            )

            # Create update event
            update_event = DashboardUpdateEvent(
                event_type="data_update",
                timestamp=datetime.now(),
                data={
                    "timestamp": snapshot.timestamp.isoformat(),
                    "overview": snapshot.overview_data,
                    "agent_performance": snapshot.agent_performance_data,
                    "workflows": snapshot.workflow_data,
                    "alerts": snapshot.alert_data,
                    "system_metrics": snapshot.system_metrics,
                },
                priority="normal",
            )

            # Notify callbacks
            for callback in self._update_callbacks:
                try:
                    callback(update_event)
                except Exception as e:
                    logger.error(f"Error in dashboard update callback: {e}")

            # Broadcast to WebSocket connections
            if self._enable_websocket:
                asyncio.run(self._broadcast_websocket_update(update_event))

            logger.debug("Triggered dashboard update")

        except Exception as e:
            logger.error(f"Failed to trigger dashboard update: {e}")

    def _check_for_alerts(self) -> None:
        """Check for new alerts and create update events."""
        try:
            # Get current alerts
            alert_data = self._alert_manager.get_dashboard_alert_data()
            active_alerts = alert_data.get("active_alerts", [])

            # Check for new critical alerts
            for alert in active_alerts:
                if alert.get("severity") == "critical":
                    alert_event = DashboardUpdateEvent(
                        event_type="alert",
                        timestamp=datetime.now(),
                        data={
                            "alert_id": alert.get("id"),
                            "severity": alert.get("severity"),
                            "title": alert.get("title"),
                            "message": alert.get("message"),
                            "timestamp": alert.get("timestamp"),
                        },
                        priority="critical",
                    )

                    # Notify callbacks
                    for callback in self._update_callbacks:
                        try:
                            callback(alert_event)
                        except Exception as e:
                            logger.error(f"Error in alert callback: {e}")

                    # Broadcast to WebSocket connections
                    if self._enable_websocket:
                        asyncio.run(self._broadcast_websocket_update(alert_event))

        except Exception as e:
            logger.error(f"Failed to check for alerts: {e}")

    def _check_system_health(self) -> None:
        """Check system health and create update events for significant changes."""
        try:
            # Get current system metrics
            system_metrics = self._integration_service.get_system_metrics()

            # Check for significant changes in key metrics
            error_rate = system_metrics.get("error_rate", 0.0)
            throughput = system_metrics.get("system_throughput", 0.0)

            # Create health event if metrics indicate issues
            if error_rate > 0.05 or throughput < 1.0:  # Thresholds for concern
                health_event = DashboardUpdateEvent(
                    event_type="system_status",
                    timestamp=datetime.now(),
                    data={
                        "error_rate": error_rate,
                        "throughput": throughput,
                        "status": "degraded"
                        if error_rate > 0.05 or throughput < 1.0
                        else "healthy",
                        "timestamp": datetime.now().isoformat(),
                    },
                    priority="high" if error_rate > 0.05 else "normal",
                )

                # Notify callbacks
                for callback in self._update_callbacks:
                    try:
                        callback(health_event)
                    except Exception as e:
                        logger.error(f"Error in health check callback: {e}")

                # Broadcast to WebSocket connections
                if self._enable_websocket:
                    asyncio.run(self._broadcast_websocket_update(health_event))

        except Exception as e:
            logger.error(f"Failed to check system health: {e}")

    async def _broadcast_websocket_update(self, event: DashboardUpdateEvent) -> None:
        """Broadcast update to all WebSocket connections."""
        if not self._enable_websocket:
            return

        async with self._connection_lock:
            disconnected_connections = set()

            for connection in self._websocket_connections:
                try:
                    update_message = {
                        "type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "data": event.data,
                        "priority": event.priority,
                    }
                    await connection.send_text(json.dumps(update_message))
                except Exception as e:
                    logger.debug(f"Failed to send WebSocket update: {e}")
                    disconnected_connections.add(connection)

            # Remove disconnected connections
            for connection in disconnected_connections:
                self._websocket_connections.discard(connection)

    async def _close_all_connections(self) -> None:
        """Close all WebSocket connections."""
        if not self._enable_websocket:
            return

        async with self._connection_lock:
            for connection in self._websocket_connections:
                try:
                    await connection.close()
                except Exception:
                    pass  # Connection might already be closed
            self._websocket_connections.clear()

    async def add_websocket_connection(self, websocket: WebSocket) -> bool:
        """Add a WebSocket connection for real-time updates.

        Args:
            websocket: WebSocket connection to add

        Returns:
            True if connection was added, False if at capacity
        """
        if not self._enable_websocket:
            return False

        async with self._connection_lock:
            if len(self._websocket_connections) >= self._max_connections:
                return False

            self._websocket_connections.add(websocket)

            # Send initial data
            try:
                snapshot = self._dashboard_integration.get_dashboard_snapshot()
                initial_data = {
                    "type": "initial_data",
                    "timestamp": snapshot.timestamp.isoformat(),
                    "data": {
                        "overview": snapshot.overview_data,
                        "agent_performance": snapshot.agent_performance_data,
                        "workflows": snapshot.workflow_data,
                        "alerts": snapshot.alert_data,
                        "system_metrics": snapshot.system_metrics,
                    },
                }
                await websocket.send_text(json.dumps(initial_data))
            except Exception as e:
                logger.error(f"Failed to send initial data to WebSocket: {e}")
                self._websocket_connections.discard(websocket)
                return False

            logger.debug(
                f"Added WebSocket connection. Total connections: {len(self._websocket_connections)}"
            )
            return True

    async def remove_websocket_connection(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if not self._enable_websocket:
            return

        async with self._connection_lock:
            self._websocket_connections.discard(websocket)
            logger.debug(
                f"Removed WebSocket connection. Total connections: {len(self._websocket_connections)}"
            )

    def add_update_callback(
        self, callback: Callable[[DashboardUpdateEvent], None]
    ) -> None:
        """Add a callback for dashboard update events."""
        self._update_callbacks.append(callback)
        logger.debug(
            f"Added dashboard update callback. Total callbacks: {len(self._update_callbacks)}"
        )

    def remove_update_callback(
        self, callback: Callable[[DashboardUpdateEvent], None]
    ) -> bool:
        """Remove a dashboard update callback."""
        try:
            self._update_callbacks.remove(callback)
            logger.debug(
                f"Removed dashboard update callback. Total callbacks: {len(self._update_callbacks)}"
            )
            return True
        except ValueError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the real-time dashboard manager."""
        return {
            "is_running": self._is_running,
            "update_interval_seconds": self._update_interval,
            "websocket_enabled": self._enable_websocket,
            "websocket_connections": len(self._websocket_connections)
            if self._enable_websocket
            else 0,
            "last_update_time": self._last_update_time.isoformat()
            if self._last_update_time
            else None,
            "total_updates": self._update_count,
            "update_callbacks_count": len(self._update_callbacks),
            "data_aggregator_running": getattr(
                self._data_aggregator, "_is_running", False
            ),
        }

    def force_update(self) -> None:
        """Force an immediate dashboard update."""
        self._trigger_dashboard_update()


# Global real-time dashboard manager instance
_realtime_manager: Optional[RealTimeDashboardManager] = None


def get_realtime_dashboard_manager() -> RealTimeDashboardManager:
    """Get the global real-time dashboard manager instance."""
    global _realtime_manager
    if _realtime_manager is None:
        _realtime_manager = RealTimeDashboardManager()
    return _realtime_manager


def initialize_realtime_dashboard_manager(
    update_interval_seconds: int = 5,
    enable_websocket: bool = True,
    max_connections: int = 100,
) -> RealTimeDashboardManager:
    """Initialize the global real-time dashboard manager.

    Args:
        update_interval_seconds: Interval for polling-based updates
        enable_websocket: Whether to enable WebSocket support
        max_connections: Maximum number of concurrent WebSocket connections

    Returns:
        Initialized RealTimeDashboardManager instance
    """
    global _realtime_manager
    _realtime_manager = RealTimeDashboardManager(
        update_interval_seconds=update_interval_seconds,
        enable_websocket=enable_websocket,
        max_connections=max_connections,
    )
    return _realtime_manager


# WebSocket handler for FastAPI integration
async def handle_websocket_connection(
    websocket: WebSocket, manager: RealTimeDashboardManager
):
    """Handle WebSocket connection for real-time dashboard updates.

    Args:
        websocket: WebSocket connection
        manager: RealTimeDashboardManager instance
    """
    if not WEBSOCKET_AVAILABLE:
        await websocket.close(code=1000, reason="WebSocket support not available")
        return

    # Add connection
    success = await manager.add_websocket_connection(websocket)
    if not success:
        await websocket.close(code=1013, reason="Too many connections")
        return

    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Receive message (for ping/pong or other client messages)
            message = await websocket.receive_text()

            # Handle ping/pong
            if message == "ping":
                await websocket.send_text("pong")
            elif message == "request_update":
                # Client requesting immediate update
                manager.force_update()

    except Exception as e:
        logger.debug(f"WebSocket connection closed: {e}")
    finally:
        await manager.remove_websocket_connection(websocket)
