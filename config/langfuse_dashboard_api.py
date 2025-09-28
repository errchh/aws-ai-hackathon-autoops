"""API endpoints for Langfuse dashboard integration."""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


# Pydantic models for API responses
class DashboardOverviewResponse(BaseModel):
    """Response model for dashboard overview data."""

    view_id: str
    view_config: Dict[str, Any]
    data: Dict[str, Any]
    last_updated: str


class AgentPerformanceResponse(BaseModel):
    """Response model for agent performance data."""

    view_id: str
    view_config: Dict[str, Any]
    data: Dict[str, Any]
    last_updated: str


class WorkflowDashboardResponse(BaseModel):
    """Response model for workflow dashboard data."""

    view_id: str
    view_config: Dict[str, Any]
    data: Dict[str, Any]
    last_updated: str


class AlertsDashboardResponse(BaseModel):
    """Response model for alerts dashboard data."""

    view_id: str
    view_config: Dict[str, Any]
    data: Dict[str, Any]
    last_updated: str


class CompleteDashboardResponse(BaseModel):
    """Response model for complete dashboard data."""

    overview: DashboardOverviewResponse
    agent_performance: AgentPerformanceResponse
    workflow_visualization: WorkflowDashboardResponse
    alerts: AlertsDashboardResponse
    dashboard_config: Dict[str, Any]
    last_updated: str


class WorkflowDetailResponse(BaseModel):
    """Response model for workflow detail data."""

    workflow_id: str
    detail_view: Dict[str, Any]
    flow_view: Dict[str, Any]
    last_updated: str


class AgentDetailResponse(BaseModel):
    """Response model for agent detail data."""

    agent_id: str
    summary: Dict[str, Any]
    alerts: List[Dict[str, Any]]
    historical_data: List[Dict[str, Any]]
    last_updated: str


class DashboardHealthResponse(BaseModel):
    """Response model for dashboard health status."""

    dashboard_integration: str
    langfuse_integration: str
    workflow_manager: str
    agent_performance_manager: str
    alert_manager: str
    metrics_collector: str
    cache_status: str
    last_cache_update: Optional[str]
    update_callbacks_count: int
    last_updated: str


class ExportConfigRequest(BaseModel):
    """Request model for configuration export."""

    filepath: str = Field(..., description="File path to export configuration to")


class ImportConfigRequest(BaseModel):
    """Request model for configuration import."""

    filepath: str = Field(..., description="File path to import configuration from")


# Global dashboard integration service
_dashboard_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager."""
    global _dashboard_service
    # Initialize dashboard service on startup
    try:
        from config.langfuse_dashboard_integration import get_dashboard_integration

        _dashboard_service = get_dashboard_integration()
        logger.info("Dashboard API service initialized")
    except ImportError as e:
        logger.warning(f"Could not initialize dashboard service: {e}")
    yield
    # Cleanup on shutdown
    logger.info("Dashboard API service shutting down")


# Create FastAPI application
app = FastAPI(
    title="Langfuse Dashboard API",
    description="API for Langfuse workflow visualization dashboard",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_dashboard_service():
    """Get the dashboard integration service instance."""
    if _dashboard_service is None:
        raise HTTPException(status_code=500, detail="Dashboard service not initialized")
    return _dashboard_service


@app.get("/health", response_model=DashboardHealthResponse)
async def get_health():
    """Get dashboard health status."""
    try:
        service = get_dashboard_service()
        health_data = service.get_dashboard_health()
        return DashboardHealthResponse(**health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/dashboard/overview", response_model=DashboardOverviewResponse)
async def get_dashboard_overview(
    view_id: str = Query("overview", description="Dashboard view ID"),
):
    """Get dashboard overview data."""
    try:
        service = get_dashboard_service()
        overview_data = service.get_dashboard_overview(view_id)
        if "error" in overview_data:
            raise HTTPException(status_code=404, detail=overview_data["error"])
        return DashboardOverviewResponse(**overview_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dashboard overview: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve dashboard overview"
        )


@app.get("/dashboard/agent-performance", response_model=AgentPerformanceResponse)
async def get_agent_performance_dashboard(
    view_id: str = Query("agent_performance", description="Agent performance view ID"),
):
    """Get agent performance dashboard data."""
    try:
        service = get_dashboard_service()
        performance_data = service.get_agent_performance_dashboard(view_id)
        if "error" in performance_data:
            raise HTTPException(status_code=404, detail=performance_data["error"])
        return AgentPerformanceResponse(**performance_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent performance dashboard: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve agent performance dashboard"
        )


@app.get("/dashboard/workflows", response_model=WorkflowDashboardResponse)
async def get_workflow_dashboard(
    view_id: str = Query(
        "workflow_visualization", description="Workflow visualization view ID"
    ),
):
    """Get workflow dashboard data."""
    try:
        service = get_dashboard_service()
        workflow_data = service.get_workflow_dashboard(view_id)
        if "error" in workflow_data:
            raise HTTPException(status_code=404, detail=workflow_data["error"])
        return WorkflowDashboardResponse(**workflow_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow dashboard: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve workflow dashboard"
        )


@app.get("/dashboard/alerts", response_model=AlertsDashboardResponse)
async def get_alerts_dashboard(
    view_id: str = Query("alerts", description="Alerts view ID"),
):
    """Get alerts dashboard data."""
    try:
        service = get_dashboard_service()
        alerts_data = service.get_alerts_dashboard(view_id)
        if "error" in alerts_data:
            raise HTTPException(status_code=404, detail=alerts_data["error"])
        return AlertsDashboardResponse(**alerts_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alerts dashboard: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve alerts dashboard"
        )


@app.get("/dashboard/complete", response_model=CompleteDashboardResponse)
async def get_complete_dashboard():
    """Get complete dashboard data for all views."""
    try:
        service = get_dashboard_service()
        complete_data = service.get_complete_dashboard_data()
        return CompleteDashboardResponse(**complete_data)
    except Exception as e:
        logger.error(f"Failed to get complete dashboard data: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve complete dashboard data"
        )


@app.get("/workflows/{workflow_id}", response_model=WorkflowDetailResponse)
async def get_workflow_detail(workflow_id: str):
    """Get detailed information for a specific workflow."""
    try:
        service = get_dashboard_service()
        workflow_data = service.get_workflow_detail(workflow_id)
        if not workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return WorkflowDetailResponse(**workflow_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow detail for {workflow_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve workflow detail"
        )


@app.get("/agents/{agent_id}", response_model=AgentDetailResponse)
async def get_agent_detail(agent_id: str):
    """Get detailed information for a specific agent."""
    try:
        service = get_dashboard_service()
        agent_data = service.get_agent_detail(agent_id)
        if not agent_data:
            raise HTTPException(status_code=404, detail="Agent not found")
        return AgentDetailResponse(**agent_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent detail for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve agent detail")


@app.post("/dashboard/trigger-update")
async def trigger_dashboard_update(background_tasks: BackgroundTasks):
    """Manually trigger dashboard data update."""
    try:
        service = get_dashboard_service()
        background_tasks.add_task(service.trigger_dashboard_update)
        return {
            "message": "Dashboard update triggered",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to trigger dashboard update: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to trigger dashboard update"
        )


@app.post("/dashboard/export-config")
async def export_dashboard_config(request: ExportConfigRequest):
    """Export dashboard configuration to a file."""
    try:
        service = get_dashboard_service()
        success = service.export_dashboard_configuration(request.filepath)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to export configuration"
            )
        return {
            "message": "Configuration exported successfully",
            "filepath": request.filepath,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export dashboard configuration: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to export dashboard configuration"
        )


@app.post("/dashboard/import-config")
async def import_dashboard_config(request: ImportConfigRequest):
    """Import dashboard configuration from a file."""
    try:
        service = get_dashboard_service()
        success = service.import_dashboard_configuration(request.filepath)
        if not success:
            raise HTTPException(
                status_code=500, detail="Failed to import configuration"
            )
        return {
            "message": "Configuration imported successfully",
            "filepath": request.filepath,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to import dashboard configuration: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to import dashboard configuration"
        )


@app.post("/dashboard/clear-cache")
async def clear_dashboard_cache():
    """Clear the dashboard data cache."""
    try:
        service = get_dashboard_service()
        service.clear_cache()
        return {
            "message": "Dashboard cache cleared",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to clear dashboard cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear dashboard cache")


@app.get("/dashboard/config")
async def get_dashboard_config():
    """Get current dashboard configuration."""
    try:
        service = get_dashboard_service()
        config_data = service._dashboard_config.get_dashboard_config()
        return {"config": config_data, "last_updated": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Failed to get dashboard configuration: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve dashboard configuration"
        )


@app.get("/dashboard/snapshot")
async def get_dashboard_snapshot(
    use_cache: bool = Query(True, description="Whether to use cached data"),
):
    """Get a complete snapshot of dashboard data."""
    try:
        service = get_dashboard_service()
        snapshot = service.get_dashboard_snapshot(use_cache=use_cache)

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "overview_data": snapshot.overview_data,
            "agent_performance_data": snapshot.agent_performance_data,
            "workflow_data": snapshot.workflow_data,
            "alert_data": snapshot.alert_data,
            "system_metrics": snapshot.system_metrics,
            "cache_used": use_cache and service._data_cache is not None,
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard snapshot: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve dashboard snapshot"
        )


# Optional: Add WebSocket support for real-time updates
try:
    from fastapi import WebSocket, WebSocketDisconnect
    import asyncio

    # WebSocket connection manager
    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []

        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket):
            self.active_connections.remove(websocket)

        async def broadcast(self, message: str):
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except:
                    self.disconnect(connection)

    manager = ConnectionManager()

    @app.websocket("/ws/dashboard-updates")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time dashboard updates."""
        await manager.connect(websocket)
        try:
            service = get_dashboard_service()

            # Send initial data
            snapshot = service.get_dashboard_snapshot()
            initial_data = {
                "type": "initial_data",
                "data": {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "overview": snapshot.overview_data,
                    "agent_performance": snapshot.agent_performance_data,
                    "workflows": snapshot.workflow_data,
                    "alerts": snapshot.alert_data,
                },
            }
            await websocket.send_text(json.dumps(initial_data))

            # Set up callback for updates
            def update_callback(snapshot):
                async def send_update():
                    update_data = {
                        "type": "update",
                        "data": {
                            "timestamp": snapshot.timestamp.isoformat(),
                            "overview": snapshot.overview_data,
                            "agent_performance": snapshot.agent_performance_data,
                            "workflows": snapshot.workflow_data,
                            "alerts": snapshot.alert_data,
                        },
                    }
                    await manager.broadcast(json.dumps(update_data))

                asyncio.create_task(send_update())

            service.add_update_callback(update_callback)

            # Keep connection alive
            while True:
                data = await websocket.receive_text()
                # Handle ping/pong or other client messages if needed

        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            manager.disconnect(websocket)

except ImportError:
    logger.warning(
        "WebSocket support not available. Install 'fastapi[websockets]' for real-time updates."
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
