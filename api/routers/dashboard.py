"""
Dashboard API endpoints and WebSocket handlers for the autoops retail optimization system.

This module provides REST API endpoints and WebSocket connections specifically
designed for the React.js dashboard, including real-time updates, agent status
monitoring, and manual intervention controls.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    status,
    Depends,
)
from pydantic import BaseModel, Field

from models.core import SystemMetrics, AgentDecision, MarketEvent, EventType, ActionType


router = APIRouter()


# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.agent_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, connection_type: str = "general"):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)

        if connection_type.startswith("agent_"):
            agent_id = connection_type.replace("agent_", "")
            if agent_id not in self.agent_connections:
                self.agent_connections[agent_id] = set()
            self.agent_connections[agent_id].add(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)

        # Remove from agent-specific connections
        for agent_id, connections in self.agent_connections.items():
            connections.discard(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception:
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        """Broadcast a message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_agent_subscribers(self, agent_id: str, message: str):
        """Broadcast a message to all clients subscribed to a specific agent."""
        if agent_id not in self.agent_connections:
            return

        disconnected = set()
        for connection in self.agent_connections[agent_id]:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.agent_connections[agent_id].discard(connection)


# Global connection manager instance
manager = ConnectionManager()


# Response Models
class AgentStatus(BaseModel):
    """Agent status information for dashboard display."""

    agent_id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    status: str = Field(
        ..., description="Current status (active, idle, error, offline)"
    )
    current_activity: Optional[str] = Field(
        None, description="Current activity description"
    )
    last_decision_time: Optional[datetime] = Field(
        None, description="Timestamp of last decision"
    )
    decisions_count: int = Field(default=0, description="Total decisions made")
    success_rate: float = Field(default=0.0, description="Success rate percentage")
    avg_response_time: float = Field(
        default=0.0, description="Average response time in seconds"
    )


class DashboardMetrics(BaseModel):
    """Dashboard-specific metrics for real-time visualization."""

    timestamp: datetime = Field(..., description="Metrics timestamp")
    system_metrics: SystemMetrics = Field(..., description="Core system metrics")
    agent_statuses: List[AgentStatus] = Field(..., description="Status of all agents")
    active_alerts_count: int = Field(default=0, description="Number of active alerts")
    recent_decisions_count: int = Field(
        default=0, description="Recent decisions in last hour"
    )


class AlertMessage(BaseModel):
    """Alert message for dashboard notifications."""

    id: str = Field(..., description="Unique alert identifier")
    timestamp: datetime = Field(..., description="Alert timestamp")
    severity: str = Field(
        ..., description="Alert severity (info, warning, error, critical)"
    )
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    agent_id: Optional[str] = Field(None, description="Related agent ID if applicable")
    auto_dismiss: bool = Field(
        default=False, description="Whether alert auto-dismisses"
    )


class InterventionRequest(BaseModel):
    """Manual intervention request from dashboard."""

    agent_id: str = Field(..., description="Target agent ID")
    action: str = Field(
        ..., description="Intervention action (pause, resume, reset, override)"
    )
    parameters: Optional[Dict] = Field(None, description="Action-specific parameters")
    reason: str = Field(..., description="Reason for intervention")


class InterventionResponse(BaseModel):
    """Response to manual intervention request."""

    success: bool = Field(..., description="Whether intervention was successful")
    message: str = Field(..., description="Response message")
    agent_id: str = Field(..., description="Target agent ID")
    timestamp: datetime = Field(..., description="Intervention timestamp")


class DashboardDecision(BaseModel):
    """Dashboard decision display in 4-column format."""

    trigger_detected: str = Field(..., description="Specific trigger that was detected")
    agent_decision_action: str = Field(
        ..., description="Agent decision and action taken"
    )
    value_before: str = Field(..., description="Value before the agent's change")
    value_after: str = Field(..., description="Value after the agent's change")
    timestamp: datetime = Field(..., description="Decision timestamp")
    function_name: str = Field(..., description="Name of the agent function executed")
    confidence_score: float = Field(..., description="Confidence score of the decision")


# Dashboard API Endpoints
@router.get("/agents/status", response_model=List[AgentStatus])
async def get_agents_status() -> List[AgentStatus]:
    """
    Get current status of all agents for dashboard display.

    This endpoint provides comprehensive status information for all agents
    including their current activities, performance metrics, and health status.
    """
    try:
        # Simulate agent status data (in real implementation, this would query agent states)
        current_time = datetime.now(timezone.utc)

        agents_status = [
            AgentStatus(
                agent_id="pricing_agent",
                name="Pricing Agent",
                status="active",
                current_activity="Analyzing demand elasticity for SKU123456",
                last_decision_time=current_time,
                decisions_count=45,
                success_rate=92.3,
                avg_response_time=1.2,
            ),
            AgentStatus(
                agent_id="inventory_agent",
                name="Inventory Agent",
                status="active",
                current_activity="Forecasting demand for beverages category",
                last_decision_time=current_time,
                decisions_count=38,
                success_rate=89.7,
                avg_response_time=2.1,
            ),
            AgentStatus(
                agent_id="promotion_agent",
                name="Promotion Agent",
                status="idle",
                current_activity=None,
                last_decision_time=current_time,
                decisions_count=22,
                success_rate=94.1,
                avg_response_time=1.8,
            ),
        ]

        return agents_status

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent status: {str(e)}",
        )


@router.get("/metrics/current", response_model=DashboardMetrics)
async def get_current_dashboard_metrics() -> DashboardMetrics:
    """
    Get current dashboard metrics including system performance and agent status.

    This endpoint provides a comprehensive view of system health and performance
    specifically formatted for dashboard visualization.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Get system metrics
        system_metrics = SystemMetrics(
            total_revenue=125000.50,
            total_profit=45000.25,
            inventory_turnover=8.5,
            stockout_incidents=3,
            waste_reduction_percentage=15.2,
            price_optimization_score=0.87,
            promotion_effectiveness=0.72,
            agent_collaboration_score=0.91,
            decision_count=247,
            response_time_avg=1.35,
        )

        # Get agent statuses
        agent_statuses = await get_agents_status()

        return DashboardMetrics(
            timestamp=current_time,
            system_metrics=system_metrics,
            agent_statuses=agent_statuses,
            active_alerts_count=2,
            recent_decisions_count=15,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard metrics: {str(e)}",
        )


@router.get("/decisions/recent", response_model=List[AgentDecision])
async def get_recent_decisions(
    limit: int = 20, agent_id: Optional[str] = None
) -> List[AgentDecision]:
    """
    Get recent agent decisions for dashboard timeline display.

    This endpoint provides a chronological list of recent agent decisions
    with full context and rationale for dashboard visualization.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate recent decisions (in real implementation, this would query the database)
        recent_decisions = [
            AgentDecision(
                agent_id="pricing_agent",
                timestamp=current_time,
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={
                    "product_id": "SKU123456",
                    "old_price": 24.99,
                    "new_price": 22.99,
                    "discount_percentage": 8.0,
                },
                rationale="Applied markdown due to high inventory levels and slow movement. Expected to increase demand by 15% while maintaining profit margins.",
                confidence_score=0.85,
                expected_outcome={
                    "demand_increase": 0.15,
                    "profit_impact": 0.02,
                    "inventory_reduction": 0.25,
                },
            ),
            AgentDecision(
                agent_id="inventory_agent",
                timestamp=current_time,
                action_type=ActionType.STOCK_ALERT,
                parameters={
                    "product_id": "SKU789012",
                    "current_stock": 8,
                    "recommended_quantity": 50,
                    "urgency": "high",
                },
                rationale="Stock level below reorder point with increasing demand trend. Recommended immediate restocking to prevent stockout.",
                confidence_score=0.92,
                expected_outcome={
                    "stockout_prevention": True,
                    "service_level": 0.98,
                    "carrying_cost_increase": 0.05,
                },
            ),
            AgentDecision(
                agent_id="promotion_agent",
                timestamp=current_time,
                action_type=ActionType.BUNDLE_RECOMMENDATION,
                parameters={
                    "anchor_product": "SKU123456",
                    "bundle_products": ["SKU345678", "SKU901234"],
                    "discount": 0.10,
                    "duration_hours": 48,
                },
                rationale="Created coffee + chocolate + pasta bundle based on purchase affinity analysis. Expected to increase basket size and move slow inventory.",
                confidence_score=0.78,
                expected_outcome={
                    "basket_size_increase": 0.20,
                    "cross_sell_rate": 0.35,
                    "inventory_turnover": 0.15,
                },
            ),
        ]

        # Filter by agent_id if specified
        if agent_id:
            recent_decisions = [d for d in recent_decisions if d.agent_id == agent_id]

        # Apply limit
        return recent_decisions[:limit]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve recent decisions: {str(e)}",
        )


@router.get("/alerts/active", response_model=List[AlertMessage])
async def get_active_alerts() -> List[AlertMessage]:
    """
    Get active system alerts for dashboard notifications.

    This endpoint provides current system alerts and notifications
    that require attention or provide important status updates.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate active alerts (in real implementation, this would query alert system)
        active_alerts = [
            AlertMessage(
                id=str(uuid4()),
                timestamp=current_time,
                severity="warning",
                title="Low Stock Alert",
                message="SKU789012 (Organic Green Tea) is below reorder point with only 8 units remaining",
                agent_id="inventory_agent",
                auto_dismiss=False,
            ),
            AlertMessage(
                id=str(uuid4()),
                timestamp=current_time,
                severity="info",
                title="Price Optimization Complete",
                message="Pricing Agent successfully optimized prices for 5 products in Beverages category",
                agent_id="pricing_agent",
                auto_dismiss=True,
            ),
        ]

        return active_alerts

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve active alerts: {str(e)}",
        )


@router.post("/agents/{agent_id}/intervention", response_model=InterventionResponse)
async def manual_intervention(
    agent_id: str, request: InterventionRequest
) -> InterventionResponse:
    """
    Perform manual intervention on a specific agent.

    This endpoint allows dashboard users to manually intervene in agent
    operations for testing, debugging, or emergency situations.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Validate agent_id
        valid_agents = ["pricing_agent", "inventory_agent", "promotion_agent"]
        if agent_id not in valid_agents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent {agent_id} not found",
            )

        # Validate action
        valid_actions = ["pause", "resume", "reset", "override"]
        if request.action not in valid_actions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {request.action}. Must be one of {valid_actions}",
            )

        # Simulate intervention processing (in real implementation, this would interact with agents)
        success = True
        message = f"Successfully {request.action}d {agent_id}"

        # Broadcast intervention to WebSocket clients
        intervention_update = {
            "type": "agent_intervention",
            "agent_id": agent_id,
            "action": request.action,
            "timestamp": current_time.isoformat(),
            "reason": request.reason,
        }
        await manager.broadcast(json.dumps(intervention_update))

        return InterventionResponse(
            success=success, message=message, agent_id=agent_id, timestamp=current_time
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform intervention: {str(e)}",
        )


# WebSocket Endpoints
@router.websocket("/ws/live-updates")
async def websocket_live_updates(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates.

    This endpoint provides live updates of system metrics, agent status,
    and other real-time information for dashboard visualization.
    """
    await manager.connect(websocket, "general")

    try:
        # Send initial data
        initial_data = {
            "type": "connection_established",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Connected to live updates stream",
        }
        await manager.send_personal_message(json.dumps(initial_data), websocket)

        # Keep connection alive and send periodic updates
        while True:
            # Send periodic system updates (every 5 seconds)
            await asyncio.sleep(5)

            current_metrics = await get_current_dashboard_metrics()
            update_data = {
                "type": "metrics_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": current_metrics.dict(),
            }

            await manager.send_personal_message(json.dumps(update_data), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/agents/{agent_id}/status")
async def websocket_agent_status(websocket: WebSocket, agent_id: str):
    """
    WebSocket endpoint for real-time agent-specific status updates.

    This endpoint provides live updates for a specific agent's status,
    activities, and decisions for detailed monitoring.
    """
    # Validate agent_id
    valid_agents = ["pricing_agent", "inventory_agent", "promotion_agent"]
    if agent_id not in valid_agents:
        await websocket.close(code=4004, reason=f"Invalid agent ID: {agent_id}")
        return

    await manager.connect(websocket, f"agent_{agent_id}")

    try:
        # Send initial agent status
        agents_status = await get_agents_status()
        agent_status = next((a for a in agents_status if a.agent_id == agent_id), None)

        if agent_status:
            initial_data = {
                "type": "agent_status",
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": agent_status.dict(),
            }
            await manager.send_personal_message(json.dumps(initial_data), websocket)

        # Keep connection alive and send periodic agent updates
        while True:
            await asyncio.sleep(3)

            # Send agent-specific updates
            agents_status = await get_agents_status()
            agent_status = next(
                (a for a in agents_status if a.agent_id == agent_id), None
            )

            if agent_status:
                update_data = {
                    "type": "agent_status_update",
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": agent_status.dict(),
                }
                await manager.send_personal_message(json.dumps(update_data), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Agent WebSocket error: {e}")
        manager.disconnect(websocket)


@router.websocket("/ws/metrics/real-time")
async def websocket_real_time_metrics(websocket: WebSocket):
    """
    WebSocket endpoint for real-time metrics streaming.

    This endpoint provides high-frequency updates of key performance
    indicators and system metrics for real-time dashboard charts.
    """
    await manager.connect(websocket, "metrics")

    try:
        # Send initial metrics
        current_metrics = await get_current_dashboard_metrics()
        initial_data = {
            "type": "metrics_stream",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": current_metrics.system_metrics.dict(),
        }
        await manager.send_personal_message(json.dumps(initial_data), websocket)

        # Stream metrics updates (every 2 seconds for real-time charts)
        while True:
            await asyncio.sleep(2)

            current_metrics = await get_current_dashboard_metrics()
            metrics_data = {
                "type": "metrics_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": current_metrics.system_metrics.dict(),
            }

            await manager.send_personal_message(json.dumps(metrics_data), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Metrics WebSocket error: {e}")
        manager.disconnect(websocket)


# Utility function to broadcast system events
async def broadcast_system_event(event_type: str, data: Dict):
    """
    Utility function to broadcast system events to all connected clients.

    This function can be called from other parts of the system to notify
    the dashboard of important events in real-time.
    """
    event_message = {
        "type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    await manager.broadcast(json.dumps(event_message))


@router.get("/decisions/pricing", response_model=List[DashboardDecision])
async def get_pricing_decisions(limit: int = 10) -> List[DashboardDecision]:
    """
    Get recent pricing agent decisions for dashboard display.

    This endpoint provides pricing decisions in the 4-column format
    showing triggers, actions, and value changes.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate pricing decisions covering all pricing agent functions
        pricing_decisions = [
            DashboardDecision(
                trigger_detected="High inventory levels detected for SKU123456",
                agent_decision_action="Applied markdown strategy (8% discount)",
                value_before="$24.99",
                value_after="$22.99",
                timestamp=current_time,
                function_name="apply_markdown_strategy",
                confidence_score=0.85,
            ),
            DashboardDecision(
                trigger_detected="Demand elasticity analysis completed",
                agent_decision_action="Calculated optimal price based on elasticity",
                value_before="$25.00",
                value_after="$23.50",
                timestamp=current_time,
                function_name="calculate_optimal_price",
                confidence_score=0.92,
            ),
            DashboardDecision(
                trigger_detected="Competitor price monitoring alert",
                agent_decision_action="Retrieved competitor pricing data",
                value_before="N/A",
                value_after="Competitor avg: $22.75",
                timestamp=current_time,
                function_name="get_competitor_prices",
                confidence_score=0.88,
            ),
            DashboardDecision(
                trigger_detected="Price change evaluation requested",
                agent_decision_action="Evaluated price impact on demand",
                value_before="Current demand: 100 units/day",
                value_after="Projected demand: 115 units/day",
                timestamp=current_time,
                function_name="evaluate_price_impact",
                confidence_score=0.79,
            ),
            DashboardDecision(
                trigger_detected="Pricing history analysis",
                agent_decision_action="Retrieved historical pricing data",
                value_before="N/A",
                value_after="Historical avg: $24.20",
                timestamp=current_time,
                function_name="retrieve_pricing_history",
                confidence_score=0.95,
            ),
            DashboardDecision(
                trigger_detected="Market event: Seasonal demand spike",
                agent_decision_action="Made pricing decision for SKU789012",
                value_before="$19.99",
                value_after="$21.99",
                timestamp=current_time,
                function_name="make_pricing_decision",
                confidence_score=0.87,
            ),
            DashboardDecision(
                trigger_detected="Decision outcome tracking",
                agent_decision_action="Updated decision outcome metrics",
                value_before="Success rate: 91%",
                value_after="Success rate: 92.3%",
                timestamp=current_time,
                function_name="update_decision_outcome",
                confidence_score=0.96,
            ),
            DashboardDecision(
                trigger_detected="Demand pattern analysis",
                agent_decision_action="Analyzed demand elasticity for category",
                value_before="Elasticity: Unknown",
                value_after="Elasticity: -1.2 (elastic)",
                timestamp=current_time,
                function_name="analyze_demand_elasticity",
                confidence_score=0.83,
            ),
        ]

        return pricing_decisions[:limit]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve pricing decisions: {str(e)}",
        )


@router.get("/decisions/inventory", response_model=List[DashboardDecision])
async def get_inventory_decisions(limit: int = 10) -> List[DashboardDecision]:
    """
    Get recent inventory agent decisions for dashboard display.

    This endpoint provides inventory decisions in the 4-column format
    showing triggers, actions, and value changes.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate inventory decisions covering all inventory agent functions
        inventory_decisions = [
            DashboardDecision(
                trigger_detected="Low stock alert for SKU789012",
                agent_decision_action="Generated restock alert (high urgency)",
                value_before="8 units",
                value_after="Recommended: 50 units",
                timestamp=current_time,
                function_name="generate_restock_alert",
                confidence_score=0.92,
            ),
            DashboardDecision(
                trigger_detected="Demand forecasting request",
                agent_decision_action="Forecasted probabilistic demand",
                value_before="Current demand: 20/day",
                value_after="Projected: 35/day (±5)",
                timestamp=current_time,
                function_name="forecast_demand_probabilistic",
                confidence_score=0.89,
            ),
            DashboardDecision(
                trigger_detected="Safety stock calculation",
                agent_decision_action="Calculated safety buffer for SKU345678",
                value_before="Min stock: 10",
                value_after="Safety buffer: 25 units",
                timestamp=current_time,
                function_name="calculate_safety_buffer",
                confidence_score=0.94,
            ),
            DashboardDecision(
                trigger_detected="Inventory analysis completed",
                agent_decision_action="Identified slow-moving inventory",
                value_before="Turnover: 2.1x",
                value_after="Slow movers: SKU111111, SKU222222",
                timestamp=current_time,
                function_name="identify_slow_moving_inventory",
                confidence_score=0.87,
            ),
            DashboardDecision(
                trigger_detected="Demand pattern monitoring",
                agent_decision_action="Analyzed demand patterns for beverages",
                value_before="Pattern: Stable",
                value_after="Pattern: Increasing trend",
                timestamp=current_time,
                function_name="analyze_demand_patterns",
                confidence_score=0.91,
            ),
            DashboardDecision(
                trigger_detected="Inventory history review",
                agent_decision_action="Retrieved inventory history data",
                value_before="N/A",
                value_after="Historical avg stock: 45 units",
                timestamp=current_time,
                function_name="retrieve_inventory_history",
                confidence_score=0.96,
            ),
            DashboardDecision(
                trigger_detected="Stock optimization opportunity",
                agent_decision_action="Made inventory decision for SKU901234",
                value_before="Current stock: 60",
                value_after="Optimized stock: 40 units",
                timestamp=current_time,
                function_name="make_inventory_decision",
                confidence_score=0.85,
            ),
            DashboardDecision(
                trigger_detected="Decision performance tracking",
                agent_decision_action="Updated inventory decision outcomes",
                value_before="Accuracy: 88%",
                value_after="Accuracy: 89.7%",
                timestamp=current_time,
                function_name="update_decision_outcome",
                confidence_score=0.97,
            ),
        ]

        return inventory_decisions[:limit]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve inventory decisions: {str(e)}",
        )


@router.get("/decisions/promotion", response_model=List[DashboardDecision])
async def get_promotion_decisions(limit: int = 10) -> List[DashboardDecision]:
    """
    Get recent promotion agent decisions for dashboard display.

    This endpoint provides promotion decisions in the 4-column format
    showing triggers, actions, and value changes.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate promotion decisions covering all promotion agent functions
        promotion_decisions = [
            DashboardDecision(
                trigger_detected="High inventory + low demand detected",
                agent_decision_action="Created flash sale for SKU123456",
                value_before="Regular price: $24.99",
                value_after="Flash sale: $19.99 (20% off)",
                timestamp=current_time,
                function_name="create_flash_sale",
                confidence_score=0.82,
            ),
            DashboardDecision(
                trigger_detected="Purchase affinity analysis completed",
                agent_decision_action="Generated bundle recommendation",
                value_before="Individual prices",
                value_after="Bundle: Coffee + Chocolate + Pasta (10% off)",
                timestamp=current_time,
                function_name="generate_bundle_recommendation",
                confidence_score=0.78,
            ),
            DashboardDecision(
                trigger_detected="Social media monitoring",
                agent_decision_action="Analyzed social sentiment for promotion",
                value_before="Sentiment: Neutral",
                value_after="Sentiment: Positive (78%)",
                timestamp=current_time,
                function_name="analyze_social_sentiment",
                confidence_score=0.85,
            ),
            DashboardDecision(
                trigger_detected="Promotional opportunity identified",
                agent_decision_action="Scheduled promotional campaign",
                value_before="No active promotion",
                value_after="48-hour bundle campaign",
                timestamp=current_time,
                function_name="schedule_promotional_campaign",
                confidence_score=0.91,
            ),
            DashboardDecision(
                trigger_detected="Campaign performance review",
                agent_decision_action="Evaluated campaign effectiveness",
                value_before="Campaign running",
                value_after="Effectiveness: 72% (basket size +20%)",
                timestamp=current_time,
                function_name="evaluate_campaign_effectiveness",
                confidence_score=0.88,
            ),
            DashboardDecision(
                trigger_detected="Cross-agent coordination request",
                agent_decision_action="Coordinated with pricing agent",
                value_before="Independent pricing",
                value_after="Coordinated discount strategy",
                timestamp=current_time,
                function_name="coordinate_with_pricing_agent",
                confidence_score=0.93,
            ),
            DashboardDecision(
                trigger_detected="Promotion inventory check",
                agent_decision_action="Validated inventory availability",
                value_before="Stock check pending",
                value_after="All bundle items available",
                timestamp=current_time,
                function_name="validate_inventory_availability",
                confidence_score=0.96,
            ),
            DashboardDecision(
                trigger_detected="Promotion history analysis",
                agent_decision_action="Retrieved promotion history",
                value_before="N/A",
                value_after="Historical success rate: 94.1%",
                timestamp=current_time,
                function_name="retrieve_promotion_history",
                confidence_score=0.95,
            ),
        ]

        return promotion_decisions[:limit]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve promotion decisions: {str(e)}",
        )


@router.get("/decisions/orchestrator", response_model=List[DashboardDecision])
async def get_orchestrator_decisions(limit: int = 10) -> List[DashboardDecision]:
    """
    Get recent orchestrator decisions for dashboard display.

    This endpoint provides orchestrator decisions in the 4-column format
    showing system-level coordination and event processing.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate orchestrator decisions covering orchestrator functions
        orchestrator_decisions = [
            DashboardDecision(
                trigger_detected="Market event detected: Demand spike",
                agent_decision_action="Processed market event for seasonal change",
                value_before="Normal demand pattern",
                value_after="Spike response activated",
                timestamp=current_time,
                function_name="process_market_event",
                confidence_score=0.89,
            ),
            DashboardDecision(
                trigger_detected="Inter-agent coordination needed",
                agent_decision_action="Coordinated agents for collaborative response",
                value_before="Independent operation",
                value_after="Coordinated workflow active",
                timestamp=current_time,
                function_name="coordinate_agents",
                confidence_score=0.94,
            ),
            DashboardDecision(
                trigger_detected="Complex market scenario detected",
                agent_decision_action="Triggered collaboration workflow",
                value_before="Single agent response",
                value_after="Multi-agent collaboration",
                timestamp=current_time,
                function_name="trigger_collaboration_workflow",
                confidence_score=0.87,
            ),
            DashboardDecision(
                trigger_detected="System startup",
                agent_decision_action="Registered all agents with orchestrator",
                value_before="Agents offline",
                value_after="All agents registered and active",
                timestamp=current_time,
                function_name="register_agents",
                confidence_score=0.99,
            ),
            DashboardDecision(
                trigger_detected="Status monitoring request",
                agent_decision_action="Retrieved comprehensive system status",
                value_before="N/A",
                value_after="System health: 98% (247 decisions)",
                timestamp=current_time,
                function_name="get_system_status",
                confidence_score=0.95,
            ),
        ]

        return orchestrator_decisions[:limit]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve orchestrator decisions: {str(e)}",
        )


@router.get("/decisions/collaboration", response_model=List[DashboardDecision])
async def get_collaboration_decisions(limit: int = 10) -> List[DashboardDecision]:
    """
    Get recent collaboration decisions for dashboard display.

    This endpoint provides inter-agent collaboration decisions in the 4-column format
    showing cross-agent coordination and learning.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate collaboration decisions covering collaboration functions
        collaboration_decisions = [
            DashboardDecision(
                trigger_detected="Slow-moving inventory alert from inventory agent",
                agent_decision_action="Sent pricing alert for slow-moving items",
                value_before="Independent pricing",
                value_after="Coordinated markdown strategy",
                timestamp=current_time,
                function_name="inventory_to_pricing_slow_moving_alert",
                confidence_score=0.91,
            ),
            DashboardDecision(
                trigger_detected="Discount opportunity identified",
                agent_decision_action="Coordinated pricing discount with promotion",
                value_before="Standard pricing",
                value_after="Promotional pricing activated",
                timestamp=current_time,
                function_name="pricing_to_promotion_discount_coordination",
                confidence_score=0.88,
            ),
            DashboardDecision(
                trigger_detected="Promotion stock validation request",
                agent_decision_action="Validated inventory for promotional campaign",
                value_before="Promotion pending validation",
                value_after="Promotion inventory confirmed",
                timestamp=current_time,
                function_name="promotion_to_inventory_stock_validation",
                confidence_score=0.93,
            ),
            DashboardDecision(
                trigger_detected="Decision outcomes available",
                agent_decision_action="Applied cross-agent learning from outcomes",
                value_before="Individual learning",
                value_after="Collaborative learning applied",
                timestamp=current_time,
                function_name="cross_agent_learning_from_outcomes",
                confidence_score=0.85,
            ),
            DashboardDecision(
                trigger_detected="Complex market event",
                agent_decision_action="Executed collaborative market event response",
                value_before="Single agent response",
                value_after="Multi-agent coordinated response",
                timestamp=current_time,
                function_name="collaborative_market_event_response",
                confidence_score=0.90,
            ),
        ]

        return collaboration_decisions[:limit]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve collaboration decisions: {str(e)}",
        )


@router.get("/coverage/metrics", response_model=Dict[str, Any])
async def get_function_coverage_metrics() -> Dict[str, Any]:
    """
    Get comprehensive function coverage metrics for demo validation.

    This endpoint provides detailed metrics on which agent functions have been
    executed during the simulation, ensuring all capabilities are demonstrated.
    """
    try:
        current_time = datetime.now(timezone.utc)

        # Simulate comprehensive coverage metrics
        # In a real implementation, this would come from the function tracker
        coverage_metrics = {
            "timestamp": current_time.isoformat(),
            "overall_coverage": {
                "total_functions": 34,
                "executed_functions": 28,
                "coverage_percentage": 82.4,
                "not_executed_functions": [
                    "trigger_collaboration_workflow",
                    "register_agents",
                    "inventory_to_pricing_slow_moving_alert",
                    "collaborative_market_event_response",
                    "cross_agent_learning_from_outcomes",
                    "promotion_to_inventory_stock_validation",
                ],
            },
            "coverage_by_agent": {
                "pricing": {
                    "total": 8,
                    "executed": 7,
                    "percentage": 87.5,
                },
                "inventory": {
                    "total": 8,
                    "executed": 8,
                    "percentage": 100.0,
                },
                "promotion": {
                    "total": 8,
                    "executed": 6,
                    "percentage": 75.0,
                },
                "orchestrator": {
                    "total": 5,
                    "executed": 3,
                    "percentage": 60.0,
                },
                "collaboration": {
                    "total": 5,
                    "executed": 4,
                    "percentage": 80.0,
                },
            },
            "execution_statistics": {
                "total_executions": 156,
                "most_executed_function": ["forecast_demand_probabilistic", 12],
                "least_executed_function": ["register_agents", 1],
                "average_executions_per_function": 4.6,
            },
            "function_status": {
                "fully_covered_agents": ["inventory"],
                "partially_covered_agents": [
                    "pricing",
                    "promotion",
                    "orchestrator",
                    "collaboration",
                ],
                "uncovered_agents": [],
                "demo_readiness_score": 82.4,
            },
        }

        return coverage_metrics

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve coverage metrics: {str(e)}",
        )


# Utility function to broadcast agent-specific events
async def broadcast_agent_event(agent_id: str, event_type: str, data: Dict):
    """
    Utility function to broadcast agent-specific events to subscribed clients.

    This function can be called when agents make decisions or change status
    to provide real-time updates to dashboard users monitoring specific agents.
    """
    event_message = {
        "type": event_type,
        "agent_id": agent_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    await manager.broadcast_to_agent_subscribers(agent_id, json.dumps(event_message))
