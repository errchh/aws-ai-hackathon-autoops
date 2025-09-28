"""
Tests for dashboard WebSocket endpoints.

This module contains tests for the WebSocket functionality of the dashboard
API endpoints to ensure real-time communication works correctly.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from api.main import app


class TestDashboardWebSockets:
    """Test suite for dashboard WebSocket endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_websocket_live_updates_connection(self):
        """Test WebSocket connection for live updates."""
        with self.client.websocket_connect("/api/dashboard/ws/live-updates") as websocket:
            # Should receive connection established message
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert "timestamp" in data
            assert data["message"] == "Connected to live updates stream"
    
    def test_websocket_agent_status_connection(self):
        """Test WebSocket connection for agent status updates."""
        agent_id = "pricing_agent"
        with self.client.websocket_connect(f"/api/dashboard/ws/agents/{agent_id}/status") as websocket:
            # Should receive initial agent status
            data = websocket.receive_json()
            assert data["type"] == "agent_status"
            assert data["agent_id"] == agent_id
            assert "timestamp" in data
            assert "data" in data
    
    def test_websocket_invalid_agent_id(self):
        """Test WebSocket connection with invalid agent ID."""
        with pytest.raises(Exception):  # Should close connection
            with self.client.websocket_connect("/api/dashboard/ws/agents/invalid_agent/status") as websocket:
                pass
    
    def test_websocket_metrics_stream_connection(self):
        """Test WebSocket connection for real-time metrics."""
        with self.client.websocket_connect("/api/dashboard/ws/metrics/real-time") as websocket:
            # Should receive initial metrics
            data = websocket.receive_json()
            assert data["type"] == "metrics_stream"
            assert "timestamp" in data
            assert "data" in data
            
            # Verify metrics data structure
            metrics_data = data["data"]
            assert "total_revenue" in metrics_data
            assert "total_profit" in metrics_data
            assert "inventory_turnover" in metrics_data


class TestDashboardAPI:
    """Test suite for dashboard REST API endpoints."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_get_agents_status(self):
        """Test getting agent status information."""
        response = self.client.get("/api/dashboard/agents/status")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Verify agent status structure
        agent_status = data[0]
        assert "agent_id" in agent_status
        assert "name" in agent_status
        assert "status" in agent_status
        assert "decisions_count" in agent_status
        assert "success_rate" in agent_status
    
    def test_get_current_dashboard_metrics(self):
        """Test getting current dashboard metrics."""
        response = self.client.get("/api/dashboard/metrics/current")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "system_metrics" in data
        assert "agent_statuses" in data
        assert "active_alerts_count" in data
        assert "recent_decisions_count" in data
    
    def test_get_recent_decisions(self):
        """Test getting recent agent decisions."""
        response = self.client.get("/api/dashboard/decisions/recent")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there are decisions
            decision = data[0]
            assert "agent_id" in decision
            assert "action_type" in decision
            assert "parameters" in decision
            assert "rationale" in decision
            assert "confidence_score" in decision
    
    def test_get_recent_decisions_with_agent_filter(self):
        """Test getting recent decisions filtered by agent."""
        response = self.client.get("/api/dashboard/decisions/recent?agent_id=pricing_agent")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        # All decisions should be from the specified agent
        for decision in data:
            assert decision["agent_id"] == "pricing_agent"
    
    def test_get_active_alerts(self):
        """Test getting active system alerts."""
        response = self.client.get("/api/dashboard/alerts/active")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        
        if data:  # If there are alerts
            alert = data[0]
            assert "id" in alert
            assert "timestamp" in alert
            assert "severity" in alert
            assert "title" in alert
            assert "message" in alert
    
    def test_manual_intervention_valid_agent(self):
        """Test manual intervention with valid agent and action."""
        intervention_data = {
            "agent_id": "pricing_agent",
            "action": "pause",
            "reason": "Testing manual intervention"
        }
        
        response = self.client.post(
            "/api/dashboard/agents/pricing_agent/intervention",
            json=intervention_data
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["agent_id"] == "pricing_agent"
        assert "timestamp" in data
        assert "message" in data
    
    def test_manual_intervention_invalid_agent(self):
        """Test manual intervention with invalid agent ID."""
        intervention_data = {
            "agent_id": "invalid_agent",
            "action": "pause",
            "reason": "Testing invalid agent"
        }
        
        response = self.client.post(
            "/api/dashboard/agents/invalid_agent/intervention",
            json=intervention_data
        )
        assert response.status_code == 404
    
    def test_manual_intervention_invalid_action(self):
        """Test manual intervention with invalid action."""
        intervention_data = {
            "agent_id": "pricing_agent",
            "action": "invalid_action",
            "reason": "Testing invalid action"
        }
        
        response = self.client.post(
            "/api/dashboard/agents/pricing_agent/intervention",
            json=intervention_data
        )
        assert response.status_code == 400


@pytest.mark.asyncio
class TestWebSocketConnectionManager:
    """Test suite for WebSocket connection management."""
    
    async def test_connection_manager_basic_operations(self):
        """Test basic connection manager operations."""
        from api.routers.dashboard import ConnectionManager
        
        manager = ConnectionManager()
        
        # Mock WebSocket connections
        class MockWebSocket:
            def __init__(self, id: str):
                self.id = id
                self.closed = False
            
            async def accept(self):
                pass
            
            async def send_text(self, message: str):
                if self.closed:
                    raise Exception("Connection closed")
        
        # Test connection management
        ws1 = MockWebSocket("ws1")
        ws2 = MockWebSocket("ws2")
        
        await manager.connect(ws1, "general")
        await manager.connect(ws2, "agent_pricing_agent")
        
        assert ws1 in manager.active_connections
        assert ws2 in manager.active_connections
        assert ws2 in manager.agent_connections["pricing_agent"]
        
        # Test disconnection
        manager.disconnect(ws1)
        assert ws1 not in manager.active_connections
        
        # Test broadcasting
        await manager.broadcast("test message")
        
        # Test agent-specific broadcasting
        await manager.broadcast_to_agent_subscribers("pricing_agent", "agent message")


if __name__ == "__main__":
    pytest.main([__file__])