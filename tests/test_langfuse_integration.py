"""Tests for Langfuse integration foundation."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from config.langfuse_config import LangfuseConfig, LangfuseClient
from config.langfuse_integration import LangfuseIntegrationService


class TestLangfuseConfig:
    """Test Langfuse configuration."""
    
    def test_config_creation(self):
        """Test creating a Langfuse configuration."""
        config = LangfuseConfig(
            public_key="pk-test",
            secret_key="sk-test",
            host="https://test.langfuse.com",
            enabled=True
        )
        
        assert config.public_key == "pk-test"
        assert config.secret_key == "sk-test"
        assert config.host == "https://test.langfuse.com"
        assert config.enabled is True
        assert config.flush_interval == 5.0
        assert config.flush_at == 15
        assert config.timeout == 60
        assert config.tracing_enabled is True


class TestLangfuseClient:
    """Test Langfuse client wrapper."""
    
    def test_client_initialization_without_credentials(self):
        """Test client initialization without credentials."""
        config = LangfuseConfig(
            public_key="",
            secret_key="",
            enabled=False
        )
        
        client = LangfuseClient(config)
        
        assert not client.is_available
        assert client.client is None
        assert client.config.enabled is False
    
    def test_client_health_check(self):
        """Test client health check functionality."""
        config = LangfuseConfig(
            public_key="",
            secret_key="",
            enabled=False
        )
        
        client = LangfuseClient(config)
        health = client.health_check()
        
        assert "initialized" in health
        assert "available" in health
        assert "fallback_mode" in health
        assert "enabled" in health
        assert "connection_status" in health
        assert health["connection_status"] == "unavailable"
    
    def test_fallback_mode(self):
        """Test fallback mode functionality."""
        config = LangfuseConfig(
            public_key="pk-test",
            secret_key="sk-test",
            enabled=True
        )
        
        client = LangfuseClient(config)
        client.enable_fallback_mode()
        
        assert not client.is_available
        assert client._fallback_mode is True


class TestLangfuseIntegrationService:
    """Test Langfuse integration service."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        service = LangfuseIntegrationService()
        
        assert service is not None
        assert hasattr(service, '_client')
        assert hasattr(service, '_active_traces')
        assert hasattr(service, '_active_spans')
    
    def test_service_unavailable_graceful_degradation(self):
        """Test service graceful degradation when Langfuse is unavailable."""
        service = LangfuseIntegrationService()
        
        # These should all return None/do nothing when service is unavailable
        trace_id = service.create_simulation_trace({"type": "test"})
        assert trace_id is None
        
        span_id = service.start_agent_span("test_agent", "test_operation")
        assert span_id is None
        
        # These should not raise exceptions
        service.end_agent_span("nonexistent_span")
        service.log_agent_decision("test_agent", {"type": "test"})
        service.finalize_trace("nonexistent_trace")
        service.flush()
    
    def test_health_check(self):
        """Test integration service health check."""
        service = LangfuseIntegrationService()
        health = service.health_check()
        
        assert "active_traces" in health
        assert "active_spans" in health
        assert "integration_service" in health
        assert health["integration_service"] == "ready"
        assert health["active_traces"] == 0
        assert health["active_spans"] == 0
    
    @patch('config.langfuse_integration.get_langfuse_client')
    def test_service_with_mock_client(self, mock_get_client):
        """Test service with mocked available client."""
        # Create mock client that appears available
        mock_client = Mock()
        mock_client.is_available = True
        mock_langfuse = Mock()
        mock_client.client = mock_langfuse
        mock_get_client.return_value = mock_client
        
        service = LangfuseIntegrationService()
        
        # Test trace creation
        mock_trace = Mock()
        mock_langfuse.trace.return_value = mock_trace
        
        trace_id = service.create_simulation_trace({"type": "test", "source": "unit_test"})
        
        assert trace_id is not None
        assert trace_id.startswith("sim_")
        mock_langfuse.trace.assert_called_once()
        
        # Verify trace was stored
        assert trace_id in service._active_traces
    
    @patch('config.langfuse_integration.get_langfuse_client')
    def test_agent_span_operations(self, mock_get_client):
        """Test agent span start/end operations."""
        # Create mock client that appears available
        mock_client = Mock()
        mock_client.is_available = True
        mock_langfuse = Mock()
        mock_client.client = mock_langfuse
        mock_get_client.return_value = mock_client
        
        service = LangfuseIntegrationService()
        
        # Mock span creation
        mock_span = Mock()
        mock_span.metadata = {}
        mock_langfuse.span.return_value = mock_span
        
        # Test span start
        span_id = service.start_agent_span(
            "inventory_agent", 
            "forecast_demand",
            input_data={"product_id": "123"}
        )
        
        assert span_id is not None
        assert span_id.startswith("inventory_agent_forecast_demand_")
        assert span_id in service._active_spans
        
        # Test span end
        service.end_agent_span(span_id, {"forecast": "high_demand"})
        
        assert span_id not in service._active_spans
        mock_span.update.assert_called()
    
    def test_trace_operation_context_manager(self):
        """Test trace operation context manager."""
        service = LangfuseIntegrationService()
        
        # Should work gracefully even when unavailable
        with service.trace_operation("test_operation") as trace:
            assert trace is None  # Since service is unavailable
    
    @patch('config.langfuse_integration.get_langfuse_client')
    def test_collaboration_tracking(self, mock_get_client):
        """Test collaboration workflow tracking."""
        # Create mock client that appears available
        mock_client = Mock()
        mock_client.is_available = True
        mock_langfuse = Mock()
        mock_client.client = mock_langfuse
        mock_get_client.return_value = mock_client
        
        service = LangfuseIntegrationService()
        
        # Mock trace creation
        mock_trace = Mock()
        mock_langfuse.trace.return_value = mock_trace
        
        # Test collaboration tracking
        trace_id = service.track_collaboration(
            "pricing_inventory_sync",
            ["pricing_agent", "inventory_agent"],
            {"workflow_type": "price_optimization"}
        )
        
        assert trace_id is not None
        assert trace_id.startswith("collab_pricing_inventory_sync_")
        mock_langfuse.trace.assert_called_once()
        
        # Verify the trace call had correct parameters
        call_args = mock_langfuse.trace.call_args
        assert call_args[1]["name"] == "agent_collaboration"
        assert "participating_agents" in call_args[1]["metadata"]
        assert call_args[1]["metadata"]["participating_agents"] == ["pricing_agent", "inventory_agent"]


if __name__ == "__main__":
    pytest.main([__file__])