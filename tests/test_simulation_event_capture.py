"""
Tests for SimulationEventCapture functionality.

This module tests the simulation event capture system that creates
root traces in Langfuse for workflow visualization.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any

from config.simulation_event_capture import (
    SimulationEventCapture,
    TriggerEventType,
    SimulationEventData,
    EventTraceMapping,
    get_simulation_event_capture,
    initialize_simulation_event_capture
)
from config.langfuse_integration import LangfuseIntegrationService
from models.core import MarketEvent, EventType


class TestSimulationEventCapture:
    """Test cases for SimulationEventCapture class."""
    
    @pytest.fixture
    def mock_integration_service(self):
        """Create a mock Langfuse integration service."""
        service = Mock(spec=LangfuseIntegrationService)
        service.is_available = True
        service.create_simulation_trace.return_value = "test_trace_id"
        service.start_agent_span.return_value = "test_span_id"
        service.end_agent_span.return_value = None
        service.finalize_trace.return_value = None
        return service
    
    @pytest.fixture
    def event_capture(self, mock_integration_service):
        """Create SimulationEventCapture instance with mock service."""
        return SimulationEventCapture(mock_integration_service)
    
    def test_initialization(self, event_capture):
        """Test proper initialization of SimulationEventCapture."""
        assert event_capture._integration_service is not None
        assert len(event_capture._trace_mappings) == 6  # All event types mapped
        assert event_capture._active_event_traces == {}
        assert event_capture._event_callbacks == {}
    
    def test_trace_mappings_initialization(self, event_capture):
        """Test that all event types have proper trace mappings."""
        expected_event_types = [
            TriggerEventType.SIMULATION_START,
            TriggerEventType.SIMULATION_STOP,
            TriggerEventType.TRIGGER_SCENARIO,
            TriggerEventType.MARKET_EVENT,
            TriggerEventType.AGENT_ACTIVATION,
            TriggerEventType.COLLABORATION_WORKFLOW
        ]
        
        for event_type in expected_event_types:
            assert event_type in event_capture._trace_mappings
            mapping = event_capture._trace_mappings[event_type]
            assert isinstance(mapping, EventTraceMapping)
            assert mapping.trace_name
            assert isinstance(mapping.input_fields, list)
            assert isinstance(mapping.metadata_fields, list)
            assert isinstance(mapping.tags, list)
    
    def test_capture_trigger_event_success(self, event_capture, mock_integration_service):
        """Test successful trigger event capture."""
        trigger_data = {
            "type": "trigger_scenario",
            "event_id": "test_event_123",
            "source": "test_engine",
            "scenario_id": "test_scenario",
            "scenario_name": "Test Scenario",
            "intensity": "high",
            "affected_agents": ["pricing_agent"],
            "metadata": {"test_key": "test_value"}
        }
        
        event_id = event_capture.capture_trigger_event(trigger_data)
        
        assert event_id == "test_event_123"
        assert event_id in event_capture._active_event_traces
        assert event_capture._active_event_traces[event_id] == "test_trace_id"
        
        # Verify integration service was called
        mock_integration_service.create_simulation_trace.assert_called_once()
        call_args = mock_integration_service.create_simulation_trace.call_args[0][0]
        assert call_args["type"] == "trigger_scenario"
        assert call_args["source"] == "test_engine"
        assert call_args["event_id"] == "test_event_123"
    
    def test_capture_trigger_event_langfuse_unavailable(self, mock_integration_service):
        """Test trigger event capture when Langfuse is unavailable."""
        mock_integration_service.is_available = False
        mock_integration_service.create_simulation_trace.return_value = None
        
        event_capture = SimulationEventCapture(mock_integration_service)
        
        trigger_data = {
            "type": "trigger_scenario",
            "event_id": "test_event_123"
        }
        
        event_id = event_capture.capture_trigger_event(trigger_data)
        
        assert event_id is None
        assert len(event_capture._active_event_traces) == 0
    
    def test_capture_trigger_event_with_default_values(self, event_capture):
        """Test trigger event capture with minimal data (using defaults)."""
        trigger_data = {
            "scenario_name": "Minimal Test Scenario"
        }
        
        event_id = event_capture.capture_trigger_event(trigger_data)
        
        assert event_id is not None
        assert event_id.startswith("event_")
        assert event_id in event_capture._active_event_traces
    
    def test_build_trace_input(self, event_capture):
        """Test building trace input data from event data."""
        event_data = SimulationEventData(
            event_id="test_event",
            event_type=TriggerEventType.TRIGGER_SCENARIO,
            timestamp=datetime.now(),
            source="test_source",
            affected_agents=["agent1", "agent2"],
            trigger_data={
                "scenario_id": "test_scenario",
                "intensity": "high",
                "conditions": {"key": "value"}
            }
        )
        
        mapping = event_capture._trace_mappings[TriggerEventType.TRIGGER_SCENARIO]
        trace_input = event_capture._build_trace_input(event_data, mapping)
        
        assert trace_input["event_type"] == "trigger_scenario"
        assert trace_input["affected_agents"] == ["agent1", "agent2"]
        assert trace_input["source"] == "test_source"
        assert trace_input["scenario_id"] == "test_scenario"
        assert trace_input["intensity"] == "high"
    
    def test_build_trace_metadata(self, event_capture):
        """Test building trace metadata from event data."""
        event_data = SimulationEventData(
            event_id="test_event",
            event_type=TriggerEventType.TRIGGER_SCENARIO,
            timestamp=datetime.now(),
            source="test_source",
            metadata={"custom_key": "custom_value"},
            trigger_data={
                "agent_type": "pricing",
                "trigger_type": "demand_spike"
            }
        )
        
        mapping = event_capture._trace_mappings[TriggerEventType.TRIGGER_SCENARIO]
        metadata = event_capture._build_trace_metadata(event_data, mapping)
        
        assert metadata["event_id"] == "test_event"
        assert metadata["system"] == "autoops_retail_optimization"
        assert metadata["component"] == "simulation_engine"
        assert metadata["custom_key"] == "custom_value"
        assert metadata["agent_type"] == "pricing"
        assert metadata["trigger_type"] == "demand_spike"
    
    def test_track_event_propagation(self, event_capture, mock_integration_service):
        """Test tracking event propagation through agents."""
        # First capture an event
        trigger_data = {"type": "trigger_scenario", "event_id": "test_event"}
        event_id = event_capture.capture_trigger_event(trigger_data)
        
        # Track agent responses
        agent_responses = [
            {
                "agent_id": "pricing_agent",
                "response": {"action": "price_update"},
                "processing_time": 0.5,
                "actions": ["update_price"],
                "decisions": ["increase_price_by_10_percent"],
                "success": True
            },
            {
                "agent_id": "inventory_agent",
                "response": {"action": "restock_alert"},
                "processing_time": 0.3,
                "actions": ["generate_alert"],
                "decisions": ["restock_needed"],
                "success": True
            }
        ]
        
        event_capture.track_event_propagation(event_id, agent_responses)
        
        # Verify spans were created for each agent
        assert mock_integration_service.start_agent_span.call_count == 2
        assert mock_integration_service.end_agent_span.call_count == 2
        
        # Check first agent span call
        first_call = mock_integration_service.start_agent_span.call_args_list[0]
        assert first_call[1]["agent_id"] == "pricing_agent"
        assert first_call[1]["operation"] == "event_response"
        assert first_call[1]["parent_trace_id"] == "test_trace_id"
    
    def test_track_event_propagation_no_active_trace(self, event_capture, mock_integration_service):
        """Test tracking event propagation when no active trace exists."""
        agent_responses = [{"agent_id": "test_agent"}]
        
        # Should not raise exception, just log warning
        event_capture.track_event_propagation("nonexistent_event", agent_responses)
        
        # No spans should be created
        mock_integration_service.start_agent_span.assert_not_called()
    
    def test_finalize_event_trace(self, event_capture, mock_integration_service):
        """Test finalizing an event trace."""
        # First capture an event
        trigger_data = {"type": "trigger_scenario", "event_id": "test_event"}
        event_id = event_capture.capture_trigger_event(trigger_data)
        
        final_outcome = {
            "total_agents_activated": 3,
            "successful_actions": 5,
            "duration_seconds": 120.5
        }
        
        event_capture.finalize_event_trace(event_id, final_outcome)
        
        # Verify trace was finalized
        mock_integration_service.finalize_trace.assert_called_once_with(
            "test_trace_id", final_outcome
        )
        
        # Event should be removed from active traces
        assert event_id not in event_capture._active_event_traces
    
    def test_register_and_execute_event_callback(self, event_capture):
        """Test registering and executing event callbacks."""
        callback_executed = False
        received_event_data = None
        
        def test_callback(event_data: SimulationEventData):
            nonlocal callback_executed, received_event_data
            callback_executed = True
            received_event_data = event_data
        
        # Register callback
        event_capture.register_event_callback(TriggerEventType.TRIGGER_SCENARIO, test_callback)
        
        # Trigger event
        trigger_data = {"type": "trigger_scenario", "event_id": "test_event"}
        event_capture.capture_trigger_event(trigger_data)
        
        # Verify callback was executed
        assert callback_executed
        assert received_event_data is not None
        assert received_event_data.event_id == "test_event"
        assert received_event_data.event_type == TriggerEventType.TRIGGER_SCENARIO
    
    def test_capture_market_event(self, event_capture, mock_integration_service):
        """Test capturing a MarketEvent."""
        from uuid import uuid4
        
        market_event = MarketEvent(
            id=uuid4(),
            event_type=EventType.DEMAND_SPIKE,
            affected_products=["product_1", "product_2"],
            impact_magnitude=0.8,
            description="Test market event",
            timestamp=datetime.now(),
            metadata={"source": "external"}
        )
        
        event_id = event_capture.capture_market_event(market_event)
        
        assert event_id is not None
        assert event_id in event_capture._active_event_traces
        
        # Verify correct data was passed to create trace
        mock_integration_service.create_simulation_trace.assert_called_once()
        call_args = mock_integration_service.create_simulation_trace.call_args[0][0]
        assert call_args["type"] == "market_event"
        assert call_args["source"] == "market_simulation"
        assert call_args["affected_products"] == ["product_1", "product_2"]
        assert call_args["impact_magnitude"] == 0.8
    
    def test_capture_scenario_trigger(self, event_capture, mock_integration_service):
        """Test capturing a scenario trigger."""
        scenario_data = {
            "scenario_id": "pricing_scenario_123",
            "scenario_name": "Demand Spike Response",
            "agent_type": "pricing",
            "trigger_type": "demand_spike",
            "intensity": "high",
            "conditions": {"demand_multiplier": 2.5},
            "effects": {"function_calls": ["analyze_demand"]},
            "cooldown_minutes": 30
        }
        
        event_id = event_capture.capture_scenario_trigger(scenario_data)
        
        assert event_id is not None
        assert event_id.startswith("scenario_pricing_scenario_123_")
        assert event_id in event_capture._active_event_traces
        
        # Verify correct data was passed
        mock_integration_service.create_simulation_trace.assert_called_once()
        call_args = mock_integration_service.create_simulation_trace.call_args[0][0]
        assert call_args["type"] == "trigger_scenario"
        assert call_args["source"] == "trigger_engine"
        assert call_args["scenario_id"] == "pricing_scenario_123"
        assert call_args["intensity"] == "high"
    
    def test_get_active_event_traces(self, event_capture):
        """Test getting active event traces."""
        # Initially empty
        active_traces = event_capture.get_active_event_traces()
        assert active_traces == {}
        
        # Add some events
        event_capture.capture_trigger_event({"event_id": "event1"})
        event_capture.capture_trigger_event({"event_id": "event2"})
        
        active_traces = event_capture.get_active_event_traces()
        assert len(active_traces) == 2
        assert "event1" in active_traces
        assert "event2" in active_traces
        
        # Should return a copy, not the original dict
        active_traces["event3"] = "trace3"
        assert "event3" not in event_capture._active_event_traces
    
    def test_health_check(self, event_capture):
        """Test health check functionality."""
        # Add some test data
        event_capture.capture_trigger_event({"event_id": "test_event"})
        event_capture.register_event_callback(TriggerEventType.TRIGGER_SCENARIO, lambda x: None)
        
        health = event_capture.health_check()
        
        assert health["service"] == "simulation_event_capture"
        assert health["status"] == "healthy"
        assert health["active_traces"] == 1
        assert health["registered_callbacks"] == 1
        assert health["trace_mappings"] == 6
        assert health["langfuse_available"] is True
    
    def test_health_check_degraded(self, mock_integration_service):
        """Test health check when Langfuse is unavailable."""
        mock_integration_service.is_available = False
        event_capture = SimulationEventCapture(mock_integration_service)
        
        health = event_capture.health_check()
        
        assert health["status"] == "degraded"
        assert health["langfuse_available"] is False


class TestGlobalFunctions:
    """Test global functions for SimulationEventCapture."""
    
    def test_get_simulation_event_capture_singleton(self):
        """Test that get_simulation_event_capture returns singleton."""
        capture1 = get_simulation_event_capture()
        capture2 = get_simulation_event_capture()
        
        assert capture1 is capture2
        assert isinstance(capture1, SimulationEventCapture)
    
    @patch('config.simulation_event_capture._event_capture', None)
    def test_initialize_simulation_event_capture(self):
        """Test initialization with custom integration service."""
        mock_service = Mock(spec=LangfuseIntegrationService)
        
        capture = initialize_simulation_event_capture(mock_service)
        
        assert isinstance(capture, SimulationEventCapture)
        assert capture._integration_service is mock_service
        
        # Should also set global instance
        assert get_simulation_event_capture() is capture


class TestEventTraceMapping:
    """Test EventTraceMapping data structure."""
    
    def test_event_trace_mapping_creation(self):
        """Test creating EventTraceMapping instances."""
        mapping = EventTraceMapping(
            trace_name="test_trace",
            input_fields=["field1", "field2"],
            metadata_fields=["meta1", "meta2"],
            tags=["tag1", "tag2"]
        )
        
        assert mapping.trace_name == "test_trace"
        assert mapping.input_fields == ["field1", "field2"]
        assert mapping.metadata_fields == ["meta1", "meta2"]
        assert mapping.tags == ["tag1", "tag2"]
    
    def test_event_trace_mapping_defaults(self):
        """Test EventTraceMapping with default values."""
        mapping = EventTraceMapping(
            trace_name="test_trace",
            input_fields=["field1"],
            metadata_fields=["meta1"]
        )
        
        assert mapping.tags == []  # Default empty list


class TestSimulationEventData:
    """Test SimulationEventData data structure."""
    
    def test_simulation_event_data_creation(self):
        """Test creating SimulationEventData instances."""
        timestamp = datetime.now()
        
        event_data = SimulationEventData(
            event_id="test_event",
            event_type=TriggerEventType.TRIGGER_SCENARIO,
            timestamp=timestamp,
            source="test_source",
            metadata={"key": "value"},
            affected_agents=["agent1"],
            trigger_data={"trigger_key": "trigger_value"}
        )
        
        assert event_data.event_id == "test_event"
        assert event_data.event_type == TriggerEventType.TRIGGER_SCENARIO
        assert event_data.timestamp == timestamp
        assert event_data.source == "test_source"
        assert event_data.metadata == {"key": "value"}
        assert event_data.affected_agents == ["agent1"]
        assert event_data.trigger_data == {"trigger_key": "trigger_value"}
    
    def test_simulation_event_data_defaults(self):
        """Test SimulationEventData with default values."""
        timestamp = datetime.now()
        
        event_data = SimulationEventData(
            event_id="test_event",
            event_type=TriggerEventType.TRIGGER_SCENARIO,
            timestamp=timestamp,
            source="test_source"
        )
        
        assert event_data.metadata == {}
        assert event_data.affected_agents == []
        assert event_data.trigger_data is None


if __name__ == "__main__":
    pytest.main([__file__])