"""
Integration tests for simulation engine event capture.

This module tests the integration between the simulation engine
and the event capture system for Langfuse tracing.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from simulation.engine import SimulationEngine, SimulationMode
from config.simulation_event_capture import TriggerEventType


class TestSimulationEngineIntegration:
    """Test integration between simulation engine and event capture."""
    
    @pytest.fixture
    def mock_event_capture(self):
        """Create a mock event capture system."""
        mock_capture = Mock()
        mock_capture.capture_trigger_event.return_value = "test_event_id"
        mock_capture.capture_scenario_trigger.return_value = "test_scenario_event_id"
        return mock_capture
    
    @pytest.fixture
    def simulation_engine(self, mock_event_capture):
        """Create simulation engine with mocked event capture."""
        with patch('simulation.engine.get_simulation_event_capture', return_value=mock_event_capture):
            engine = SimulationEngine(SimulationMode.DEMO)
            return engine
    
    @pytest.mark.asyncio
    async def test_simulation_start_captures_event(self, simulation_engine, mock_event_capture):
        """Test that starting simulation captures a start event."""
        # Mock the initialization methods to avoid complex setup
        simulation_engine.initialize = AsyncMock()
        
        await simulation_engine.start_simulation()
        
        # Verify start event was captured
        mock_event_capture.capture_trigger_event.assert_called_once()
        call_args = mock_event_capture.capture_trigger_event.call_args[0][0]
        
        assert call_args["type"] == TriggerEventType.SIMULATION_START.value
        assert call_args["source"] == "simulation_engine"
        assert call_args["mode"] == "demo"
        assert "components" in call_args
        assert "metadata" in call_args
    
    @pytest.mark.asyncio
    async def test_simulation_stop_captures_event(self, simulation_engine, mock_event_capture):
        """Test that stopping simulation captures a stop event."""
        # Start simulation first
        simulation_engine.state.is_running = True
        simulation_engine.state.current_time = datetime.now()
        
        await simulation_engine.stop_simulation()
        
        # Verify stop event was captured (should be second call after start)
        assert mock_event_capture.capture_trigger_event.call_count >= 1
        
        # Find the stop event call
        stop_call = None
        for call in mock_event_capture.capture_trigger_event.call_args_list:
            if call[0][0]["type"] == TriggerEventType.SIMULATION_STOP.value:
                stop_call = call[0][0]
                break
        
        assert stop_call is not None
        assert stop_call["source"] == "simulation_engine"
        assert "duration" in stop_call
        assert "final_metrics" in stop_call
        assert "stop_reason" in stop_call
    
    @pytest.mark.asyncio
    async def test_trigger_scenario_captures_event(self, simulation_engine, mock_event_capture):
        """Test that triggering a scenario captures an event."""
        # Mock the trigger engine methods
        simulation_engine.trigger_engine.get_available_triggers = AsyncMock(return_value=[
            {
                "id": "test_scenario_id",
                "name": "Test Scenario",
                "agent_type": "pricing",
                "trigger_type": "demand_spike",
                "cooldown_minutes": 30
            }
        ])
        simulation_engine.trigger_engine.trigger_scenario = AsyncMock(return_value=True)
        
        success = await simulation_engine.trigger_scenario("Test Scenario", "high")
        
        assert success is True
        
        # Verify scenario trigger event was captured
        mock_event_capture.capture_trigger_event.assert_called()
        
        # Find the trigger scenario call
        trigger_call = None
        for call in mock_event_capture.capture_trigger_event.call_args_list:
            if call[0][0]["type"] == TriggerEventType.TRIGGER_SCENARIO.value:
                trigger_call = call[0][0]
                break
        
        assert trigger_call is not None
        assert trigger_call["scenario_name"] == "Test Scenario"
        assert trigger_call["intensity"] == "high"
        assert trigger_call["affected_agents"] == ["pricing"]
    
    @pytest.mark.asyncio
    async def test_trigger_scenario_failure_no_event(self, simulation_engine, mock_event_capture):
        """Test that failed scenario trigger doesn't capture event."""
        # Mock the trigger engine to fail
        simulation_engine.trigger_engine.get_available_triggers = AsyncMock(return_value=[])
        simulation_engine.trigger_engine.trigger_scenario = AsyncMock(return_value=False)
        
        success = await simulation_engine.trigger_scenario("Nonexistent Scenario", "medium")
        
        assert success is False
        
        # Should not capture trigger scenario event for failed triggers
        trigger_calls = [
            call for call in mock_event_capture.capture_trigger_event.call_args_list
            if call[0][0]["type"] == TriggerEventType.TRIGGER_SCENARIO.value
        ]
        assert len(trigger_calls) == 0


class TestTriggerEngineIntegration:
    """Test integration between trigger engine and event capture."""
    
    @pytest.fixture
    def mock_event_capture(self):
        """Create a mock event capture system."""
        mock_capture = Mock()
        mock_capture.capture_scenario_trigger.return_value = "test_scenario_event_id"
        return mock_capture
    
    @pytest.fixture
    def trigger_engine(self, mock_event_capture):
        """Create trigger engine with mocked event capture."""
        with patch('simulation.triggers.get_simulation_event_capture', return_value=mock_event_capture):
            from simulation.triggers import TriggerEngine
            engine = TriggerEngine()
            return engine
    
    @pytest.mark.asyncio
    async def test_trigger_scenario_captures_event(self, trigger_engine, mock_event_capture):
        """Test that trigger engine captures events when scenarios are triggered."""
        # Initialize with minimal setup
        await trigger_engine.initialize(None)
        
        # Get a test scenario
        scenarios = list(trigger_engine.scenarios.values())
        assert len(scenarios) > 0
        
        test_scenario = scenarios[0]
        
        # Trigger the scenario
        success = await trigger_engine.trigger_scenario(test_scenario.name, "medium")
        
        assert success is True
        
        # Verify event was captured
        mock_event_capture.capture_scenario_trigger.assert_called_once()
        call_args = mock_event_capture.capture_scenario_trigger.call_args[0][0]
        
        assert call_args["scenario_id"] == test_scenario.id
        assert call_args["scenario_name"] == test_scenario.name
        assert call_args["agent_type"] == test_scenario.agent_type
        assert call_args["trigger_type"] == test_scenario.trigger_type
        assert call_args["intensity"] == "medium"
        assert call_args["conditions"] == test_scenario.conditions
        assert call_args["effects"] == test_scenario.effects


if __name__ == "__main__":
    pytest.main([__file__])