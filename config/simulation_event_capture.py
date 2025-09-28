"""
Simulation Event Capture for Langfuse Integration

This module provides the SimulationEventCapture class that intercepts
simulation engine events and creates root traces in Langfuse for workflow visualization.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from config.langfuse_integration import LangfuseIntegrationService, get_langfuse_integration
from models.core import MarketEvent, EventType

logger = logging.getLogger(__name__)


class TriggerEventType(str, Enum):
    """Types of trigger events that can be captured."""
    SIMULATION_START = "simulation_start"
    SIMULATION_STOP = "simulation_stop"
    TRIGGER_SCENARIO = "trigger_scenario"
    MARKET_EVENT = "market_event"
    AGENT_ACTIVATION = "agent_activation"
    COLLABORATION_WORKFLOW = "collaboration_workflow"


@dataclass
class SimulationEventData:
    """Data structure for simulation events."""
    event_id: str
    event_type: TriggerEventType
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    affected_agents: List[str] = field(default_factory=list)
    trigger_data: Optional[Dict[str, Any]] = None


@dataclass
class EventTraceMapping:
    """Mapping configuration for event types to trace structure."""
    trace_name: str
    input_fields: List[str]
    metadata_fields: List[str]
    tags: List[str] = field(default_factory=list)


class SimulationEventCapture:
    """
    Captures simulation engine events and creates root traces in Langfuse
    for comprehensive workflow visualization.
    """
    
    def __init__(self, integration_service: Optional[LangfuseIntegrationService] = None):
        """Initialize the simulation event capture system.
        
        Args:
            integration_service: Optional Langfuse integration service instance
        """
        self._integration_service = integration_service or get_langfuse_integration()
        self._active_event_traces: Dict[str, str] = {}  # event_id -> trace_id
        self._event_callbacks: Dict[TriggerEventType, List[Callable]] = {}
        self._trace_mappings = self._initialize_trace_mappings()
        
        logger.info("Initialized SimulationEventCapture")
    
    def _initialize_trace_mappings(self) -> Dict[TriggerEventType, EventTraceMapping]:
        """Initialize event-to-trace mapping configurations."""
        return {
            TriggerEventType.SIMULATION_START: EventTraceMapping(
                trace_name="simulation_lifecycle",
                input_fields=["mode", "timestamp", "components"],
                metadata_fields=["simulation_mode", "start_time", "system_version"],
                tags=["simulation", "lifecycle", "start"]
            ),
            TriggerEventType.SIMULATION_STOP: EventTraceMapping(
                trace_name="simulation_lifecycle",
                input_fields=["duration", "final_metrics", "stop_reason"],
                metadata_fields=["stop_time", "total_events", "performance_summary"],
                tags=["simulation", "lifecycle", "stop"]
            ),
            TriggerEventType.TRIGGER_SCENARIO: EventTraceMapping(
                trace_name="trigger_scenario_execution",
                input_fields=["scenario_id", "scenario_name", "intensity", "conditions"],
                metadata_fields=["agent_type", "trigger_type", "expected_effects"],
                tags=["trigger", "scenario", "agent_activation"]
            ),
            TriggerEventType.MARKET_EVENT: EventTraceMapping(
                trace_name="market_event_response",
                input_fields=["event_type", "affected_products", "impact_magnitude"],
                metadata_fields=["event_source", "duration", "market_conditions"],
                tags=["market", "event", "external_trigger"]
            ),
            TriggerEventType.AGENT_ACTIVATION: EventTraceMapping(
                trace_name="agent_workflow_execution",
                input_fields=["agent_id", "activation_reason", "input_parameters"],
                metadata_fields=["agent_type", "trigger_source", "expected_actions"],
                tags=["agent", "activation", "workflow"]
            ),
            TriggerEventType.COLLABORATION_WORKFLOW: EventTraceMapping(
                trace_name="multi_agent_collaboration",
                input_fields=["workflow_id", "participating_agents", "coordination_type"],
                metadata_fields=["workflow_type", "complexity_level", "expected_outcome"],
                tags=["collaboration", "multi_agent", "coordination"]
            )
        }
    
    def capture_trigger_event(self, trigger_data: Dict[str, Any]) -> Optional[str]:
        """Capture a simulation trigger event and create a root trace.
        
        Args:
            trigger_data: Dictionary containing trigger event information
            
        Returns:
            Event ID if successful, None if capture failed
        """
        try:
            # Extract event information
            event_type = TriggerEventType(trigger_data.get("type", "trigger_scenario"))
            event_id = trigger_data.get("event_id", f"event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
            source = trigger_data.get("source", "simulation_engine")
            
            # Create simulation event data
            event_data = SimulationEventData(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                source=source,
                metadata=trigger_data.get("metadata", {}),
                affected_agents=trigger_data.get("affected_agents", []),
                trigger_data=trigger_data
            )
            
            # Create root trace using event-to-trace mapping
            trace_id = self._create_root_trace(event_data)
            
            if trace_id:
                self._active_event_traces[event_id] = trace_id
                
                # Execute registered callbacks
                self._execute_event_callbacks(event_type, event_data)
                
                logger.info(f"Captured trigger event: {event_id} -> trace: {trace_id}")
                return event_id
            else:
                logger.warning(f"Failed to create trace for event: {event_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to capture trigger event: {e}")
            return None
    
    def _create_root_trace(self, event_data: SimulationEventData) -> Optional[str]:
        """Create a root trace for the simulation event.
        
        Args:
            event_data: Simulation event data
            
        Returns:
            Trace ID if successful, None otherwise
        """
        if not self._integration_service.is_available:
            logger.debug("Langfuse not available, skipping trace creation")
            return None
        
        try:
            # Get trace mapping for event type
            mapping = self._trace_mappings.get(event_data.event_type)
            if not mapping:
                logger.warning(f"No trace mapping found for event type: {event_data.event_type}")
                return None
            
            # Build trace input data
            trace_input = self._build_trace_input(event_data, mapping)
            
            # Build trace metadata
            trace_metadata = self._build_trace_metadata(event_data, mapping)
            
            # Create the root trace
            trace_data = {
                "type": event_data.event_type.value,
                "source": event_data.source,
                "timestamp": event_data.timestamp.isoformat(),
                "event_id": event_data.event_id,
                **trace_input
            }
            
            trace_id = self._integration_service.create_simulation_trace(trace_data)
            
            if trace_id:
                # Update trace with additional metadata
                self._update_trace_metadata(trace_id, trace_metadata, mapping.tags)
            
            return trace_id
            
        except Exception as e:
            logger.error(f"Failed to create root trace for event {event_data.event_id}: {e}")
            return None
    
    def _build_trace_input(self, event_data: SimulationEventData, mapping: EventTraceMapping) -> Dict[str, Any]:
        """Build trace input data based on mapping configuration.
        
        Args:
            event_data: Simulation event data
            mapping: Event trace mapping configuration
            
        Returns:
            Dictionary of trace input data
        """
        trace_input = {}
        
        # Extract specified input fields from trigger data
        if event_data.trigger_data:
            for field in mapping.input_fields:
                if field in event_data.trigger_data:
                    trace_input[field] = event_data.trigger_data[field]
        
        # Add standard fields
        trace_input.update({
            "event_type": event_data.event_type.value,
            "affected_agents": event_data.affected_agents,
            "source": event_data.source
        })
        
        return trace_input
    
    def _build_trace_metadata(self, event_data: SimulationEventData, mapping: EventTraceMapping) -> Dict[str, Any]:
        """Build trace metadata based on mapping configuration.
        
        Args:
            event_data: Simulation event data
            mapping: Event trace mapping configuration
            
        Returns:
            Dictionary of trace metadata
        """
        metadata = {
            "event_id": event_data.event_id,
            "capture_timestamp": event_data.timestamp.isoformat(),
            "system": "autoops_retail_optimization",
            "component": "simulation_engine"
        }
        
        # Add event-specific metadata
        metadata.update(event_data.metadata)
        
        # Extract specified metadata fields from trigger data
        if event_data.trigger_data:
            for field in mapping.metadata_fields:
                if field in event_data.trigger_data:
                    metadata[field] = event_data.trigger_data[field]
        
        return metadata
    
    def _update_trace_metadata(self, trace_id: str, metadata: Dict[str, Any], tags: List[str]) -> None:
        """Update trace with additional metadata and tags.
        
        Args:
            trace_id: Trace ID to update
            metadata: Additional metadata to add
            tags: Tags to apply to the trace
        """
        try:
            # This would update the trace metadata if the integration service supports it
            # For now, we log the metadata that would be added
            logger.debug(f"Trace {trace_id} metadata: {metadata}")
            logger.debug(f"Trace {trace_id} tags: {tags}")
            
        except Exception as e:
            logger.error(f"Failed to update trace metadata for {trace_id}: {e}")
    
    def track_event_propagation(self, event_id: str, agent_responses: List[Dict[str, Any]]) -> None:
        """Track how an event propagates through the agent system.
        
        Args:
            event_id: The original event ID
            agent_responses: List of agent response data
        """
        if event_id not in self._active_event_traces:
            logger.warning(f"No active trace found for event: {event_id}")
            return
        
        trace_id = self._active_event_traces[event_id]
        
        try:
            # Create spans for each agent response
            for response in agent_responses:
                agent_id = response.get("agent_id")
                if agent_id:
                    span_id = self._integration_service.start_agent_span(
                        agent_id=agent_id,
                        operation="event_response",
                        parent_trace_id=trace_id,
                        input_data={
                            "original_event_id": event_id,
                            "response_data": response.get("response", {}),
                            "processing_time": response.get("processing_time")
                        }
                    )
                    
                    if span_id:
                        # End the span with outcome data
                        self._integration_service.end_agent_span(
                            span_id=span_id,
                            outcome={
                                "actions_taken": response.get("actions", []),
                                "decisions_made": response.get("decisions", []),
                                "success": response.get("success", True)
                            }
                        )
            
            logger.debug(f"Tracked event propagation for {event_id} with {len(agent_responses)} agent responses")
            
        except Exception as e:
            logger.error(f"Failed to track event propagation for {event_id}: {e}")
    
    def finalize_event_trace(self, event_id: str, final_outcome: Dict[str, Any]) -> None:
        """Finalize an event trace with final outcome data.
        
        Args:
            event_id: The event ID to finalize
            final_outcome: Final outcome data for the event
        """
        if event_id not in self._active_event_traces:
            logger.warning(f"No active trace found for event: {event_id}")
            return
        
        trace_id = self._active_event_traces[event_id]
        
        try:
            # Finalize the trace with outcome data
            self._integration_service.finalize_trace(trace_id, final_outcome)
            
            # Remove from active traces
            del self._active_event_traces[event_id]
            
            logger.info(f"Finalized event trace: {event_id}")
            
        except Exception as e:
            logger.error(f"Failed to finalize event trace for {event_id}: {e}")
    
    def register_event_callback(self, event_type: TriggerEventType, callback: Callable[[SimulationEventData], None]) -> None:
        """Register a callback function for specific event types.
        
        Args:
            event_type: Type of event to listen for
            callback: Function to call when event occurs
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        
        self._event_callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for event type: {event_type}")
    
    def _execute_event_callbacks(self, event_type: TriggerEventType, event_data: SimulationEventData) -> None:
        """Execute registered callbacks for an event type.
        
        Args:
            event_type: Type of event that occurred
            event_data: Event data to pass to callbacks
        """
        callbacks = self._event_callbacks.get(event_type, [])
        
        for callback in callbacks:
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error executing callback for {event_type}: {e}")
    
    def get_active_event_traces(self) -> Dict[str, str]:
        """Get currently active event traces.
        
        Returns:
            Dictionary mapping event IDs to trace IDs
        """
        return self._active_event_traces.copy()
    
    def capture_market_event(self, market_event: MarketEvent) -> Optional[str]:
        """Capture a market event and create appropriate trace.
        
        Args:
            market_event: MarketEvent instance to capture
            
        Returns:
            Event ID if successful, None otherwise
        """
        trigger_data = {
            "type": TriggerEventType.MARKET_EVENT.value,
            "event_id": str(market_event.id),
            "source": "market_simulation",
            "event_type": market_event.event_type.value,
            "affected_products": market_event.affected_products,
            "impact_magnitude": market_event.impact_magnitude,
            "description": market_event.description,
            "metadata": {
                "market_event_id": str(market_event.id),
                "timestamp": market_event.timestamp.isoformat(),
                **market_event.metadata
            }
        }
        
        return self.capture_trigger_event(trigger_data)
    
    def capture_scenario_trigger(self, scenario_data: Dict[str, Any]) -> Optional[str]:
        """Capture a trigger scenario activation.
        
        Args:
            scenario_data: Scenario trigger data
            
        Returns:
            Event ID if successful, None otherwise
        """
        trigger_data = {
            "type": TriggerEventType.TRIGGER_SCENARIO.value,
            "event_id": f"scenario_{scenario_data.get('scenario_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "source": "trigger_engine",
            "scenario_id": scenario_data.get("scenario_id"),
            "scenario_name": scenario_data.get("scenario_name"),
            "intensity": scenario_data.get("intensity", "medium"),
            "conditions": scenario_data.get("conditions", {}),
            "affected_agents": [scenario_data.get("agent_type", "unknown")],
            "metadata": {
                "agent_type": scenario_data.get("agent_type"),
                "trigger_type": scenario_data.get("trigger_type"),
                "expected_effects": scenario_data.get("effects", {}),
                "cooldown_minutes": scenario_data.get("cooldown_minutes", 30)
            }
        }
        
        return self.capture_trigger_event(trigger_data)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status.
        
        Returns:
            Dictionary containing health check results
        """
        return {
            "service": "simulation_event_capture",
            "status": "healthy" if self._integration_service.is_available else "degraded",
            "active_traces": len(self._active_event_traces),
            "registered_callbacks": sum(len(callbacks) for callbacks in self._event_callbacks.values()),
            "trace_mappings": len(self._trace_mappings),
            "langfuse_available": self._integration_service.is_available
        }


# Global instance
_event_capture: Optional[SimulationEventCapture] = None


def get_simulation_event_capture() -> SimulationEventCapture:
    """Get the global simulation event capture instance."""
    global _event_capture
    if _event_capture is None:
        _event_capture = SimulationEventCapture()
    return _event_capture


def initialize_simulation_event_capture(integration_service: Optional[LangfuseIntegrationService] = None) -> SimulationEventCapture:
    """Initialize simulation event capture with optional integration service.
    
    Args:
        integration_service: Optional LangfuseIntegrationService instance
        
    Returns:
        Initialized SimulationEventCapture instance
    """
    global _event_capture
    _event_capture = SimulationEventCapture(integration_service)
    return _event_capture