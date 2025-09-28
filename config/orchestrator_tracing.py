"""
Orchestrator Workflow Tracing for Langfuse Integration

This module provides tracing capabilities for the RetailOptimizationOrchestrator,
including workflow coordination, conflict resolution, and inter-agent communication.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from config.langfuse_integration import LangfuseIntegrationService, get_langfuse_integration

logger = logging.getLogger(__name__)


class OrchestrationEventType(str, Enum):
    """Types of orchestration events that can be traced."""
    MARKET_EVENT_PROCESSING = "market_event_processing"
    AGENT_COORDINATION = "agent_coordination"
    CONFLICT_DETECTION = "conflict_detection"
    CONFLICT_RESOLUTION = "conflict_resolution"
    WORKFLOW_COMPLETION = "workflow_completion"
    COLLABORATION_TRIGGER = "collaboration_trigger"
    INTER_AGENT_COMMUNICATION = "inter_agent_communication"


@dataclass
class OrchestrationContext:
    """Context information for orchestration operations."""
    workflow_id: str
    event_id: Optional[str] = None
    participating_agents: List[str] = field(default_factory=list)
    operation_type: Optional[OrchestrationEventType] = None
    start_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class OrchestrationTracer:
    """
    Traces orchestrator coordination and conflict resolution workflows.
    
    This class provides comprehensive tracing for the RetailOptimizationOrchestrator,
    capturing workflow coordination, agent communication, and conflict resolution processes.
    """
    
    def __init__(self, integration_service: Optional[LangfuseIntegrationService] = None):
        """Initialize the orchestration tracer.
        
        Args:
            integration_service: Optional Langfuse integration service instance
        """
        self._integration_service = integration_service or get_langfuse_integration()
        self._active_workflows: Dict[str, str] = {}  # workflow_id -> trace_id
        self._active_spans: Dict[str, str] = {}  # operation_id -> span_id
        
        logger.info("Initialized OrchestrationTracer")
    
    def trace_workflow_start(
        self, 
        workflow_id: str, 
        participating_agents: List[str],
        event_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Start tracing a workflow coordination process.
        
        Args:
            workflow_id: Unique identifier for the workflow
            participating_agents: List of agent IDs participating in the workflow
            event_data: Optional event data that triggered the workflow
            
        Returns:
            Trace ID if successful, None if tracing is unavailable
        """
        if not self._integration_service.is_available:
            logger.debug("Langfuse not available, skipping workflow trace")
            return None
        
        try:
            trace_data = {
                "workflow_id": workflow_id,
                "participating_agents": participating_agents,
                "agent_count": len(participating_agents),
                "event_data": event_data or {},
                "start_time": datetime.now().isoformat(),
                "type": "orchestration_workflow"
            }
            
            trace_id = self._integration_service.create_simulation_trace(trace_data)
            
            if trace_id:
                self._active_workflows[workflow_id] = trace_id
                logger.debug(f"Started workflow trace: {workflow_id} -> {trace_id}")
            
            return trace_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow trace for {workflow_id}: {e}")
            return None
    
    def trace_agent_coordination(
        self, 
        coordination_id: str, 
        messages: List[Dict[str, Any]],
        workflow_id: Optional[str] = None
    ) -> Optional[str]:
        """Trace agent coordination activities.
        
        Args:
            coordination_id: Unique identifier for the coordination activity
            messages: List of coordination messages between agents
            workflow_id: Optional parent workflow ID
            
        Returns:
            Span ID if successful, None if tracing is unavailable
        """
        if not self._integration_service.is_available:
            return None
        
        try:
            parent_trace_id = None
            if workflow_id and workflow_id in self._active_workflows:
                parent_trace_id = self._active_workflows[workflow_id]
            
            span_id = self._integration_service.start_agent_span(
                agent_id="orchestrator",
                operation="agent_coordination",
                parent_trace_id=parent_trace_id,
                input_data={
                    "coordination_id": coordination_id,
                    "message_count": len(messages),
                    "participating_agents": list(set(
                        msg.get("recipient", "") for msg in messages
                    )),
                    "coordination_messages": messages
                }
            )
            
            if span_id:
                self._active_spans[coordination_id] = span_id
                logger.debug(f"Started coordination trace: {coordination_id} -> {span_id}")
            
            return span_id
            
        except Exception as e:
            logger.error(f"Failed to trace agent coordination {coordination_id}: {e}")
            return None
    
    def trace_conflict_resolution(
        self, 
        conflict_data: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> Optional[str]:
        """Trace conflict detection and resolution processes.
        
        Args:
            conflict_data: Information about the detected conflicts
            workflow_id: Optional parent workflow ID
            
        Returns:
            Span ID if successful, None if tracing is unavailable
        """
        if not self._integration_service.is_available:
            return None
        
        try:
            conflict_id = conflict_data.get("conflict_id", f"conflict_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
            
            parent_trace_id = None
            if workflow_id and workflow_id in self._active_workflows:
                parent_trace_id = self._active_workflows[workflow_id]
            
            span_id = self._integration_service.start_agent_span(
                agent_id="orchestrator",
                operation="conflict_resolution",
                parent_trace_id=parent_trace_id,
                input_data={
                    "conflict_id": conflict_id,
                    "conflict_type": conflict_data.get("conflict_type"),
                    "involved_agents": conflict_data.get("agents", []),
                    "conflict_details": conflict_data.get("details", {}),
                    "resolution_strategy": conflict_data.get("resolution_strategy")
                }
            )
            
            if span_id:
                self._active_spans[conflict_id] = span_id
                logger.debug(f"Started conflict resolution trace: {conflict_id} -> {span_id}")
            
            return span_id
            
        except Exception as e:
            logger.error(f"Failed to trace conflict resolution: {e}")
            return None
    
    def trace_inter_agent_communication(
        self,
        sender_agent: str,
        recipient_agent: str,
        message_data: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> Optional[str]:
        """Trace communication between agents.
        
        Args:
            sender_agent: ID of the sending agent
            recipient_agent: ID of the receiving agent
            message_data: Message content and metadata
            workflow_id: Optional parent workflow ID
            
        Returns:
            Span ID if successful, None if tracing is unavailable
        """
        if not self._integration_service.is_available:
            return None
        
        try:
            communication_id = f"comm_{sender_agent}_{recipient_agent}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            parent_trace_id = None
            if workflow_id and workflow_id in self._active_workflows:
                parent_trace_id = self._active_workflows[workflow_id]
            
            span_id = self._integration_service.start_agent_span(
                agent_id="orchestrator",
                operation="inter_agent_communication",
                parent_trace_id=parent_trace_id,
                input_data={
                    "communication_id": communication_id,
                    "sender": sender_agent,
                    "recipient": recipient_agent,
                    "message_type": message_data.get("message_type"),
                    "message_content": message_data.get("content", {}),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            if span_id:
                self._active_spans[communication_id] = span_id
                logger.debug(f"Started communication trace: {communication_id} -> {span_id}")
            
            return span_id
            
        except Exception as e:
            logger.error(f"Failed to trace inter-agent communication: {e}")
            return None
    
    def end_coordination_span(
        self,
        coordination_id: str,
        outcome: Dict[str, Any],
        error: Optional[Exception] = None
    ) -> None:
        """End a coordination span with outcome data.
        
        Args:
            coordination_id: The coordination ID to end
            outcome: Coordination outcome data
            error: Optional error if coordination failed
        """
        if coordination_id not in self._active_spans:
            return
        
        try:
            span_id = self._active_spans[coordination_id]
            
            self._integration_service.end_agent_span(
                span_id=span_id,
                outcome=outcome,
                error=error
            )
            
            del self._active_spans[coordination_id]
            logger.debug(f"Ended coordination span: {coordination_id}")
            
        except Exception as e:
            logger.error(f"Failed to end coordination span {coordination_id}: {e}")
    
    def end_conflict_resolution_span(
        self,
        conflict_id: str,
        resolution_outcome: Dict[str, Any],
        error: Optional[Exception] = None
    ) -> None:
        """End a conflict resolution span with outcome data.
        
        Args:
            conflict_id: The conflict ID to end
            resolution_outcome: Resolution outcome data
            error: Optional error if resolution failed
        """
        if conflict_id not in self._active_spans:
            return
        
        try:
            span_id = self._active_spans[conflict_id]
            
            self._integration_service.end_agent_span(
                span_id=span_id,
                outcome=resolution_outcome,
                error=error
            )
            
            del self._active_spans[conflict_id]
            logger.debug(f"Ended conflict resolution span: {conflict_id}")
            
        except Exception as e:
            logger.error(f"Failed to end conflict resolution span {conflict_id}: {e}")
    
    def end_communication_span(
        self,
        communication_id: str,
        response_data: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """End an inter-agent communication span.
        
        Args:
            communication_id: The communication ID to end
            response_data: Optional response data from recipient
            error: Optional error if communication failed
        """
        if communication_id not in self._active_spans:
            return
        
        try:
            span_id = self._active_spans[communication_id]
            
            outcome = {
                "response_received": response_data is not None,
                "response_data": response_data or {},
                "communication_success": error is None
            }
            
            self._integration_service.end_agent_span(
                span_id=span_id,
                outcome=outcome,
                error=error
            )
            
            del self._active_spans[communication_id]
            logger.debug(f"Ended communication span: {communication_id}")
            
        except Exception as e:
            logger.error(f"Failed to end communication span {communication_id}: {e}")
    
    def finalize_workflow(
        self,
        workflow_id: str,
        final_outcome: Dict[str, Any]
    ) -> None:
        """Finalize a workflow trace with final outcome data.
        
        Args:
            workflow_id: The workflow ID to finalize
            final_outcome: Final workflow outcome data
        """
        if workflow_id not in self._active_workflows:
            logger.warning(f"No active workflow trace found for: {workflow_id}")
            return
        
        try:
            trace_id = self._active_workflows[workflow_id]
            
            self._integration_service.finalize_trace(trace_id, final_outcome)
            
            del self._active_workflows[workflow_id]
            logger.info(f"Finalized workflow trace: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Failed to finalize workflow {workflow_id}: {e}")
    
    def track_collaboration(
        self, 
        workflow_id: str, 
        participating_agents: List[str],
        workflow_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Track cross-agent collaboration workflows.
        
        Args:
            workflow_id: Identifier for the collaboration workflow
            participating_agents: List of agent IDs participating
            workflow_data: Optional workflow data
            
        Returns:
            Trace ID if successful, None if tracing is unavailable
        """
        if not self._integration_service.is_available:
            return None
        
        try:
            trace_id = self._integration_service.track_collaboration(
                workflow_id=workflow_id,
                participating_agents=participating_agents,
                workflow_data=workflow_data
            )
            
            if trace_id:
                self._active_workflows[workflow_id] = trace_id
                logger.debug(f"Started collaboration tracking: {workflow_id} -> {trace_id}")
            
            return trace_id
            
        except Exception as e:
            logger.error(f"Failed to track collaboration {workflow_id}: {e}")
            return None

    def log_orchestration_decision(
        self,
        decision_data: Dict[str, Any],
        workflow_id: Optional[str] = None
    ) -> None:
        """Log orchestrator decision-making process.
        
        Args:
            decision_data: Decision information and reasoning
            workflow_id: Optional workflow context
        """
        trace_context = {}
        if workflow_id and workflow_id in self._active_workflows:
            trace_context["workflow_id"] = workflow_id
            trace_context["trace_id"] = self._active_workflows[workflow_id]
        
        self._integration_service.log_agent_decision(
            agent_id="orchestrator",
            decision_data=decision_data,
            trace_context=trace_context
        )
    
    def get_active_workflows(self) -> Dict[str, str]:
        """Get currently active workflow traces.
        
        Returns:
            Dictionary mapping workflow IDs to trace IDs
        """
        return self._active_workflows.copy()
    
    def get_active_spans(self) -> Dict[str, str]:
        """Get currently active spans.
        
        Returns:
            Dictionary mapping operation IDs to span IDs
        """
        return self._active_spans.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status.
        
        Returns:
            Dictionary containing health check results
        """
        return {
            "service": "orchestration_tracer",
            "status": "healthy" if self._integration_service.is_available else "degraded",
            "active_workflows": len(self._active_workflows),
            "active_spans": len(self._active_spans),
            "langfuse_available": self._integration_service.is_available
        }


# Global instance
_orchestration_tracer: Optional[OrchestrationTracer] = None


def get_orchestration_tracer() -> OrchestrationTracer:
    """Get the global orchestration tracer instance."""
    global _orchestration_tracer
    if _orchestration_tracer is None:
        _orchestration_tracer = OrchestrationTracer()
    return _orchestration_tracer


def initialize_orchestration_tracer(integration_service: Optional[LangfuseIntegrationService] = None) -> OrchestrationTracer:
    """Initialize orchestration tracer with optional integration service.
    
    Args:
        integration_service: Optional LangfuseIntegrationService instance
        
    Returns:
        Initialized OrchestrationTracer instance
    """
    global _orchestration_tracer
    _orchestration_tracer = OrchestrationTracer(integration_service)
    return _orchestration_tracer