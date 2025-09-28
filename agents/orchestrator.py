"""
AWS Strands Agents Orchestrator for Retail Optimization System.

This module implements the RetailOptimizationOrchestrator that coordinates
communication between specialized agents, manages system workflow, and
handles conflict resolution using the AWS Strands framework.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from enum import Enum

from strands import Agent
from strands.models import BedrockModel

from agents.memory import agent_memory
from agents.pricing_agent import pricing_agent
from agents.inventory_agent import inventory_agent
from agents.promotion_agent import promotion_agent
from agents.collaboration import collaboration_workflow
from config.settings import get_settings
from config.orchestrator_tracing import get_orchestration_tracer
from models.core import AgentDecision, ActionType, SystemMetrics


# Configure logging
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of agent types in the system."""

    PRICING = "pricing_agent"
    INVENTORY = "inventory_agent"
    PROMOTION = "promotion_agent"


class MessageType(Enum):
    """Enumeration of message types for inter-agent communication."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    COORDINATION = "coordination"
    CONFLICT_RESOLUTION = "conflict_resolution"


class SystemStatus(Enum):
    """Enumeration of system status states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class RetailOptimizationOrchestrator:
    """
    AWS Strands Agents Orchestrator for coordinating retail optimization agents.

    This orchestrator manages communication between Pricing, Inventory, and Promotion
    agents, handles conflict resolution, and maintains system-wide coordination
    using the AWS Strands framework.
    """

    def __init__(self):
        """Initialize the Retail Optimization Orchestrator."""
        self.settings = get_settings()
        self.orchestrator_id = "retail_optimization_orchestrator"

        # Initialize Bedrock model for orchestrator reasoning
        self.model = BedrockModel(
            model_id=self.settings.bedrock.model_id,
            temperature=0.3,  # Lower temperature for more consistent orchestration
            max_tokens=self.settings.bedrock.max_tokens,
            streaming=False,
        )

        # Initialize message queue for inter-agent communication
        self.message_queue = asyncio.Queue()
        self.agent_messages = {}

        # Agent registry
        self.agents: Dict[str, Agent] = {}
        self.agent_status: Dict[str, Dict[str, Any]] = {}

        # System state
        self.system_status = SystemStatus.OFFLINE
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.conflict_resolution_queue: List[Dict[str, Any]] = []

        # Initialize orchestration tracer
        self.tracer = get_orchestration_tracer()

        # Performance metrics
        self.system_metrics = SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            total_revenue=0.0,
            total_profit=0.0,
            inventory_turnover=0.0,
            stockout_incidents=0,
            waste_reduction_percentage=0.0,
            price_optimization_score=0.0,
            promotion_effectiveness=0.0,
            agent_collaboration_score=0.0,
            decision_count=0,
            response_time_avg=0.0,
        )

        logger.info(
            "orchestrator_id=<%s> | Retail Optimization Orchestrator initialized",
            self.orchestrator_id,
        )

    def register_agents(self, agents: List[Agent]) -> bool:
        """
        Register agents with the orchestrator and configure communication.

        Args:
            agents: List of agents to register

        Returns:
            Boolean indicating successful registration
        """
        try:
            logger.info(
                "agent_count=<%d> | registering agents with orchestrator", len(agents)
            )

            for agent in agents:
                agent_id = getattr(agent, "agent_id", str(uuid4()))

                # Register agent
                self.agents[agent_id] = agent

                # Initialize agent status
                self.agent_status[agent_id] = {
                    "status": "active",
                    "last_heartbeat": datetime.now(timezone.utc),
                    "message_count": 0,
                    "error_count": 0,
                    "success_rate": 1.0,
                    "average_response_time": 0.0,
                }

                # Initialize agent message queue
                if agent_id not in self.agent_messages:
                    self.agent_messages[agent_id] = []

                logger.info("agent_id=<%s> | agent registered successfully", agent_id)

            # Update system status
            if len(self.agents) >= 3:  # All three agents registered
                self.system_status = SystemStatus.HEALTHY
            elif len(self.agents) >= 2:
                self.system_status = SystemStatus.DEGRADED
            else:
                self.system_status = SystemStatus.CRITICAL

            logger.info(
                "registered_agents=<%d>, system_status=<%s> | agent registration completed",
                len(self.agents),
                self.system_status.value,
            )

            return True

        except Exception as e:
            logger.error("error=<%s> | failed to register agents", str(e))
            self.system_status = SystemStatus.CRITICAL
            return False

    async def process_market_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market events and coordinate agent responses.

        Args:
            event_data: Market event information

        Returns:
            Dictionary containing orchestrated response
        """
        try:
            event_id = str(uuid4())
            event_type = event_data.get("event_type", "unknown")

            logger.info(
                "event_id=<%s>, event_type=<%s> | processing market event",
                event_id,
                event_type,
            )

            # Determine which agents should respond to this event
            responding_agents = self._determine_responding_agents(event_data)

            if not responding_agents:
                return {
                    "event_id": event_id,
                    "status": "no_response_needed",
                    "analysis": "No agents required to respond to this event",
                }

            # Create workflow for this event
            workflow_id = str(uuid4())
            self.active_workflows[workflow_id] = {
                "event_id": event_id,
                "event_data": event_data,
                "responding_agents": responding_agents,
                "status": "in_progress",
                "start_time": datetime.now(timezone.utc),
                "agent_responses": {},
                "conflicts": [],
                "resolution": None,
            }

            # Start workflow tracing with enhanced metadata
            trace_id = self.tracer.trace_workflow_start(
                workflow_id=workflow_id,
                participating_agents=responding_agents,
                event_data={
                    "event_id": event_id,
                    "event_type": event_type,
                    "event_data": event_data,
                    "orchestrator_id": self.orchestrator_id,
                    "system_status": self.system_status.value,
                    "total_registered_agents": len(self.agents)
                }
            )

            # Send coordination messages to responding agents
            coordination_tasks = []
            coordination_messages = []
            
            for agent_id in responding_agents:
                # Create coordination message
                message = {
                    "id": str(uuid4()),
                    "sender": self.orchestrator_id,
                    "recipient": agent_id,
                    "message_type": "market_event_coordination",
                    "workflow_id": workflow_id,
                    "event_data": event_data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                coordination_messages.append(message)
                
                task = self._send_coordination_message(
                    agent_id, event_data, workflow_id
                )
                coordination_tasks.append(task)

            # Trace agent coordination with enhanced context
            coordination_id = f"coord_{workflow_id}"
            coord_span_id = self.tracer.trace_agent_coordination(
                coordination_id=coordination_id,
                messages=coordination_messages,
                workflow_id=workflow_id
            )
            
            # Log orchestrator decision for agent selection
            self.tracer.log_orchestration_decision(
                decision_data={
                    "decision_type": "agent_selection",
                    "selected_agents": responding_agents,
                    "event_type": event_type,
                    "selection_criteria": self._get_agent_selection_rationale(event_data),
                    "confidence": 0.9,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                workflow_id=workflow_id
            )

            # Wait for agent responses with timeout
            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*coordination_tasks, return_exceptions=True),
                    timeout=30.0,  # 30 second timeout
                )

                # Process responses
                valid_responses = []
                for i, response in enumerate(responses):
                    agent_id = responding_agents[i]
                    if isinstance(response, Exception):
                        logger.error(
                            "agent_id=<%s>, error=<%s> | agent response failed",
                            agent_id,
                            str(response),
                        )
                        self._update_agent_error_count(agent_id)
                    else:
                        valid_responses.append((agent_id, response))
                        self.active_workflows[workflow_id]["agent_responses"][
                            agent_id
                        ] = response

                # End coordination span
                coordination_outcome = {
                    "successful_responses": len(valid_responses),
                    "failed_responses": len(responses) - len(valid_responses),
                    "agent_responses": [resp[1] for resp in valid_responses]
                }
                self.tracer.end_coordination_span(coordination_id, coordination_outcome)

                # Check for conflicts
                conflicts = self._detect_conflicts(valid_responses)
                self.active_workflows[workflow_id]["conflicts"] = conflicts

                # Resolve conflicts if any
                if conflicts:
                    # Trace conflict resolution with detailed context
                    conflict_data = {
                        "conflict_id": f"conflict_{workflow_id}",
                        "conflict_type": "agent_response_conflicts",
                        "agents": [conflict.get("agents", []) for conflict in conflicts],
                        "details": conflicts,
                        "resolution_strategy": "orchestrator_mediation",
                        "conflict_count": len(conflicts),
                        "severity_levels": [conflict.get("severity", "unknown") for conflict in conflicts]
                    }
                    conflict_span_id = self.tracer.trace_conflict_resolution(
                        conflict_data=conflict_data,
                        workflow_id=workflow_id
                    )
                    
                    # Log orchestrator decision for conflict resolution strategy
                    self.tracer.log_orchestration_decision(
                        decision_data={
                            "decision_type": "conflict_resolution_strategy",
                            "strategy": "orchestrator_mediation",
                            "conflicts_detected": len(conflicts),
                            "resolution_approach": "prioritize_based_on_event_type",
                            "confidence": 0.85,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        },
                        workflow_id=workflow_id
                    )
                    
                    resolution = await self._resolve_conflicts(conflicts, event_data, workflow_id)
                    self.active_workflows[workflow_id]["resolution"] = resolution
                    
                    # End conflict resolution span with detailed outcome
                    if conflict_span_id:
                        resolution_outcome = {
                            "conflicts_resolved": len(conflicts),
                            "resolution_method": "orchestrator_mediation",
                            "final_resolution": resolution,
                            "resolution_success": resolution.get("status") == "resolved",
                            "resolution_time": datetime.now(timezone.utc).isoformat()
                        }
                        self.tracer.end_conflict_resolution_span(
                            conflict_data["conflict_id"], 
                            resolution_outcome
                        )
                else:
                    resolution = self._synthesize_responses(valid_responses)
                    self.active_workflows[workflow_id]["resolution"] = resolution

                # Update workflow status
                self.active_workflows[workflow_id]["status"] = "completed"
                self.active_workflows[workflow_id]["end_time"] = datetime.now(
                    timezone.utc
                )

                # Update system metrics
                self._update_system_metrics(workflow_id, valid_responses)

                result = {
                    "event_id": event_id,
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "responding_agents": responding_agents,
                    "agent_responses": len(valid_responses),
                    "conflicts_detected": len(conflicts),
                    "resolution": resolution,
                    "processing_time": (
                        self.active_workflows[workflow_id]["end_time"]
                        - self.active_workflows[workflow_id]["start_time"]
                    ).total_seconds(),
                    "analysis": f"Market event processed with {len(valid_responses)} agent responses",
                }

                # Finalize workflow trace
                final_outcome = {
                    "workflow_status": "completed",
                    "event_processed": True,
                    "agent_responses_count": len(valid_responses),
                    "conflicts_detected": len(conflicts),
                    "conflicts_resolved": len(conflicts) > 0,
                    "processing_time_seconds": result["processing_time"],
                    "final_resolution": resolution
                }
                self.tracer.finalize_workflow(workflow_id, final_outcome)

                logger.info(
                    "event_id=<%s>, workflow_id=<%s>, responses=<%d> | market event processed successfully",
                    event_id,
                    workflow_id,
                    len(valid_responses),
                )

                return result

            except asyncio.TimeoutError:
                logger.error(
                    "event_id=<%s>, workflow_id=<%s> | market event processing timed out",
                    event_id,
                    workflow_id,
                )
                self.active_workflows[workflow_id]["status"] = "timeout"

                # End coordination span with timeout error
                if coord_span_id:
                    timeout_error = Exception("Agent coordination timed out")
                    self.tracer.end_coordination_span(coordination_id, {}, timeout_error)

                # Finalize workflow trace with timeout outcome
                timeout_outcome = {
                    "workflow_status": "timeout",
                    "event_processed": False,
                    "timeout_duration_seconds": 30.0,
                    "responding_agents": responding_agents
                }
                self.tracer.finalize_workflow(workflow_id, timeout_outcome)

                return {
                    "event_id": event_id,
                    "workflow_id": workflow_id,
                    "status": "timeout",
                    "analysis": "Market event processing timed out",
                }

        except Exception as e:
            logger.error(
                "event_id=<%s>, error=<%s> | failed to process market event",
                event_data.get("event_id", "unknown"),
                str(e),
            )
            
            # Finalize workflow trace with error outcome if workflow was created
            if 'workflow_id' in locals():
                error_outcome = {
                    "workflow_status": "error",
                    "event_processed": False,
                    "error_message": str(e),
                    "error_type": type(e).__name__
                }
                self.tracer.finalize_workflow(workflow_id, error_outcome)
            
            return {
                "event_id": event_data.get("event_id", "unknown"),
                "status": "error",
                "analysis": f"Error processing market event: {str(e)}",
            }

    async def coordinate_agents(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate communication between agents for collaborative decisions.

        Args:
            request: Agent coordination request

        Returns:
            Dictionary containing coordination results
        """
        try:
            coordination_id = str(uuid4())
            requesting_agent = request.get("requesting_agent")
            target_agents = request.get("target_agents", [])
            coordination_type = request.get("coordination_type", "consultation")

            logger.info(
                "coordination_id=<%s>, requesting_agent=<%s>, targets=<%s> | coordinating agents",
                coordination_id,
                requesting_agent,
                target_agents,
            )

            if not target_agents:
                return {
                    "coordination_id": coordination_id,
                    "status": "error",
                    "analysis": "No target agents specified for coordination",
                }

            # Validate agents exist and are active
            available_agents = []
            for agent_id in target_agents:
                if (
                    agent_id in self.agents
                    and self.agent_status[agent_id]["status"] == "active"
                ):
                    available_agents.append(agent_id)
                else:
                    logger.warning(
                        "agent_id=<%s> | target agent not available for coordination",
                        agent_id,
                    )

            if not available_agents:
                return {
                    "coordination_id": coordination_id,
                    "status": "error",
                    "analysis": "No target agents available for coordination",
                }

            # Send coordination messages
            coordination_tasks = []
            coordination_messages = []
            
            for agent_id in available_agents:
                message = {
                    "id": str(uuid4()),
                    "sender": requesting_agent,
                    "recipient": agent_id,
                    "message_type": MessageType.COORDINATION.value,
                    "content": request.get("content", {}),
                    "timestamp": datetime.now(timezone.utc),
                }
                coordination_messages.append(message)

                task = self._send_message_to_agent(agent_id, message)
                coordination_tasks.append(task)

            # Trace agent coordination with workflow context
            coord_span_id = self.tracer.trace_agent_coordination(
                coordination_id=coordination_id,
                messages=coordination_messages,
                workflow_id=None  # This is direct coordination, not part of a market event workflow
            )
            
            # Log orchestrator decision for coordination approach
            self.tracer.log_orchestration_decision(
                decision_data={
                    "decision_type": "coordination_approach",
                    "requesting_agent": requesting_agent,
                    "target_agents": available_agents,
                    "coordination_type": coordination_type,
                    "approach": "direct_messaging",
                    "confidence": 0.9,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )

            # Wait for responses
            try:
                responses = await asyncio.wait_for(
                    asyncio.gather(*coordination_tasks, return_exceptions=True),
                    timeout=20.0,  # 20 second timeout for coordination
                )

                # Process coordination responses
                coordination_results = []
                for i, response in enumerate(responses):
                    agent_id = available_agents[i]
                    if isinstance(response, Exception):
                        logger.error(
                            "agent_id=<%s>, error=<%s> | coordination response failed",
                            agent_id,
                            str(response),
                        )
                    else:
                        coordination_results.append(
                            {
                                "agent_id": agent_id,
                                "response": response,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                        )

                # End coordination span
                coordination_outcome = {
                    "successful_responses": len(coordination_results),
                    "failed_responses": len(responses) - len(coordination_results),
                    "coordination_results": coordination_results
                }
                self.tracer.end_coordination_span(coordination_id, coordination_outcome)

                result = {
                    "coordination_id": coordination_id,
                    "requesting_agent": requesting_agent,
                    "coordination_type": coordination_type,
                    "target_agents": available_agents,
                    "successful_responses": len(coordination_results),
                    "coordination_results": coordination_results,
                    "status": "completed" if coordination_results else "failed",
                    "analysis": f"Agent coordination completed with {len(coordination_results)} successful responses",
                }

                logger.info(
                    "coordination_id=<%s>, successful_responses=<%d> | agent coordination completed",
                    coordination_id,
                    len(coordination_results),
                )

                return result

            except asyncio.TimeoutError:
                logger.error(
                    "coordination_id=<%s> | agent coordination timed out",
                    coordination_id,
                )
                
                # End coordination span with timeout error
                timeout_error = Exception("Agent coordination timed out")
                self.tracer.end_coordination_span(coordination_id, {}, timeout_error)
                
                return {
                    "coordination_id": coordination_id,
                    "status": "timeout",
                    "analysis": "Agent coordination timed out",
                }

        except Exception as e:
            logger.error("error=<%s> | failed to coordinate agents", str(e))
            return {
                "coordination_id": str(uuid4()),
                "status": "error",
                "analysis": f"Error coordinating agents: {str(e)}",
            }

    async def trigger_collaboration_workflow(
        self, workflow_type: str, workflow_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Trigger specific collaboration workflows between agents.

        Args:
            workflow_type: Type of collaboration workflow to trigger
            workflow_data: Data required for the collaboration workflow

        Returns:
            Dictionary containing collaboration workflow results
        """
        try:
            logger.info(
                "workflow_type=<%s> | triggering collaboration workflow", workflow_type
            )

            # Start collaboration workflow tracing
            collaboration_id = f"collab_{workflow_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            participating_agents = self._get_participating_agents_for_workflow(workflow_type)
            
            trace_id = self.tracer.track_collaboration(
                workflow_id=collaboration_id,
                participating_agents=participating_agents,
                workflow_data={
                    "workflow_type": workflow_type,
                    "workflow_data": workflow_data
                }
            )

            if workflow_type == "inventory_to_pricing_slow_moving":
                slow_moving_items = workflow_data.get("slow_moving_items", [])
                result = (
                    await collaboration_workflow.inventory_to_pricing_slow_moving_alert(
                        slow_moving_items=slow_moving_items
                    )
                )

            elif workflow_type == "pricing_to_promotion_discount":
                discount_opportunities = workflow_data.get("discount_opportunities", [])
                result = await collaboration_workflow.pricing_to_promotion_discount_coordination(
                    discount_opportunities=discount_opportunities
                )

            elif workflow_type == "promotion_to_inventory_validation":
                campaign_requests = workflow_data.get("campaign_requests", [])
                result = await collaboration_workflow.promotion_to_inventory_stock_validation(
                    campaign_requests=campaign_requests
                )

            elif workflow_type == "cross_agent_learning":
                decision_outcomes = workflow_data.get("decision_outcomes", [])
                learning_context = workflow_data.get("learning_context", {})
                result = (
                    await collaboration_workflow.cross_agent_learning_from_outcomes(
                        decision_outcomes=decision_outcomes,
                        learning_context=learning_context,
                    )
                )

            elif workflow_type == "market_event_response":
                market_event = workflow_data.get("market_event", {})
                participating_agents = workflow_data.get("participating_agents")
                result = (
                    await collaboration_workflow.collaborative_market_event_response(
                        market_event=market_event,
                        participating_agents=participating_agents,
                    )
                )

            else:
                return {
                    "workflow_type": workflow_type,
                    "status": "error",
                    "analysis": f"Unknown collaboration workflow type: {workflow_type}",
                }

            # Update system metrics based on collaboration results
            if result.get("status") in ["initiated", "completed"]:
                self._update_collaboration_metrics(workflow_type, result)

            # Finalize collaboration trace
            if trace_id:
                final_outcome = {
                    "collaboration_status": result.get("status", "unknown"),
                    "workflow_type": workflow_type,
                    "participating_agents": participating_agents,
                    "result_data": result
                }
                self.tracer.finalize_workflow(collaboration_id, final_outcome)

            logger.info(
                "workflow_type=<%s>, status=<%s> | collaboration workflow completed",
                workflow_type,
                result.get("status", "unknown"),
            )

            return result

        except Exception as e:
            logger.error(
                "workflow_type=<%s>, error=<%s> | failed to trigger collaboration workflow",
                workflow_type,
                str(e),
            )
            
            # Finalize collaboration trace with error if it was started
            if 'collaboration_id' in locals() and 'trace_id' in locals() and trace_id:
                error_outcome = {
                    "collaboration_status": "error",
                    "workflow_type": workflow_type,
                    "error_message": str(e),
                    "error_type": type(e).__name__
                }
                self.tracer.finalize_workflow(collaboration_id, error_outcome)
            
            return {
                "workflow_type": workflow_type,
                "status": "error",
                "analysis": f"Error triggering collaboration workflow: {str(e)}",
            }

    def _get_participating_agents_for_workflow(self, workflow_type: str) -> List[str]:
        """
        Determine which agents participate in a specific workflow type.

        Args:
            workflow_type: Type of collaboration workflow

        Returns:
            List of agent IDs that participate in the workflow
        """
        workflow_agent_mapping = {
            "inventory_to_pricing_slow_moving": ["inventory_agent", "pricing_agent"],
            "pricing_to_promotion_discount": ["pricing_agent", "promotion_agent"],
            "promotion_to_inventory_validation": ["promotion_agent", "inventory_agent"],
            "cross_agent_learning": ["inventory_agent", "pricing_agent", "promotion_agent"],
            "market_event_response": ["inventory_agent", "pricing_agent", "promotion_agent"],
        }
        
        return workflow_agent_mapping.get(workflow_type, [])

    def _get_agent_selection_rationale(self, event_data: Dict[str, Any]) -> str:
        """
        Get rationale for agent selection based on event data.

        Args:
            event_data: Market event information

        Returns:
            String explaining agent selection rationale
        """
        event_type = event_data.get("event_type", "unknown")
        
        rationale_mapping = {
            "demand_spike": "Selected inventory and pricing agents to handle increased demand through stock management and dynamic pricing",
            "competitor_price_change": "Selected pricing agent to analyze competitive positioning and adjust prices accordingly",
            "supply_disruption": "Selected inventory agent to manage supply chain issues and find alternative suppliers",
            "social_trend": "Selected promotion agent to capitalize on social media trends through targeted campaigns",
            "iot_sensor_alert": "Selected inventory agent to respond to real-time inventory monitoring alerts",
            "weather_impact": "Selected all agents to coordinate response to weather-related market changes",
            "seasonal_demand": "Selected inventory and promotion agents to manage seasonal inventory and marketing campaigns"
        }
        
        return rationale_mapping.get(event_type, f"Selected agents based on standard response protocol for {event_type} events")

    def _update_collaboration_metrics(
        self, workflow_type: str, collaboration_result: Dict[str, Any]
    ):
        """
        Update system metrics based on collaboration workflow results.

        Args:
            workflow_type: Type of collaboration workflow
            collaboration_result: Results from the collaboration workflow
        """
        try:
            # Update agent collaboration score based on successful collaborations
            if collaboration_result.get("status") in ["initiated", "completed"]:
                current_score = self.system_metrics.agent_collaboration_score
                improvement = 0.05  # 5% improvement per successful collaboration

                self.system_metrics.agent_collaboration_score = min(
                    1.0, current_score + improvement
                )

                logger.debug(
                    "workflow_type=<%s>, new_score=<%f> | collaboration metrics updated",
                    workflow_type,
                    self.system_metrics.agent_collaboration_score,
                )

        except Exception as e:
            logger.error(
                "workflow_type=<%s>, error=<%s> | failed to update collaboration metrics",
                workflow_type,
                str(e),
            )

    def _determine_responding_agents(self, event_data: Dict[str, Any]) -> List[str]:
        """
        Determine which agents should respond to a market event.

        Args:
            event_data: Market event information

        Returns:
            List of agent IDs that should respond
        """
        event_type = event_data.get("event_type", "unknown")
        affected_products = event_data.get("affected_products", [])

        responding_agents = []

        # Determine agents based on event type
        if event_type in ["demand_spike", "competitor_price_change"]:
            responding_agents.extend(["pricing_agent", "inventory_agent"])
        elif event_type == "social_trend":
            responding_agents.extend(["promotion_agent", "pricing_agent"])
        elif event_type == "supply_disruption":
            responding_agents.extend(["inventory_agent", "pricing_agent"])
        elif event_type == "seasonal_change":
            responding_agents.extend(
                ["pricing_agent", "inventory_agent", "promotion_agent"]
            )
        else:
            # Default: all agents respond to unknown events
            responding_agents.extend(
                ["pricing_agent", "inventory_agent", "promotion_agent"]
            )

        # Filter to only include registered agents
        available_agents = [
            agent_id for agent_id in responding_agents if agent_id in self.agents
        ]

        logger.debug(
            "event_type=<%s>, responding_agents=<%s> | determined responding agents",
            event_type,
            available_agents,
        )

        return available_agents

    async def _send_coordination_message(
        self, agent_id: str, event_data: Dict[str, Any], workflow_id: str
    ) -> Dict[str, Any]:
        """
        Send coordination message to a specific agent.

        Args:
            agent_id: Target agent ID
            event_data: Market event data
            workflow_id: Workflow identifier

        Returns:
            Agent response dictionary
        """
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not registered")

            # Create coordination message
            message = {
                "message_type": "market_event_coordination",
                "workflow_id": workflow_id,
                "event_data": event_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "requesting_orchestrator": self.orchestrator_id,
            }

            # Trace inter-agent communication with enhanced metadata
            communication_id = f"comm_{self.orchestrator_id}_{agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            comm_span_id = self.tracer.trace_inter_agent_communication(
                sender_agent=self.orchestrator_id,
                recipient_agent=agent_id,
                message_data={
                    **message,
                    "communication_id": communication_id,
                    "message_size": len(str(message)),
                    "priority": "high" if event_data.get("impact_magnitude", 0) > 0.7 else "normal"
                },
                workflow_id=workflow_id
            )

            # Store message for agent
            if agent_id not in self.agent_messages:
                self.agent_messages[agent_id] = []
            self.agent_messages[agent_id].append(message)

            # Simulate agent response (in real implementation, this would invoke the actual agent)
            response = {
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "response_type": "coordination_response",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis": f"Agent {agent_id} analyzed market event {event_data.get('event_type')}",
                "recommended_actions": self._generate_mock_agent_response(
                    agent_id, event_data
                ),
                "confidence_score": 0.85,
            }

            # End communication span with enhanced response data
            if comm_span_id:
                response_data = {
                    **response,
                    "response_size": len(str(response)),
                    "processing_success": True,
                    "response_time": datetime.now(timezone.utc).isoformat()
                }
                self.tracer.end_communication_span(communication_id, response_data)

            # Update agent status
            self._update_agent_status(agent_id, "message_processed")

            return response

        except Exception as e:
            logger.error(
                "agent_id=<%s>, error=<%s> | failed to send coordination message",
                agent_id,
                str(e),
            )
            raise

    def _generate_mock_agent_response(
        self, agent_id: str, event_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate mock agent response for demonstration purposes.

        Args:
            agent_id: Agent identifier
            event_data: Market event data

        Returns:
            List of recommended actions
        """
        event_type = event_data.get("event_type", "unknown")
        affected_products = event_data.get("affected_products", [])

        if agent_id == "pricing_agent":
            if event_type == "demand_spike":
                return [
                    {
                        "action": "increase_prices",
                        "products": affected_products[:2],
                        "adjustment": "5-10%",
                        "rationale": "Capitalize on increased demand",
                    }
                ]
            elif event_type == "competitor_price_change":
                return [
                    {
                        "action": "match_competitor_prices",
                        "products": affected_products,
                        "adjustment": "match_within_5%",
                        "rationale": "Maintain competitive position",
                    }
                ]
        elif agent_id == "inventory_agent":
            if event_type == "demand_spike":
                return [
                    {
                        "action": "increase_stock_orders",
                        "products": affected_products,
                        "quantity_increase": "50%",
                        "rationale": "Prevent stockouts during demand spike",
                    }
                ]
            elif event_type == "supply_disruption":
                return [
                    {
                        "action": "find_alternative_suppliers",
                        "products": affected_products,
                        "urgency": "high",
                        "rationale": "Mitigate supply chain risk",
                    }
                ]
        elif agent_id == "promotion_agent":
            if event_type == "social_trend":
                return [
                    {
                        "action": "create_trend_campaign",
                        "products": affected_products,
                        "campaign_type": "social_media_boost",
                        "rationale": "Leverage social media trend",
                    }
                ]

        return [
            {
                "action": "monitor_situation",
                "rationale": f"Continue monitoring {event_type} event",
            }
        ]

    def _detect_conflicts(
        self, agent_responses: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Detect conflicts between agent responses.

        Args:
            agent_responses: List of (agent_id, response) tuples

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Check for pricing conflicts
        pricing_actions = []
        inventory_actions = []

        for agent_id, response in agent_responses:
            recommended_actions = response.get("recommended_actions", [])

            for action in recommended_actions:
                if agent_id == "pricing_agent" and action.get("action") in [
                    "increase_prices",
                    "decrease_prices",
                ]:
                    pricing_actions.append((agent_id, action))
                elif agent_id == "inventory_agent" and action.get("action") in [
                    "increase_stock_orders",
                    "reduce_stock_orders",
                ]:
                    inventory_actions.append((agent_id, action))

        # Detect pricing vs inventory conflicts
        for pricing_action in pricing_actions:
            for inventory_action in inventory_actions:
                pricing_products = set(pricing_action[1].get("products", []))
                inventory_products = set(inventory_action[1].get("products", []))

                if pricing_products.intersection(inventory_products):
                    if (
                        pricing_action[1].get("action") == "increase_prices"
                        and inventory_action[1].get("action") == "increase_stock_orders"
                    ):
                        # This might actually be complementary, not conflicting
                        continue

                    conflicts.append(
                        {
                            "conflict_type": "pricing_inventory_mismatch",
                            "agents": [pricing_action[0], inventory_action[0]],
                            "products": list(
                                pricing_products.intersection(inventory_products)
                            ),
                            "pricing_action": pricing_action[1],
                            "inventory_action": inventory_action[1],
                            "severity": "medium",
                        }
                    )

        return conflicts

    async def _resolve_conflicts(
        self, conflicts: List[Dict[str, Any]], event_data: Dict[str, Any], workflow_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve conflicts between agent recommendations.

        Args:
            conflicts: List of detected conflicts
            event_data: Original market event data

        Returns:
            Conflict resolution result
        """
        try:
            resolution_id = str(uuid4())
            resolutions = []

            for conflict in conflicts:
                conflict_type = conflict.get("conflict_type")
                agents = conflict.get("agents", [])
                products = conflict.get("products", [])

                if conflict_type == "pricing_inventory_mismatch":
                    # Resolve by prioritizing based on event type
                    event_type = event_data.get("event_type", "unknown")

                    if event_type == "demand_spike":
                        # Prioritize inventory increase over price increase
                        resolution = {
                            "conflict_id": str(uuid4()),
                            "resolution_type": "prioritize_inventory",
                            "decision": "Increase inventory first, then adjust pricing based on stock levels",
                            "rationale": "During demand spikes, ensuring stock availability is more critical",
                            "affected_products": products,
                            "implementation_order": [
                                "inventory_agent",
                                "pricing_agent",
                            ],
                        }
                    else:
                        # Default: coordinate both actions
                        resolution = {
                            "conflict_id": str(uuid4()),
                            "resolution_type": "coordinate_actions",
                            "decision": "Coordinate pricing and inventory actions simultaneously",
                            "rationale": "Both actions can be complementary when properly coordinated",
                            "affected_products": products,
                            "implementation_order": ["simultaneous"],
                        }

                    resolutions.append(resolution)

            result = {
                "resolution_id": resolution_id,
                "conflicts_resolved": len(conflicts),
                "resolutions": resolutions,
                "status": "resolved" if resolutions else "no_conflicts",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(
                "resolution_id=<%s>, conflicts=<%d> | conflicts resolved",
                resolution_id,
                len(conflicts),
            )

            return result

        except Exception as e:
            logger.error("error=<%s> | failed to resolve conflicts", str(e))
            return {"resolution_id": str(uuid4()), "status": "failed", "error": str(e)}

    def _synthesize_responses(
        self, agent_responses: List[Tuple[str, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        Synthesize agent responses into a coordinated plan.

        Args:
            agent_responses: List of (agent_id, response) tuples

        Returns:
            Synthesized coordination plan
        """
        synthesis_id = str(uuid4())

        # Collect all recommended actions
        all_actions = []
        agent_contributions = {}

        for agent_id, response in agent_responses:
            recommended_actions = response.get("recommended_actions", [])
            agent_contributions[agent_id] = {
                "actions_count": len(recommended_actions),
                "confidence_score": response.get("confidence_score", 0.0),
                "actions": recommended_actions,
            }

            for action in recommended_actions:
                action["contributing_agent"] = agent_id
                all_actions.append(action)

        # Create coordinated plan
        coordinated_plan = {
            "synthesis_id": synthesis_id,
            "participating_agents": list(agent_contributions.keys()),
            "total_actions": len(all_actions),
            "coordinated_actions": all_actions,
            "agent_contributions": agent_contributions,
            "implementation_priority": self._determine_action_priority(all_actions),
            "expected_outcomes": self._predict_combined_outcomes(all_actions),
            "coordination_score": self._calculate_coordination_score(
                agent_contributions
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return coordinated_plan

    def _determine_action_priority(
        self, actions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Determine implementation priority for actions."""
        priority_map = {
            "find_alternative_suppliers": 1,  # Highest priority
            "increase_stock_orders": 2,
            "create_trend_campaign": 3,
            "increase_prices": 4,
            "match_competitor_prices": 5,
            "monitor_situation": 6,  # Lowest priority
        }

        prioritized = sorted(
            actions,
            key=lambda x: priority_map.get(x.get("action", "monitor_situation"), 10),
        )

        return [
            {
                "priority": i + 1,
                "action": action.get("action"),
                "agent": action.get("contributing_agent"),
                "products": action.get("products", []),
                "rationale": action.get("rationale"),
            }
            for i, action in enumerate(prioritized)
        ]

    def _predict_combined_outcomes(
        self, actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Predict combined outcomes of all actions."""
        return {
            "revenue_impact": "positive",
            "inventory_optimization": "improved",
            "customer_satisfaction": "maintained",
            "risk_mitigation": "enhanced",
            "implementation_complexity": "medium",
        }

    def _calculate_coordination_score(
        self, agent_contributions: Dict[str, Any]
    ) -> float:
        """Calculate coordination effectiveness score."""
        if not agent_contributions:
            return 0.0

        # Base score on number of participating agents and their confidence
        participation_score = (
            len(agent_contributions) / 3.0
        )  # Normalize to max 3 agents

        avg_confidence = sum(
            contrib.get("confidence_score", 0.0)
            for contrib in agent_contributions.values()
        ) / len(agent_contributions)

        coordination_score = (participation_score * 0.4) + (avg_confidence * 0.6)

        return min(1.0, coordination_score)

    async def _send_message_to_agent(
        self, agent_id: str, message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send message to a specific agent.

        Args:
            agent_id: Target agent ID
            message: Message to send

        Returns:
            Agent response
        """
        try:
            # Store message for agent
            if agent_id not in self.agent_messages:
                self.agent_messages[agent_id] = []
            self.agent_messages[agent_id].append(message)

            # Simulate agent processing (in real implementation, this would invoke the actual agent)
            response = {
                "agent_id": agent_id,
                "message_id": message.get("id"),
                "response_type": "coordination_response",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "processed",
                "analysis": f"Agent {agent_id} processed coordination message",
            }

            # Update agent status
            self._update_agent_status(agent_id, "message_processed")

            return response

        except Exception as e:
            logger.error(
                "agent_id=<%s>, error=<%s> | failed to send message to agent",
                agent_id,
                str(e),
            )
            raise

    def _update_agent_status(self, agent_id: str, status_update: str):
        """Update agent status tracking."""
        if agent_id in self.agent_status:
            self.agent_status[agent_id]["last_heartbeat"] = datetime.now(timezone.utc)
            self.agent_status[agent_id]["message_count"] += 1

    def _update_agent_error_count(self, agent_id: str):
        """Update agent error count."""
        if agent_id in self.agent_status:
            self.agent_status[agent_id]["error_count"] += 1
            total_messages = self.agent_status[agent_id]["message_count"]
            error_count = self.agent_status[agent_id]["error_count"]
            self.agent_status[agent_id]["success_rate"] = (
                total_messages - error_count
            ) / max(1, total_messages)

    def _update_system_metrics(
        self, workflow_id: str, agent_responses: List[Tuple[str, Dict[str, Any]]]
    ):
        """Update system metrics based on workflow results."""
        try:
            # Update decision count
            self.system_metrics.decision_count += len(agent_responses)

            # Update collaboration score based on successful agent responses
            if agent_responses:
                current_score = self.system_metrics.agent_collaboration_score
                improvement = 0.02 * len(agent_responses)  # 2% per successful response
                self.system_metrics.agent_collaboration_score = min(
                    1.0, current_score + improvement
                )

            # Update timestamp
            self.system_metrics.timestamp = datetime.now(timezone.utc)

            logger.debug(
                "workflow_id=<%s>, responses=<%d> | system metrics updated",
                workflow_id,
                len(agent_responses),
            )

        except Exception as e:
            logger.error(
                "workflow_id=<%s>, error=<%s> | failed to update system metrics",
                workflow_id,
                str(e),
            )

    async def _handle_agent_message(self, message: Dict[str, Any]):
        """Handle incoming messages from agents."""
        try:
            await self.message_queue.put(message)
            logger.debug(
                "message_id=<%s> | agent message queued", message.get("id", "unknown")
            )
        except Exception as e:
            logger.error("error=<%s> | failed to handle agent message", str(e))

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and health information.

        Returns:
            Dictionary containing system status information
        """
        try:
            active_agents = sum(
                1
                for status in self.agent_status.values()
                if status["status"] == "active"
            )

            return {
                "orchestrator_id": self.orchestrator_id,
                "system_status": self.system_status.value,
                "registered_agents": len(self.agents),
                "active_agents": active_agents,
                "active_workflows": len(self.active_workflows),
                "total_messages_processed": sum(
                    status["message_count"] for status in self.agent_status.values()
                ),
                "system_metrics": {
                    "total_decisions": self.system_metrics.decision_count,
                    "collaboration_score": self.system_metrics.agent_collaboration_score,
                    "response_time_avg": self.system_metrics.response_time_avg,
                },
                "agent_health": {
                    agent_id: {
                        "status": status["status"],
                        "success_rate": status["success_rate"],
                        "last_heartbeat": status["last_heartbeat"].isoformat(),
                    }
                    for agent_id, status in self.agent_status.items()
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error("error=<%s> | failed to get system status", str(e))
            return {
                "orchestrator_id": self.orchestrator_id,
                "system_status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }


# Global orchestrator instance
retail_optimization_orchestrator = RetailOptimizationOrchestrator()
