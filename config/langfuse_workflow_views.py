"""Workflow visualization views for Langfuse dashboard integration."""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum


class WorkflowNodeType(Enum):
    """Types of workflow nodes."""

    SIMULATION_EVENT = "simulation_event"
    ORCHESTRATOR = "orchestrator"
    INVENTORY_AGENT = "inventory_agent"
    PRICING_AGENT = "pricing_agent"
    PROMOTION_AGENT = "promotion_agent"
    COLLABORATION = "collaboration"
    DECISION = "decision"
    EXTERNAL_SYSTEM = "external_system"


class WorkflowEdgeType(Enum):
    """Types of workflow edges."""

    TRIGGER = "trigger"
    COORDINATION = "coordination"
    DATA_FLOW = "data_flow"
    DECISION_FLOW = "decision_flow"
    COLLABORATION = "collaboration"


@dataclass
class WorkflowNode:
    """Represents a node in the workflow visualization."""

    id: str
    type: WorkflowNodeType
    label: str
    status: str = "active"  # 'active', 'completed', 'failed', 'pending'
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkflowEdge:
    """Represents an edge/connection in the workflow visualization."""

    id: str
    source: str
    target: str
    type: WorkflowEdgeType
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowVisualization:
    """Complete workflow visualization data."""

    workflow_id: str
    name: str
    description: str
    nodes: List[WorkflowNode] = field(default_factory=list)
    edges: List[WorkflowEdge] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "active"  # 'active', 'completed', 'failed'
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowVisualizationBuilder:
    """Builds workflow visualizations from trace data."""

    def __init__(self):
        self.node_positions = {
            WorkflowNodeType.SIMULATION_EVENT: {"x": 100.0, "y": 100.0},
            WorkflowNodeType.ORCHESTRATOR: {"x": 300.0, "y": 100.0},
            WorkflowNodeType.INVENTORY_AGENT: {"x": 500.0, "y": 50.0},
            WorkflowNodeType.PRICING_AGENT: {"x": 500.0, "y": 150.0},
            WorkflowNodeType.PROMOTION_AGENT: {"x": 500.0, "y": 250.0},
            WorkflowNodeType.COLLABORATION: {"x": 700.0, "y": 150.0},
            WorkflowNodeType.DECISION: {"x": 900.0, "y": 150.0},
        }

    def build_from_trace_data(
        self, trace_data: Dict[str, Any]
    ) -> WorkflowVisualization:
        """Build workflow visualization from Langfuse trace data."""

        workflow_id = trace_data.get("trace_id", "unknown")
        nodes = []
        edges = []

        # Create simulation event node
        simulation_node = WorkflowNode(
            id=f"sim_{workflow_id}",
            type=WorkflowNodeType.SIMULATION_EVENT,
            label="Simulation Event",
            status="completed",
            start_time=datetime.fromisoformat(
                trace_data.get("start_time", datetime.now().isoformat())
            ),
            metadata={
                "event_type": trace_data.get("event_type", "unknown"),
                "trigger_source": trace_data.get("trigger_source", "simulation"),
            },
            position=self.node_positions[WorkflowNodeType.SIMULATION_EVENT],
        )
        nodes.append(simulation_node)

        # Create orchestrator node
        orchestrator_node = WorkflowNode(
            id=f"orch_{workflow_id}",
            type=WorkflowNodeType.ORCHESTRATOR,
            label="Orchestrator",
            status="active",
            start_time=datetime.fromisoformat(
                trace_data.get("start_time", datetime.now().isoformat())
            ),
            metadata={
                "workflow_type": trace_data.get("workflow_type", "market_event"),
                "participating_agents": trace_data.get("participating_agents", []),
            },
            position=self.node_positions[WorkflowNodeType.ORCHESTRATOR],
        )
        nodes.append(orchestrator_node)

        # Create edge from simulation to orchestrator
        edges.append(
            WorkflowEdge(
                id=f"edge_sim_orch_{workflow_id}",
                source=simulation_node.id,
                target=orchestrator_node.id,
                type=WorkflowEdgeType.TRIGGER,
                label="Triggers",
                metadata={"trigger_type": "market_event"},
            )
        )

        # Add agent nodes based on participating agents
        participating_agents = trace_data.get("participating_agents", [])
        agent_nodes = {}

        for i, agent_id in enumerate(participating_agents):
            if agent_id == "inventory_agent":
                node_type = WorkflowNodeType.INVENTORY_AGENT
                label = "Inventory Agent"
            elif agent_id == "pricing_agent":
                node_type = WorkflowNodeType.PRICING_AGENT
                label = "Pricing Agent"
            elif agent_id == "promotion_agent":
                node_type = WorkflowNodeType.PROMOTION_AGENT
                label = "Promotion Agent"
            else:
                continue

            agent_node = WorkflowNode(
                id=f"{agent_id}_{workflow_id}",
                type=node_type,
                label=label,
                status="pending",
                metadata={
                    "agent_id": agent_id,
                    "operations": trace_data.get(f"{agent_id}_operations", []),
                },
                position=self.node_positions[node_type],
            )
            agent_nodes[agent_id] = agent_node
            nodes.append(agent_node)

            # Create edge from orchestrator to agent
            edges.append(
                WorkflowEdge(
                    id=f"edge_orch_{agent_id}_{workflow_id}",
                    source=orchestrator_node.id,
                    target=agent_node.id,
                    type=WorkflowEdgeType.COORDINATION,
                    label="Coordinates",
                    metadata={"coordination_type": "task_assignment"},
                )
            )

        # Add collaboration node if multiple agents
        collaboration_node = None
        if len(participating_agents) > 1:
            collaboration_node = WorkflowNode(
                id=f"collab_{workflow_id}",
                type=WorkflowNodeType.COLLABORATION,
                label="Agent Collaboration",
                status="pending",
                metadata={
                    "collaboration_type": "cross_agent_workflow",
                    "conflicts": trace_data.get("conflicts", []),
                },
                position=self.node_positions[WorkflowNodeType.COLLABORATION],
            )
            nodes.append(collaboration_node)

            # Create edges from agents to collaboration
            for agent_id in participating_agents:
                if agent_id in agent_nodes:
                    edges.append(
                        WorkflowEdge(
                            id=f"edge_{agent_id}_collab_{workflow_id}",
                            source=agent_nodes[agent_id].id,
                            target=collaboration_node.id,
                            type=WorkflowEdgeType.COLLABORATION,
                            label="Collaborates",
                            metadata={"collaboration_type": "information_sharing"},
                        )
                    )

        # Add decision node
        decision_node = WorkflowNode(
            id=f"decision_{workflow_id}",
            type=WorkflowNodeType.DECISION,
            label="Final Decision",
            status="pending",
            metadata={
                "decision_type": "optimization_decision",
                "expected_outcomes": trace_data.get("expected_outcomes", {}),
            },
            position=self.node_positions[WorkflowNodeType.DECISION],
        )
        nodes.append(decision_node)

        # Create final edges
        if collaboration_node is not None:
            edges.append(
                WorkflowEdge(
                    id=f"edge_collab_decision_{workflow_id}",
                    source=collaboration_node.id,
                    target=decision_node.id,
                    type=WorkflowEdgeType.DECISION_FLOW,
                    label="Produces",
                    metadata={"flow_type": "consensus_decision"},
                )
            )
        else:
            # Single agent workflow
            if participating_agents and participating_agents[0] in agent_nodes:
                edges.append(
                    WorkflowEdge(
                        id=f"edge_{participating_agents[0]}_decision_{workflow_id}",
                        source=agent_nodes[participating_agents[0]].id,
                        target=decision_node.id,
                        type=WorkflowEdgeType.DECISION_FLOW,
                        label="Produces",
                        metadata={"flow_type": "direct_decision"},
                    )
                )

        return WorkflowVisualization(
            workflow_id=workflow_id,
            name=f"Workflow {workflow_id}",
            description="Retail optimization workflow from simulation to decision",
            nodes=nodes,
            edges=edges,
            start_time=datetime.fromisoformat(
                trace_data.get("start_time", datetime.now().isoformat())
            ),
            status="active",
            metadata=trace_data,
        )

    def get_workflow_template(self, workflow_type: str) -> WorkflowVisualization:
        """Get a template workflow visualization for common workflow types."""

        if workflow_type == "market_event":
            return self._create_market_event_template()
        elif workflow_type == "demand_spike":
            return self._create_demand_spike_template()
        elif workflow_type == "price_optimization":
            return self._create_price_optimization_template()
        else:
            return self._create_generic_template()

    def _create_market_event_template(self) -> WorkflowVisualization:
        """Create template for market event workflows."""
        workflow_id = "template_market_event"

        nodes = [
            WorkflowNode(
                id="sim_event",
                type=WorkflowNodeType.SIMULATION_EVENT,
                label="Market Event",
                status="completed",
                position={"x": 100, "y": 100},
            ),
            WorkflowNode(
                id="orchestrator",
                type=WorkflowNodeType.ORCHESTRATOR,
                label="Orchestrator",
                status="active",
                position={"x": 300, "y": 100},
            ),
            WorkflowNode(
                id="inventory_agent",
                type=WorkflowNodeType.INVENTORY_AGENT,
                label="Inventory Agent",
                status="pending",
                position={"x": 500, "y": 50},
            ),
            WorkflowNode(
                id="pricing_agent",
                type=WorkflowNodeType.PRICING_AGENT,
                label="Pricing Agent",
                status="pending",
                position={"x": 500, "y": 150},
            ),
            WorkflowNode(
                id="promotion_agent",
                type=WorkflowNodeType.PROMOTION_AGENT,
                label="Promotion Agent",
                status="pending",
                position={"x": 500, "y": 250},
            ),
            WorkflowNode(
                id="collaboration",
                type=WorkflowNodeType.COLLABORATION,
                label="Collaboration",
                status="pending",
                position={"x": 700, "y": 150},
            ),
            WorkflowNode(
                id="decision",
                type=WorkflowNodeType.DECISION,
                label="Optimization Decision",
                status="pending",
                position={"x": 900, "y": 150},
            ),
        ]

        edges = [
            WorkflowEdge(
                "sim_orch",
                "sim_event",
                "orchestrator",
                WorkflowEdgeType.TRIGGER,
                "Triggers",
            ),
            WorkflowEdge(
                "orch_inventory",
                "orchestrator",
                "inventory_agent",
                WorkflowEdgeType.COORDINATION,
                "Coordinates",
            ),
            WorkflowEdge(
                "orch_pricing",
                "orchestrator",
                "pricing_agent",
                WorkflowEdgeType.COORDINATION,
                "Coordinates",
            ),
            WorkflowEdge(
                "orch_promotion",
                "orchestrator",
                "promotion_agent",
                WorkflowEdgeType.COORDINATION,
                "Coordinates",
            ),
            WorkflowEdge(
                "inventory_collab",
                "inventory_agent",
                "collaboration",
                WorkflowEdgeType.COLLABORATION,
                "Collaborates",
            ),
            WorkflowEdge(
                "pricing_collab",
                "pricing_agent",
                "collaboration",
                WorkflowEdgeType.COLLABORATION,
                "Collaborates",
            ),
            WorkflowEdge(
                "promotion_collab",
                "promotion_agent",
                "collaboration",
                WorkflowEdgeType.COLLABORATION,
                "Collaborates",
            ),
            WorkflowEdge(
                "collab_decision",
                "collaboration",
                "decision",
                WorkflowEdgeType.DECISION_FLOW,
                "Produces",
            ),
        ]

        return WorkflowVisualization(
            workflow_id=workflow_id,
            name="Market Event Workflow Template",
            description="Template for market event processing workflows",
            nodes=nodes,
            edges=edges,
            status="template",
        )

    def _create_demand_spike_template(self) -> WorkflowVisualization:
        """Create template for demand spike workflows."""
        workflow_id = "template_demand_spike"

        nodes = [
            WorkflowNode(
                id="demand_spike",
                type=WorkflowNodeType.SIMULATION_EVENT,
                label="Demand Spike",
                status="completed",
                position={"x": 100, "y": 100},
            ),
            WorkflowNode(
                id="orchestrator",
                type=WorkflowNodeType.ORCHESTRATOR,
                label="Orchestrator",
                status="active",
                position={"x": 300, "y": 100},
            ),
            WorkflowNode(
                id="inventory_agent",
                type=WorkflowNodeType.INVENTORY_AGENT,
                label="Inventory Agent",
                status="pending",
                position={"x": 500, "y": 75},
            ),
            WorkflowNode(
                id="pricing_agent",
                type=WorkflowNodeType.PRICING_AGENT,
                label="Pricing Agent",
                status="pending",
                position={"x": 500, "y": 125},
            ),
            WorkflowNode(
                id="decision",
                type=WorkflowNodeType.DECISION,
                label="Demand Response",
                status="pending",
                position={"x": 700, "y": 100},
            ),
        ]

        edges = [
            WorkflowEdge(
                "spike_orch",
                "demand_spike",
                "orchestrator",
                WorkflowEdgeType.TRIGGER,
                "Triggers",
            ),
            WorkflowEdge(
                "orch_inventory",
                "orchestrator",
                "inventory_agent",
                WorkflowEdgeType.COORDINATION,
                "Coordinates",
            ),
            WorkflowEdge(
                "orch_pricing",
                "orchestrator",
                "pricing_agent",
                WorkflowEdgeType.COORDINATION,
                "Coordinates",
            ),
            WorkflowEdge(
                "inventory_decision",
                "inventory_agent",
                "decision",
                WorkflowEdgeType.DECISION_FLOW,
                "Contributes",
            ),
            WorkflowEdge(
                "pricing_decision",
                "pricing_agent",
                "decision",
                WorkflowEdgeType.DECISION_FLOW,
                "Contributes",
            ),
        ]

        return WorkflowVisualization(
            workflow_id=workflow_id,
            name="Demand Spike Workflow Template",
            description="Template for demand spike response workflows",
            nodes=nodes,
            edges=edges,
            status="template",
        )

    def _create_price_optimization_template(self) -> WorkflowVisualization:
        """Create template for price optimization workflows."""
        workflow_id = "template_price_optimization"

        nodes = [
            WorkflowNode(
                id="price_trigger",
                type=WorkflowNodeType.SIMULATION_EVENT,
                label="Price Optimization Trigger",
                status="completed",
                position={"x": 100, "y": 100},
            ),
            WorkflowNode(
                id="orchestrator",
                type=WorkflowNodeType.ORCHESTRATOR,
                label="Orchestrator",
                status="active",
                position={"x": 300, "y": 100},
            ),
            WorkflowNode(
                id="pricing_agent",
                type=WorkflowNodeType.PRICING_AGENT,
                label="Pricing Agent",
                status="pending",
                position={"x": 500, "y": 100},
            ),
            WorkflowNode(
                id="decision",
                type=WorkflowNodeType.DECISION,
                label="Price Decision",
                status="pending",
                position={"x": 700, "y": 100},
            ),
        ]

        edges = [
            WorkflowEdge(
                "trigger_orch",
                "price_trigger",
                "orchestrator",
                WorkflowEdgeType.TRIGGER,
                "Triggers",
            ),
            WorkflowEdge(
                "orch_pricing",
                "orchestrator",
                "pricing_agent",
                WorkflowEdgeType.COORDINATION,
                "Coordinates",
            ),
            WorkflowEdge(
                "pricing_decision",
                "pricing_agent",
                "decision",
                WorkflowEdgeType.DECISION_FLOW,
                "Produces",
            ),
        ]

        return WorkflowVisualization(
            workflow_id=workflow_id,
            name="Price Optimization Workflow Template",
            description="Template for price optimization workflows",
            nodes=nodes,
            edges=edges,
            status="template",
        )

    def _create_generic_template(self) -> WorkflowVisualization:
        """Create a generic workflow template."""
        workflow_id = "template_generic"

        nodes = [
            WorkflowNode(
                id="start",
                type=WorkflowNodeType.SIMULATION_EVENT,
                label="Start Event",
                status="completed",
                position={"x": 100, "y": 100},
            ),
            WorkflowNode(
                id="orchestrator",
                type=WorkflowNodeType.ORCHESTRATOR,
                label="Orchestrator",
                status="active",
                position={"x": 300, "y": 100},
            ),
            WorkflowNode(
                id="agents",
                type=WorkflowNodeType.COLLABORATION,
                label="Agent Processing",
                status="pending",
                position={"x": 500, "y": 100},
            ),
            WorkflowNode(
                id="decision",
                type=WorkflowNodeType.DECISION,
                label="Decision",
                status="pending",
                position={"x": 700, "y": 100},
            ),
        ]

        edges = [
            WorkflowEdge(
                "start_orch",
                "start",
                "orchestrator",
                WorkflowEdgeType.TRIGGER,
                "Triggers",
            ),
            WorkflowEdge(
                "orch_agents",
                "orchestrator",
                "agents",
                WorkflowEdgeType.COORDINATION,
                "Coordinates",
            ),
            WorkflowEdge(
                "agents_decision",
                "agents",
                "decision",
                WorkflowEdgeType.DECISION_FLOW,
                "Produces",
            ),
        ]

        return WorkflowVisualization(
            workflow_id=workflow_id,
            name="Generic Workflow Template",
            description="Generic workflow template for various scenarios",
            nodes=nodes,
            edges=edges,
            status="template",
        )

    def export_to_json(self, workflow: WorkflowVisualization) -> str:
        """Export workflow visualization to JSON format."""
        data = {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "start_time": workflow.start_time.isoformat()
            if workflow.start_time
            else None,
            "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type.value,
                    "label": node.label,
                    "status": node.status,
                    "start_time": node.start_time.isoformat()
                    if node.start_time
                    else None,
                    "end_time": node.end_time.isoformat() if node.end_time else None,
                    "metadata": node.metadata,
                    "position": node.position,
                }
                for node in workflow.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type.value,
                    "label": edge.label,
                    "metadata": edge.metadata,
                }
                for edge in workflow.edges
            ],
            "metadata": workflow.metadata,
        }

        return json.dumps(data, indent=2)


class LangfuseWorkflowViewManager:
    """Manages workflow visualization views for the dashboard."""

    def __init__(self):
        self.builder = WorkflowVisualizationBuilder()
        self.active_workflows: Dict[str, WorkflowVisualization] = {}
        self.workflow_templates: Dict[str, WorkflowVisualization] = {}

    def register_workflow(self, workflow: WorkflowVisualization) -> None:
        """Register an active workflow for visualization."""
        self.active_workflows[workflow.workflow_id] = workflow

    def unregister_workflow(self, workflow_id: str) -> bool:
        """Unregister a workflow."""
        if workflow_id in self.active_workflows:
            del self.active_workflows[workflow_id]
            return True
        return False

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowVisualization]:
        """Get a workflow visualization by ID."""
        return self.active_workflows.get(workflow_id)

    def get_all_workflows(self) -> List[WorkflowVisualization]:
        """Get all active workflow visualizations."""
        return list(self.active_workflows.values())

    def get_workflow_templates(self) -> Dict[str, WorkflowVisualization]:
        """Get all workflow templates."""
        if not self.workflow_templates:
            self._initialize_templates()
        return self.workflow_templates

    def _initialize_templates(self) -> None:
        """Initialize workflow templates."""
        self.workflow_templates = {
            "market_event": self.builder.get_workflow_template("market_event"),
            "demand_spike": self.builder.get_workflow_template("demand_spike"),
            "price_optimization": self.builder.get_workflow_template(
                "price_optimization"
            ),
            "generic": self.builder.get_workflow_template("generic"),
        }

    def get_dashboard_workflow_data(self) -> Dict[str, Any]:
        """Get workflow data formatted for dashboard display."""
        return {
            "active_workflows": [
                {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "status": workflow.status,
                    "start_time": workflow.start_time.isoformat()
                    if workflow.start_time
                    else None,
                    "node_count": len(workflow.nodes),
                    "edge_count": len(workflow.edges),
                }
                for workflow in self.active_workflows.values()
            ],
            "workflow_templates": [
                {
                    "id": template_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "node_count": len(workflow.nodes),
                }
                for template_id, workflow in self.workflow_templates.items()
            ],
            "total_active": len(self.active_workflows),
            "last_updated": datetime.now().isoformat(),
        }


# Global workflow view manager instance
_workflow_manager: Optional[LangfuseWorkflowViewManager] = None


def get_workflow_view_manager() -> LangfuseWorkflowViewManager:
    """Get the global workflow view manager instance."""
    global _workflow_manager
    if _workflow_manager is None:
        _workflow_manager = LangfuseWorkflowViewManager()
    return _workflow_manager


def initialize_workflow_view_manager() -> LangfuseWorkflowViewManager:
    """Initialize the global workflow view manager."""
    global _workflow_manager
    _workflow_manager = LangfuseWorkflowViewManager()
    return _workflow_manager
