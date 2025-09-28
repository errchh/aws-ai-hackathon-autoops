"""Workflow visualization views for dashboard display."""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from config.langfuse_workflow_views import (
    WorkflowVisualization,
    WorkflowNode,
    WorkflowEdge,
    WorkflowNodeType,
    WorkflowEdgeType,
    WorkflowVisualizationBuilder,
    LangfuseWorkflowViewManager,
)


@dataclass
class WorkflowViewFilter:
    """Filter criteria for workflow views."""

    agent_ids: List[str] = field(default_factory=list)
    workflow_types: List[str] = field(default_factory=list)
    status: List[str] = field(default_factory=list)  # 'active', 'completed', 'failed'
    time_range_hours: int = 24
    include_collaboration: bool = True


@dataclass
class WorkflowMetrics:
    """Metrics for workflow visualization."""

    total_workflows: int
    active_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_duration: float
    longest_workflow: float
    shortest_workflow: float
    collaboration_rate: float


class WorkflowVisualizationRenderer:
    """Renders workflow visualizations for dashboard display."""

    def __init__(self, workflow_manager: Optional[LangfuseWorkflowViewManager] = None):
        # Reference to workflow manager for dashboard views
        from config.langfuse_workflow_views import get_workflow_view_manager

        self.workflow_manager = workflow_manager or get_workflow_view_manager()

    def get_workflow_list_view(
        self, filter_criteria: Optional[WorkflowViewFilter] = None
    ) -> Dict[str, Any]:
        """Get workflow list view data."""
        filter_criteria = filter_criteria or WorkflowViewFilter()

        workflows = self.workflow_manager.get_all_workflows()
        filtered_workflows = self._apply_filters(workflows, filter_criteria)

        workflow_list = []
        for workflow in filtered_workflows:
            workflow_list.append(
                {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "status": workflow.status,
                    "start_time": workflow.start_time.isoformat()
                    if workflow.start_time
                    else None,
                    "end_time": workflow.end_time.isoformat()
                    if workflow.end_time
                    else None,
                    "node_count": len(workflow.nodes),
                    "edge_count": len(workflow.edges),
                    "duration": self._calculate_duration(workflow),
                    "participating_agents": [
                        node.metadata.get("agent_id")
                        for node in workflow.nodes
                        if node.type
                        in [
                            WorkflowNodeType.INVENTORY_AGENT,
                            WorkflowNodeType.PRICING_AGENT,
                            WorkflowNodeType.PROMOTION_AGENT,
                        ]
                    ],
                }
            )

        return {
            "workflows": workflow_list,
            "total_count": len(workflow_list),
            "filter_criteria": {
                "agent_ids": filter_criteria.agent_ids,
                "workflow_types": filter_criteria.workflow_types,
                "status": filter_criteria.status,
                "time_range_hours": filter_criteria.time_range_hours,
            },
            "last_updated": datetime.now().isoformat(),
        }

    def get_workflow_detail_view(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed view for a specific workflow."""
        workflow = self.workflow_manager.get_workflow(workflow_id)
        if not workflow:
            return None

        # Calculate workflow metrics
        metrics = self._calculate_workflow_metrics([workflow])

        # Group nodes by type
        nodes_by_type = defaultdict(list)
        for node in workflow.nodes:
            nodes_by_type[node.type.value].append(
                {
                    "id": node.id,
                    "label": node.label,
                    "status": node.status,
                    "start_time": node.start_time.isoformat()
                    if node.start_time
                    else None,
                    "end_time": node.end_time.isoformat() if node.end_time else None,
                    "metadata": node.metadata,
                    "position": node.position,
                }
            )

        # Group edges by type
        edges_by_type = defaultdict(list)
        for edge in workflow.edges:
            edges_by_type[edge.type.value].append(
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "label": edge.label,
                    "metadata": edge.metadata,
                }
            )

        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "status": workflow.status,
            "start_time": workflow.start_time.isoformat()
            if workflow.start_time
            else None,
            "end_time": workflow.end_time.isoformat() if workflow.end_time else None,
            "duration": self._calculate_duration(workflow),
            "metrics": {
                "total_nodes": len(workflow.nodes),
                "total_edges": len(workflow.edges),
                "active_nodes": len(
                    [n for n in workflow.nodes if n.status == "active"]
                ),
                "completed_nodes": len(
                    [n for n in workflow.nodes if n.status == "completed"]
                ),
                "failed_nodes": len(
                    [n for n in workflow.nodes if n.status == "failed"]
                ),
            },
            "nodes_by_type": dict(nodes_by_type),
            "edges_by_type": dict(edges_by_type),
            "visualization_data": self._get_visualization_data(workflow),
            "last_updated": datetime.now().isoformat(),
        }

    def get_workflow_flow_view(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow flow visualization data."""
        workflow = self.workflow_manager.get_workflow(workflow_id)
        if not workflow:
            return None

        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "visualization": self._get_visualization_data(workflow),
            "last_updated": datetime.now().isoformat(),
        }

    def get_workflow_metrics_view(
        self, filter_criteria: Optional[WorkflowViewFilter] = None
    ) -> Dict[str, Any]:
        """Get workflow metrics view."""
        filter_criteria = filter_criteria or WorkflowViewFilter()

        workflows = self.workflow_manager.get_all_workflows()
        filtered_workflows = self._apply_filters(workflows, filter_criteria)

        metrics = self._calculate_workflow_metrics(filtered_workflows)

        # Calculate additional metrics
        agent_participation = self._calculate_agent_participation(filtered_workflows)
        workflow_type_distribution = self._calculate_workflow_type_distribution(
            filtered_workflows
        )

        return {
            "metrics": {
                "total_workflows": metrics.total_workflows,
                "active_workflows": metrics.active_workflows,
                "completed_workflows": metrics.completed_workflows,
                "failed_workflows": metrics.failed_workflows,
                "average_duration": round(metrics.average_duration, 2),
                "longest_workflow": round(metrics.longest_workflow, 2),
                "shortest_workflow": round(metrics.shortest_workflow, 2),
                "collaboration_rate": round(metrics.collaboration_rate, 3),
            },
            "agent_participation": agent_participation,
            "workflow_type_distribution": workflow_type_distribution,
            "last_updated": datetime.now().isoformat(),
        }

    def _apply_filters(
        self,
        workflows: List[WorkflowVisualization],
        filter_criteria: WorkflowViewFilter,
    ) -> List[WorkflowVisualization]:
        """Apply filter criteria to workflows."""
        filtered = workflows

        # Filter by time range
        if filter_criteria.time_range_hours > 0:
            cutoff_time = datetime.now() - timedelta(
                hours=filter_criteria.time_range_hours
            )
            filtered = [
                w for w in filtered if w.start_time and w.start_time >= cutoff_time
            ]

        # Filter by status
        if filter_criteria.status:
            filtered = [w for w in filtered if w.status in filter_criteria.status]

        # Filter by agent IDs
        if filter_criteria.agent_ids:
            filtered = [
                w
                for w in filtered
                if any(
                    agent_id
                    in [
                        node.metadata.get("agent_id")
                        for node in w.nodes
                        if node.type
                        in [
                            WorkflowNodeType.INVENTORY_AGENT,
                            WorkflowNodeType.PRICING_AGENT,
                            WorkflowNodeType.PROMOTION_AGENT,
                        ]
                    ]
                    for agent_id in filter_criteria.agent_ids
                )
            ]

        # Filter by workflow types (based on metadata)
        if filter_criteria.workflow_types:
            filtered = [
                w
                for w in filtered
                if w.metadata.get("workflow_type") in filter_criteria.workflow_types
            ]

        return filtered

    def _calculate_duration(self, workflow: WorkflowVisualization) -> Optional[float]:
        """Calculate workflow duration in seconds."""
        if not workflow.start_time:
            return None

        end_time = workflow.end_time or datetime.now()
        return (end_time - workflow.start_time).total_seconds()

    def _calculate_workflow_metrics(
        self, workflows: List[WorkflowVisualization]
    ) -> WorkflowMetrics:
        """Calculate metrics for a list of workflows."""
        if not workflows:
            return WorkflowMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)

        durations = []
        collaboration_count = 0

        for workflow in workflows:
            if workflow.status == "active":
                continue

            duration = self._calculate_duration(workflow)
            if duration:
                durations.append(duration)

            # Count collaboration workflows
            if any(
                node.type == WorkflowNodeType.COLLABORATION for node in workflow.nodes
            ):
                collaboration_count += 1

        total = len(workflows)
        active = len([w for w in workflows if w.status == "active"])
        completed = len([w for w in workflows if w.status == "completed"])
        failed = len([w for w in workflows if w.status == "failed"])

        avg_duration = sum(durations) / len(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        min_duration = min(durations) if durations else 0.0
        collaboration_rate = collaboration_count / total if total > 0 else 0.0

        return WorkflowMetrics(
            total_workflows=total,
            active_workflows=active,
            completed_workflows=completed,
            failed_workflows=failed,
            average_duration=avg_duration,
            longest_workflow=max_duration,
            shortest_workflow=min_duration,
            collaboration_rate=collaboration_rate,
        )

    def _calculate_agent_participation(
        self, workflows: List[WorkflowVisualization]
    ) -> Dict[str, int]:
        """Calculate agent participation statistics."""
        agent_counts = defaultdict(int)

        for workflow in workflows:
            for node in workflow.nodes:
                if node.type in [
                    WorkflowNodeType.INVENTORY_AGENT,
                    WorkflowNodeType.PRICING_AGENT,
                    WorkflowNodeType.PROMOTION_AGENT,
                ]:
                    agent_id = node.metadata.get("agent_id", "unknown")
                    agent_counts[agent_id] += 1

        return dict(agent_counts)

    def _calculate_workflow_type_distribution(
        self, workflows: List[WorkflowVisualization]
    ) -> Dict[str, int]:
        """Calculate workflow type distribution."""
        type_counts = defaultdict(int)

        for workflow in workflows:
            workflow_type = workflow.metadata.get("workflow_type", "unknown")
            type_counts[workflow_type] += 1

        return dict(type_counts)

    def _get_visualization_data(
        self, workflow: WorkflowVisualization
    ) -> Dict[str, Any]:
        """Get visualization data for workflow rendering."""
        nodes = []
        edges = []

        for node in workflow.nodes:
            nodes.append(
                {
                    "id": node.id,
                    "type": node.type.value,
                    "label": node.label,
                    "status": node.status,
                    "position": node.position,
                    "metadata": node.metadata,
                }
            )

        for edge in workflow.edges:
            edges.append(
                {
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type.value,
                    "label": edge.label,
                    "metadata": edge.metadata,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "hierarchical",  # or "force", "circular", etc.
            "node_spacing": {"x": 150, "y": 100},
        }


class LangfuseWorkflowVisualizationViewManager:
    """Manages workflow visualization views for the dashboard."""

    def __init__(self):
        self.renderer = WorkflowVisualizationRenderer()

    def get_dashboard_workflow_views(self) -> Dict[str, Any]:
        """Get all workflow view data for dashboard."""
        # Get workflows from last 24 hours
        filter_criteria = WorkflowViewFilter(time_range_hours=24)

        return {
            "list_view": self.renderer.get_workflow_list_view(filter_criteria),
            "metrics_view": self.renderer.get_workflow_metrics_view(filter_criteria),
            "last_updated": datetime.now().isoformat(),
        }

    def get_workflow_dashboard_data(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get complete dashboard data for a specific workflow."""
        detail_view = self.renderer.get_workflow_detail_view(workflow_id)
        flow_view = self.renderer.get_workflow_flow_view(workflow_id)

        if not detail_view or not flow_view:
            return None

        return {
            "workflow_id": workflow_id,
            "detail_view": detail_view,
            "flow_view": flow_view,
            "last_updated": datetime.now().isoformat(),
        }


# Global workflow visualization view manager instance
_workflow_viz_manager: Optional[LangfuseWorkflowVisualizationViewManager] = None


def get_workflow_visualization_view_manager() -> (
    LangfuseWorkflowVisualizationViewManager
):
    """Get the global workflow visualization view manager instance."""
    global _workflow_viz_manager
    if _workflow_viz_manager is None:
        _workflow_viz_manager = LangfuseWorkflowVisualizationViewManager()
    return _workflow_viz_manager


def initialize_workflow_visualization_view_manager() -> (
    LangfuseWorkflowVisualizationViewManager
):
    """Initialize the global workflow visualization view manager."""
    global _workflow_viz_manager
    _workflow_viz_manager = LangfuseWorkflowVisualizationViewManager()
    return _workflow_viz_manager
