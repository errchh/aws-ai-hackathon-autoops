"""
Function execution tracker for ensuring comprehensive agent function coverage.

This module tracks which agent functions have been executed during simulation
and provides coverage metrics to ensure all functions are demonstrated.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class FunctionExecutionTracker:
    """
    Tracks execution of agent functions to ensure comprehensive coverage.

    This class monitors which functions have been called, when they were executed,
    and provides metrics for demo coverage validation.
    """

    def __init__(self):
        self.executions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.function_signatures: Dict[str, Dict[str, Any]] = {}
        self.coverage_requirements: Dict[str, int] = {}
        self._initialize_function_signatures()

    def _initialize_function_signatures(self):
        """Initialize the complete set of expected agent functions."""

        # Pricing Agent functions
        pricing_functions = [
            "analyze_demand_elasticity",
            "calculate_optimal_price",
            "apply_markdown_strategy",
            "evaluate_price_impact",
            "get_competitor_prices",
            "retrieve_pricing_history",
            "make_pricing_decision",
            "update_decision_outcome",
        ]

        # Inventory Agent functions
        inventory_functions = [
            "forecast_demand_probabilistic",
            "calculate_safety_buffer",
            "generate_restock_alert",
            "identify_slow_moving_inventory",
            "analyze_demand_patterns",
            "retrieve_inventory_history",
            "make_inventory_decision",
            "update_decision_outcome",
        ]

        # Promotion Agent functions
        promotion_functions = [
            "create_flash_sale",
            "generate_bundle_recommendation",
            "analyze_social_sentiment",
            "schedule_promotional_campaign",
            "evaluate_campaign_effectiveness",
            "coordinate_with_pricing_agent",
            "validate_inventory_availability",
            "retrieve_promotion_history",
        ]

        # Orchestrator functions
        orchestrator_functions = [
            "process_market_event",
            "coordinate_agents",
            "trigger_collaboration_workflow",
            "register_agents",
            "get_system_status",
        ]

        # Collaboration functions
        collaboration_functions = [
            "inventory_to_pricing_slow_moving_alert",
            "pricing_to_promotion_discount_coordination",
            "promotion_to_inventory_stock_validation",
            "cross_agent_learning_from_outcomes",
            "collaborative_market_event_response",
        ]

        # Combine all functions
        all_functions = (
            pricing_functions
            + inventory_functions
            + promotion_functions
            + orchestrator_functions
            + collaboration_functions
        )

        # Initialize signatures and coverage requirements
        for func in all_functions:
            self.function_signatures[func] = {
                "agent_type": self._get_agent_type(func),
                "executed": False,
                "execution_count": 0,
                "last_executed": None,
                "first_executed": None,
            }
            self.coverage_requirements[func] = 1  # At least once

    def _get_agent_type(self, function_name: str) -> str:
        """Determine agent type from function name."""
        pricing_funcs = {
            "analyze_demand_elasticity",
            "calculate_optimal_price",
            "apply_markdown_strategy",
            "evaluate_price_impact",
            "get_competitor_prices",
            "retrieve_pricing_history",
            "make_pricing_decision",
            "update_decision_outcome",
        }
        inventory_funcs = {
            "forecast_demand_probabilistic",
            "calculate_safety_buffer",
            "generate_restock_alert",
            "identify_slow_moving_inventory",
            "analyze_demand_patterns",
            "retrieve_inventory_history",
            "make_inventory_decision",
            "update_decision_outcome",
        }
        promotion_funcs = {
            "create_flash_sale",
            "generate_bundle_recommendation",
            "analyze_social_sentiment",
            "schedule_promotional_campaign",
            "evaluate_campaign_effectiveness",
            "coordinate_with_pricing_agent",
            "validate_inventory_availability",
            "retrieve_promotion_history",
        }
        orchestrator_funcs = {
            "process_market_event",
            "coordinate_agents",
            "trigger_collaboration_workflow",
            "register_agents",
            "get_system_status",
        }
        collaboration_funcs = {
            "inventory_to_pricing_slow_moving_alert",
            "pricing_to_promotion_discount_coordination",
            "promotion_to_inventory_stock_validation",
            "cross_agent_learning_from_outcomes",
            "collaborative_market_event_response",
        }

        if function_name in pricing_funcs:
            return "pricing"
        elif function_name in inventory_funcs:
            return "inventory"
        elif function_name in promotion_funcs:
            return "promotion"
        elif function_name in orchestrator_funcs:
            return "orchestrator"
        elif function_name in collaboration_funcs:
            return "collaboration"
        else:
            return "unknown"

    def track_execution(
        self,
        function_name: str,
        agent_id: str,
        result: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Track the execution of an agent function.

        Args:
            function_name: Name of the function that was executed
            agent_id: ID of the agent that executed the function
            result: Result returned by the function
            context: Execution context
        """
        current_time = datetime.now(timezone.utc)

        execution_record = {
            "timestamp": current_time.isoformat(),
            "agent_id": agent_id,
            "function_name": function_name,
            "result": result,
            "context": context,
        }

        self.executions[function_name].append(execution_record)

        # Update function signature
        if function_name in self.function_signatures:
            sig = self.function_signatures[function_name]
            sig["executed"] = True
            sig["execution_count"] += 1
            sig["last_executed"] = current_time

            if sig["first_executed"] is None:
                sig["first_executed"] = current_time

        logger.info(
            f"Tracked execution: {function_name} by {agent_id} "
            f"(total executions: {len(self.executions[function_name])})"
        )

    def get_coverage_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive coverage report.

        Returns:
            Dictionary containing coverage statistics and details
        """
        total_functions = len(self.function_signatures)
        executed_functions = sum(
            1 for sig in self.function_signatures.values() if sig["executed"]
        )
        coverage_percentage = (
            (executed_functions / total_functions) * 100 if total_functions > 0 else 0
        )

        # Coverage by agent type
        coverage_by_agent = defaultdict(
            lambda: {"total": 0, "executed": 0, "percentage": 0.0}
        )

        for func_name, sig in self.function_signatures.items():
            agent_type = sig["agent_type"]
            coverage_by_agent[agent_type]["total"] += 1
            if sig["executed"]:
                coverage_by_agent[agent_type]["executed"] += 1

        for agent_stats in coverage_by_agent.values():
            if agent_stats["total"] > 0:
                agent_stats["percentage"] = (
                    agent_stats["executed"] / agent_stats["total"]
                ) * 100

        # Functions not yet executed
        not_executed = [
            func_name
            for func_name, sig in self.function_signatures.items()
            if not sig["executed"]
        ]

        # Most and least executed functions
        execution_counts = {
            func_name: len(executions)
            for func_name, executions in self.executions.items()
        }

        most_executed = (
            max(execution_counts.items(), key=lambda x: x[1])
            if execution_counts
            else None
        )
        least_executed = (
            min(execution_counts.items(), key=lambda x: x[1])
            if execution_counts
            else None
        )

        return {
            "overall_coverage": {
                "total_functions": total_functions,
                "executed_functions": executed_functions,
                "coverage_percentage": coverage_percentage,
                "not_executed_functions": not_executed,
            },
            "coverage_by_agent": dict(coverage_by_agent),
            "execution_statistics": {
                "total_executions": sum(
                    len(execs) for execs in self.executions.values()
                ),
                "most_executed_function": most_executed,
                "least_executed_function": least_executed,
                "average_executions_per_function": sum(execution_counts.values())
                / len(execution_counts)
                if execution_counts
                else 0,
            },
            "function_details": dict(self.function_signatures),
        }

    def get_missing_functions(self) -> List[str]:
        """
        Get list of functions that haven't been executed yet.

        Returns:
            List of function names that haven't been executed
        """
        return [
            func_name
            for func_name, sig in self.function_signatures.items()
            if not sig["executed"]
        ]

    def is_fully_covered(self) -> bool:
        """
        Check if all required functions have been executed at least once.

        Returns:
            True if all functions have been executed, False otherwise
        """
        return all(sig["executed"] for sig in self.function_signatures.values())

    def get_execution_history(
        self, function_name: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get execution history for a specific function or all functions.

        Args:
            function_name: Specific function to get history for, or None for all
            limit: Maximum number of records to return

        Returns:
            List of execution records
        """
        if function_name:
            history = self.executions.get(function_name, [])
        else:
            # Combine all executions and sort by timestamp
            all_executions = []
            for executions in self.executions.values():
                all_executions.extend(executions)

            # Sort by timestamp (most recent first)
            all_executions.sort(key=lambda x: x["timestamp"], reverse=True)
            history = all_executions

        return history[:limit]

    def reset_tracking(self):
        """Reset all execution tracking data."""
        self.executions.clear()
        self._initialize_function_signatures()
        logger.info("Function execution tracking reset")

    def export_coverage_data(self, filepath: str):
        """
        Export coverage data to a JSON file.

        Args:
            filepath: Path to export the coverage data
        """
        coverage_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "coverage_report": self.get_coverage_report(),
            "execution_history": self.get_execution_history(),
        }

        try:
            with open(filepath, "w") as f:
                json.dump(coverage_data, f, indent=2, default=str)
            logger.info(f"Coverage data exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export coverage data: {e}")


# Global tracker instance
function_tracker = FunctionExecutionTracker()


def track_function_execution(
    function_name: str,
    agent_id: str,
    result: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
):
    """
    Convenience function to track function execution using the global tracker.

    Args:
        function_name: Name of the function that was executed
        agent_id: ID of the agent that executed the function
        result: Result returned by the function
        context: Execution context
    """
    function_tracker.track_execution(function_name, agent_id, result, context)
