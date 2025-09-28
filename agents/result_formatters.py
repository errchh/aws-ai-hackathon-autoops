"""
Result formatters for converting agent function outputs to dashboard display format.

This module provides formatters that convert agent function results into the
4-column dashboard decision format (trigger_detected, agent_decision_action,
value_before, value_after).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from api.routers.dashboard import DashboardDecision

logger = logging.getLogger(__name__)


class AgentResultFormatter:
    """
    Formatter for converting agent function results to dashboard decisions.

    This class provides methods to format outputs from different agent functions
    into the standardized 4-column dashboard format.
    """

    @staticmethod
    def format_pricing_decision(
        function_name: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DashboardDecision:
        """Format pricing agent function results."""

        context = context or {}
        product_id = context.get("product_id", "Unknown")
        current_time = datetime.now(timezone.utc)

        formatters = {
            "analyze_demand_elasticity": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Demand elasticity analysis requested for {product_id}",
                agent_decision_action=f"Analyzed demand elasticity: {r.get('elasticity_coefficient', 'N/A')}",
                value_before="Elasticity: Unknown",
                value_after=f"Elasticity: {r.get('elasticity_coefficient', 'N/A')} (confidence: {r.get('confidence', 0):.1%})",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "calculate_optimal_price": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Optimal price calculation for {product_id}",
                agent_decision_action=f"Calculated optimal price: ${r.get('optimal_price', 0):.2f}",
                value_before=f"Current price: ${ctx.get('current_price', 0):.2f}",
                value_after=f"Optimal price: ${r.get('optimal_price', 0):.2f}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "apply_markdown_strategy": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Markdown strategy applied to {product_id}",
                agent_decision_action=f"Applied {r.get('discount_percentage', 0):.1%} markdown",
                value_before=f"Original price: ${ctx.get('original_price', 0):.2f}",
                value_after=f"Discounted price: ${r.get('new_price', 0):.2f}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "evaluate_price_impact": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Price impact evaluation for {product_id}",
                agent_decision_action=f"Evaluated price change impact",
                value_before=f"Current demand: {ctx.get('current_demand', 0)} units",
                value_after=f"Projected demand: {r.get('projected_demand', 0)} units",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "get_competitor_prices": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Competitor price monitoring for {product_id}",
                agent_decision_action="Retrieved competitor pricing data",
                value_before="N/A",
                value_after=f"Avg competitor price: ${r.get('average_competitor_price', 0):.2f}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "retrieve_pricing_history": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Pricing history retrieval for {product_id}",
                agent_decision_action="Retrieved historical pricing data",
                value_before="N/A",
                value_after=f"Historical avg: ${r.get('historical_average', 0):.2f} ({r.get('data_points', 0)} points)",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=0.95,  # High confidence for historical data
            ),
            "make_pricing_decision": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Pricing decision required for {product_id}",
                agent_decision_action=f"Made pricing decision: {r.get('decision_type', 'Unknown')}",
                value_before=f"Price before: ${ctx.get('price_before', 0):.2f}",
                value_after=f"Price after: ${r.get('final_price', 0):.2f}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "update_decision_outcome": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Decision outcome tracking for {product_id}",
                agent_decision_action="Updated decision outcome metrics",
                value_before=f"Previous success rate: {ctx.get('previous_success_rate', 0):.1%}",
                value_after=f"Updated success rate: {r.get('new_success_rate', 0):.1%}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=0.97,  # High confidence for metrics update
            ),
        }

        formatter = formatters.get(function_name)
        if formatter:
            return formatter(result, context)
        else:
            # Generic formatter for unknown functions
            return DashboardDecision(
                trigger_detected=f"Function {function_name} executed",
                agent_decision_action=f"Executed {function_name}",
                value_before="N/A",
                value_after=str(result.get("result", "Completed")),
                timestamp=current_time,
                function_name=function_name,
                confidence_score=result.get("confidence", 0.5),
            )

    @staticmethod
    def format_inventory_decision(
        function_name: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DashboardDecision:
        """Format inventory agent function results."""

        context = context or {}
        product_id = context.get("product_id", "Unknown")
        current_time = datetime.now(timezone.utc)

        formatters = {
            "forecast_demand_probabilistic": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Demand forecasting for {product_id}",
                agent_decision_action=f"Forecasted demand: {r.get('forecasted_demand', 0):.0f} ± {r.get('uncertainty', 0):.0f}",
                value_before=f"Current demand: {ctx.get('current_demand', 0):.0f}/day",
                value_after=f"Forecast: {r.get('forecasted_demand', 0):.0f}/day",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "calculate_safety_buffer": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Safety buffer calculation for {product_id}",
                agent_decision_action=f"Calculated safety buffer: {r.get('safety_buffer', 0)} units",
                value_before=f"Min stock: {ctx.get('min_stock', 0)}",
                value_after=f"Safety buffer: {r.get('safety_buffer', 0)} units",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "generate_restock_alert": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Restock alert generated for {product_id}",
                agent_decision_action=f"Generated {r.get('urgency', 'medium')} priority restock alert",
                value_before=f"Current stock: {ctx.get('current_stock', 0)}",
                value_after=f"Recommended: {r.get('recommended_quantity', 0)} units",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "identify_slow_moving_inventory": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Slow-moving inventory analysis",
                agent_decision_action=f"Identified {len(r.get('slow_moving_items', []))} slow-moving items",
                value_before=f"Turnover: {ctx.get('current_turnover', 0):.1f}x",
                value_after=f"Slow movers: {', '.join(r.get('slow_moving_items', []))}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "analyze_demand_patterns": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Demand pattern analysis for {product_id}",
                agent_decision_action=f"Analyzed demand pattern: {r.get('pattern_type', 'Unknown')}",
                value_before="Pattern: Unknown",
                value_after=f"Pattern: {r.get('pattern_type', 'Unknown')}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "retrieve_inventory_history": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Inventory history retrieval for {product_id}",
                agent_decision_action="Retrieved inventory history data",
                value_before="N/A",
                value_after=f"Historical avg: {r.get('historical_average', 0):.0f} units",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=0.95,
            ),
            "make_inventory_decision": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Inventory decision for {product_id}",
                agent_decision_action=f"Made inventory decision: {r.get('decision_type', 'Unknown')}",
                value_before=f"Stock before: {ctx.get('stock_before', 0)}",
                value_after=f"Stock after: {r.get('final_stock', 0)}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "update_decision_outcome": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Inventory decision outcome tracking",
                agent_decision_action="Updated inventory decision metrics",
                value_before=f"Previous accuracy: {ctx.get('previous_accuracy', 0):.1%}",
                value_after=f"Updated accuracy: {r.get('new_accuracy', 0):.1%}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=0.97,
            ),
        }

        formatter = formatters.get(function_name)
        if formatter:
            return formatter(result, context)
        else:
            return DashboardDecision(
                trigger_detected=f"Function {function_name} executed",
                agent_decision_action=f"Executed {function_name}",
                value_before="N/A",
                value_after=str(result.get("result", "Completed")),
                timestamp=current_time,
                function_name=function_name,
                confidence_score=result.get("confidence", 0.5),
            )

    @staticmethod
    def format_promotion_decision(
        function_name: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DashboardDecision:
        """Format promotion agent function results."""

        context = context or {}
        current_time = datetime.now(timezone.utc)

        formatters = {
            "create_flash_sale": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Flash sale creation opportunity",
                agent_decision_action=f"Created flash sale: {r.get('discount_percentage', 0):.0%} off",
                value_before=f"Regular price: ${ctx.get('regular_price', 0):.2f}",
                value_after=f"Flash sale: ${r.get('sale_price', 0):.2f}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "generate_bundle_recommendation": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Bundle recommendation for products",
                agent_decision_action=f"Generated bundle: {r.get('bundle_name', 'Unknown')}",
                value_before="Individual pricing",
                value_after=f"Bundle discount: {r.get('discount_percentage', 0):.0%}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "analyze_social_sentiment": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Social sentiment analysis",
                agent_decision_action=f"Analyzed sentiment: {r.get('sentiment_score', 0):.1%}",
                value_before="Sentiment: Unknown",
                value_after=f"Sentiment: {r.get('sentiment_label', 'Neutral')}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "schedule_promotional_campaign": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Campaign scheduling opportunity",
                agent_decision_action=f"Scheduled campaign: {r.get('campaign_name', 'Unknown')}",
                value_before="No active campaign",
                value_after=f"Campaign duration: {r.get('duration_hours', 0)} hours",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "evaluate_campaign_effectiveness": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Campaign effectiveness evaluation",
                agent_decision_action=f"Evaluated campaign: {r.get('effectiveness_score', 0):.1%}",
                value_before="Campaign running",
                value_after=f"Effectiveness: {r.get('effectiveness_score', 0):.1%}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "coordinate_with_pricing_agent": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Cross-agent coordination request",
                agent_decision_action="Coordinated with pricing agent",
                value_before="Independent operation",
                value_after="Coordinated pricing strategy",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "validate_inventory_availability": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Inventory validation for promotion",
                agent_decision_action="Validated inventory availability",
                value_before="Inventory check pending",
                value_after=f"Available: {r.get('available_quantity', 0)} units",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "retrieve_promotion_history": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Promotion history analysis",
                agent_decision_action="Retrieved promotion history",
                value_before="N/A",
                value_after=f"Success rate: {r.get('historical_success_rate', 0):.1%}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=0.95,
            ),
        }

        formatter = formatters.get(function_name)
        if formatter:
            return formatter(result, context)
        else:
            return DashboardDecision(
                trigger_detected=f"Function {function_name} executed",
                agent_decision_action=f"Executed {function_name}",
                value_before="N/A",
                value_after=str(result.get("result", "Completed")),
                timestamp=current_time,
                function_name=function_name,
                confidence_score=result.get("confidence", 0.5),
            )

    @staticmethod
    def format_orchestrator_decision(
        function_name: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DashboardDecision:
        """Format orchestrator function results."""

        context = context or {}
        current_time = datetime.now(timezone.utc)

        formatters = {
            "process_market_event": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Market event detected: {ctx.get('event_type', 'Unknown')}",
                agent_decision_action=f"Processed market event: {r.get('event_type', 'Unknown')}",
                value_before="Normal market conditions",
                value_after=f"Event response: {r.get('response_type', 'Activated')}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "coordinate_agents": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Agent coordination required",
                agent_decision_action=f"Coordinated {len(r.get('agents_coordinated', []))} agents",
                value_before="Independent operation",
                value_after=f"Coordinated: {', '.join(r.get('agents_coordinated', []))}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "trigger_collaboration_workflow": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Complex scenario detected",
                agent_decision_action="Triggered collaboration workflow",
                value_before="Single agent response",
                value_after="Multi-agent collaboration active",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "register_agents": lambda r, ctx: DashboardDecision(
                trigger_detected=f"System initialization",
                agent_decision_action=f"Registered {r.get('agents_registered', 0)} agents",
                value_before="Agents offline",
                value_after="All agents registered and active",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=0.99,
            ),
            "get_system_status": lambda r, ctx: DashboardDecision(
                trigger_detected=f"System status monitoring",
                agent_decision_action="Retrieved system status",
                value_before="N/A",
                value_after=f"System health: {r.get('health_score', 0):.1%}",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=0.95,
            ),
        }

        formatter = formatters.get(function_name)
        if formatter:
            return formatter(result, context)
        else:
            return DashboardDecision(
                trigger_detected=f"Function {function_name} executed",
                agent_decision_action=f"Executed {function_name}",
                value_before="N/A",
                value_after=str(result.get("result", "Completed")),
                timestamp=current_time,
                function_name=function_name,
                confidence_score=result.get("confidence", 0.5),
            )

    @staticmethod
    def format_collaboration_decision(
        function_name: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> DashboardDecision:
        """Format collaboration function results."""

        context = context or {}
        current_time = datetime.now(timezone.utc)

        formatters = {
            "inventory_to_pricing_slow_moving_alert": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Slow-moving inventory alert from inventory agent",
                agent_decision_action="Sent pricing alert for slow-moving items",
                value_before="Independent pricing",
                value_after="Coordinated markdown strategy",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "pricing_to_promotion_discount_coordination": lambda r,
            ctx: DashboardDecision(
                trigger_detected=f"Discount coordination opportunity",
                agent_decision_action="Coordinated pricing discount with promotion",
                value_before="Standard pricing",
                value_after="Promotional pricing activated",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "promotion_to_inventory_stock_validation": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Promotion stock validation request",
                agent_decision_action="Validated inventory for promotion",
                value_before="Promotion pending validation",
                value_after="Inventory confirmed for promotion",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "cross_agent_learning_from_outcomes": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Decision outcomes available for learning",
                agent_decision_action="Applied cross-agent learning",
                value_before="Individual learning",
                value_after="Collaborative learning applied",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
            "collaborative_market_event_response": lambda r, ctx: DashboardDecision(
                trigger_detected=f"Complex market event",
                agent_decision_action="Executed collaborative response",
                value_before="Single agent response",
                value_after="Multi-agent coordinated response",
                timestamp=current_time,
                function_name=function_name,
                confidence_score=r.get("confidence", 0.5),
            ),
        }

        formatter = formatters.get(function_name)
        if formatter:
            return formatter(result, context)
        else:
            return DashboardDecision(
                trigger_detected=f"Function {function_name} executed",
                agent_decision_action=f"Executed {function_name}",
                value_before="N/A",
                value_after=str(result.get("result", "Completed")),
                timestamp=current_time,
                function_name=function_name,
                confidence_score=result.get("confidence", 0.5),
            )


def format_agent_result(
    agent_type: str,
    function_name: str,
    result: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> DashboardDecision:
    """
    Main function to format any agent result into dashboard decision format.

    Args:
        agent_type: Type of agent ('pricing', 'inventory', 'promotion', 'orchestrator', 'collaboration')
        function_name: Name of the function that was executed
        result: Result dictionary from the function
        context: Additional context for formatting

    Returns:
        DashboardDecision object in the 4-column format
    """
    formatter = AgentResultFormatter()

    format_methods = {
        "pricing": formatter.format_pricing_decision,
        "inventory": formatter.format_inventory_decision,
        "promotion": formatter.format_promotion_decision,
        "orchestrator": formatter.format_orchestrator_decision,
        "collaboration": formatter.format_collaboration_decision,
    }

    format_method = format_methods.get(agent_type)
    if format_method:
        return format_method(function_name, result, context)
    else:
        logger.warning(f"Unknown agent type: {agent_type}")
        return DashboardDecision(
            trigger_detected=f"Unknown agent {agent_type} function executed",
            agent_decision_action=f"Executed {function_name}",
            value_before="N/A",
            value_after=str(result.get("result", "Completed")),
            timestamp=datetime.now(timezone.utc),
            function_name=function_name,
            confidence_score=result.get("confidence", 0.5),
        )
