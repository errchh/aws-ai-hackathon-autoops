"""
Agent Collaboration Workflows for Retail Optimization System.

This module implements the collaboration workflows between Pricing, Inventory,
and Promotion agents, enabling coordinated decision-making and cross-agent learning.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from enum import Enum

from agents.memory import agent_memory
from config.langfuse_integration import get_langfuse_integration
from models.core import AgentDecision, ActionType


# Configure logging
logger = logging.getLogger(__name__)


class CollaborationType(Enum):
    """Types of collaboration between agents."""

    INVENTORY_TO_PRICING = "inventory_to_pricing"
    PRICING_TO_PROMOTION = "pricing_to_promotion"
    PROMOTION_TO_INVENTORY = "promotion_to_inventory"
    MARKET_EVENT_RESPONSE = "market_event_response"
    CROSS_AGENT_LEARNING = "cross_agent_learning"


class CollaborationWorkflow:
    """
    Manages collaboration workflows between retail optimization agents.

    This class implements the coordination patterns for agent collaboration
    including slow-moving inventory alerts, discount coordination, stock
    validation, and shared learning from decision outcomes.
    """

    def __init__(self):
        """Initialize the collaboration workflow manager."""
        self.workflow_id = "collaboration_workflow"
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: List[Dict[str, Any]] = []

        # Initialize Langfuse integration service
        self.langfuse_integration = get_langfuse_integration()

        logger.info(
            "workflow_id=<%s> | Collaboration workflow manager initialized",
            self.workflow_id,
        )

    async def inventory_to_pricing_slow_moving_alert(
        self,
        slow_moving_items: List[Dict[str, Any]],
        inventory_agent_id: str = "inventory_agent",
        pricing_agent_id: str = "pricing_agent",
    ) -> Dict[str, Any]:
        """
        Create Inventory-to-Pricing agent communication for slow-moving items.

        Args:
            slow_moving_items: List of slow-moving inventory items
            inventory_agent_id: ID of the inventory agent
            pricing_agent_id: ID of the pricing agent

        Returns:
            Dictionary containing collaboration results
        """
        # Start Langfuse span for collaboration
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.workflow_id,
            operation="inventory_to_pricing_slow_moving_alert",
            input_data={
                "slow_moving_items_count": len(slow_moving_items),
                "inventory_agent_id": inventory_agent_id,
                "pricing_agent_id": pricing_agent_id,
            },
        )

        try:
            collaboration_id = str(uuid4())
            logger.info(
                "collaboration_id=<%s>, items_count=<%d> | initiating inventory-to-pricing slow-moving alert",
                collaboration_id,
                len(slow_moving_items),
            )

            # Create collaboration context
            collaboration_context = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.INVENTORY_TO_PRICING.value,
                "initiating_agent": inventory_agent_id,
                "target_agent": pricing_agent_id,
                "timestamp": datetime.now(timezone.utc),
                "slow_moving_items": slow_moving_items,
                "urgency_level": self._calculate_urgency_level(slow_moving_items),
            }

            # Store collaboration in active workflows
            self.active_collaborations[collaboration_id] = collaboration_context

            # Generate pricing recommendations for each slow-moving item
            pricing_recommendations = []

            for item in slow_moving_items:
                product_id = item.get("product_id")
                days_without_sale = item.get("days_without_sale", 0)
                inventory_value = item.get("inventory_value", 0)
                current_stock = item.get("current_stock", 0)

                # Determine markdown urgency based on inventory metrics
                if days_without_sale > 60:
                    markdown_urgency = "critical"
                elif days_without_sale > 30:
                    markdown_urgency = "high"
                elif days_without_sale > 14:
                    markdown_urgency = "medium"
                else:
                    markdown_urgency = "low"

                # Calculate recommended markdown percentage
                base_markdown = {
                    "low": 10,
                    "medium": 20,
                    "high": 30,
                    "critical": 40,
                }.get(markdown_urgency, 15)

                # Adjust based on inventory value and stock level
                if inventory_value > 1000:  # High-value items
                    adjusted_markdown = base_markdown * 0.8  # More conservative
                elif current_stock > 50:  # High stock levels
                    adjusted_markdown = base_markdown * 1.2  # More aggressive
                else:
                    adjusted_markdown = base_markdown

                pricing_recommendation = {
                    "product_id": product_id,
                    "current_situation": {
                        "days_without_sale": days_without_sale,
                        "current_stock": current_stock,
                        "inventory_value": inventory_value,
                    },
                    "recommended_action": "apply_markdown",
                    "markdown_urgency": markdown_urgency,
                    "recommended_markdown_percentage": round(adjusted_markdown, 1),
                    "rationale": f"Item has been without sale for {days_without_sale} days with ${inventory_value:.2f} tied up in inventory",
                    "expected_outcome": {
                        "inventory_turnover_improvement": "30-50%",
                        "cash_flow_improvement": "immediate",
                        "risk_mitigation": "prevent_obsolescence",
                    },
                }

                pricing_recommendations.append(pricing_recommendation)

            # Create collaboration message for pricing agent
            collaboration_message = {
                "message_type": "slow_moving_inventory_alert",
                "from_agent": inventory_agent_id,
                "to_agent": pricing_agent_id,
                "collaboration_id": collaboration_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "urgency_level": collaboration_context["urgency_level"],
                "total_items": len(slow_moving_items),
                "total_inventory_value": sum(
                    item.get("inventory_value", 0) for item in slow_moving_items
                ),
                "pricing_recommendations": pricing_recommendations,
                "coordination_request": {
                    "action_required": "review_and_implement_markdowns",
                    "timeline": "within_24_hours"
                    if collaboration_context["urgency_level"] in ["high", "critical"]
                    else "within_72_hours",
                    "success_metrics": [
                        "inventory_turnover_increase",
                        "cash_flow_improvement",
                        "waste_reduction",
                    ],
                },
            }

            # Store collaboration in memory for learning
            memory_context = {
                "collaboration_type": CollaborationType.INVENTORY_TO_PRICING.value,
                "slow_moving_items_count": len(slow_moving_items),
                "total_value_at_risk": sum(
                    item.get("inventory_value", 0) for item in slow_moving_items
                ),
                "urgency_level": collaboration_context["urgency_level"],
            }

            memory_id = agent_memory.store_decision(
                agent_id=inventory_agent_id,
                decision=AgentDecision(
                    agent_id=inventory_agent_id,
                    action_type=ActionType.COLLABORATION_REQUEST,
                    parameters={
                        "collaboration_id": collaboration_id,
                        "target_agent": pricing_agent_id,
                        "items_count": len(slow_moving_items),
                    },
                    rationale=f"Initiated collaboration with pricing agent for {len(slow_moving_items)} slow-moving items",
                    confidence_score=0.9,
                    expected_outcome={
                        "markdown_implementations": len(pricing_recommendations),
                        "inventory_turnover_improvement": "significant",
                    },
                ),
                context=memory_context,
            )

            # Update collaboration status
            collaboration_context["status"] = "message_sent"
            collaboration_context["memory_id"] = memory_id
            collaboration_context["message"] = collaboration_message

            result = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.INVENTORY_TO_PRICING.value,
                "status": "initiated",
                "message_sent": True,
                "items_processed": len(slow_moving_items),
                "pricing_recommendations": len(pricing_recommendations),
                "urgency_level": collaboration_context["urgency_level"],
                "expected_response_time": collaboration_message["coordination_request"][
                    "timeline"
                ],
                "collaboration_message": collaboration_message,
                "analysis": f"Initiated collaboration for {len(slow_moving_items)} slow-moving items with total value ${sum(item.get('inventory_value', 0) for item in slow_moving_items):,.2f}",
            }

            logger.info(
                "collaboration_id=<%s>, recommendations=<%d> | inventory-to-pricing collaboration initiated",
                collaboration_id,
                len(pricing_recommendations),
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "collaboration_id": collaboration_id,
                    "items_processed": len(slow_moving_items),
                    "pricing_recommendations_count": len(pricing_recommendations),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "error=<%s> | failed to initiate inventory-to-pricing collaboration",
                str(e),
            )

            # End Langfuse span with error outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "success": False,
                    "error": str(e),
                },
                error=e,
            )

            return {
                "collaboration_id": str(uuid4()),
                "status": "failed",
                "analysis": f"Error initiating collaboration: {str(e)}",
            }

    async def pricing_to_promotion_discount_coordination(
        self,
        discount_opportunities: List[Dict[str, Any]],
        pricing_agent_id: str = "pricing_agent",
        promotion_agent_id: str = "promotion_agent",
    ) -> Dict[str, Any]:
        """
        Implement Pricing-to-Promotion agent coordination for discount strategies.

        Args:
            discount_opportunities: List of products with discount opportunities
            pricing_agent_id: ID of the pricing agent
            promotion_agent_id: ID of the promotion agent

        Returns:
            Dictionary containing coordination results
        """
        # Start Langfuse span for collaboration
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.workflow_id,
            operation="pricing_to_promotion_discount_coordination",
            input_data={
                "discount_opportunities_count": len(discount_opportunities),
                "pricing_agent_id": pricing_agent_id,
                "promotion_agent_id": promotion_agent_id,
            },
        )

        try:
            collaboration_id = str(uuid4())
            logger.info(
                "collaboration_id=<%s>, opportunities=<%d> | initiating pricing-to-promotion discount coordination",
                collaboration_id,
                len(discount_opportunities),
            )

            # Create collaboration context
            collaboration_context = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.PRICING_TO_PROMOTION.value,
                "initiating_agent": pricing_agent_id,
                "target_agent": promotion_agent_id,
                "timestamp": datetime.now(timezone.utc),
                "discount_opportunities": discount_opportunities,
            }

            # Store collaboration in active workflows
            self.active_collaborations[collaboration_id] = collaboration_context

            # Generate promotional campaign recommendations
            campaign_recommendations = []

            for opportunity in discount_opportunities:
                product_id = opportunity.get("product_id")
                current_price = opportunity.get("current_price", 0)
                proposed_discount = opportunity.get("discount_percentage", 0)
                discount_reason = opportunity.get("reason", "pricing_optimization")

                # Determine optimal promotional strategy based on discount reason
                if discount_reason == "slow_moving_inventory":
                    campaign_type = "clearance_sale"
                    campaign_duration = 7  # 1 week
                    target_audience = "bargain_hunters"
                elif discount_reason == "competitive_pressure":
                    campaign_type = "price_match_promotion"
                    campaign_duration = 3  # 3 days
                    target_audience = "price_conscious"
                elif discount_reason == "demand_stimulation":
                    campaign_type = "flash_sale"
                    campaign_duration = 1  # 24 hours
                    target_audience = "general"
                else:
                    campaign_type = "general_promotion"
                    campaign_duration = 5  # 5 days
                    target_audience = "general"

                # Calculate promotional impact
                expected_demand_increase = min(
                    300, proposed_discount * 8
                )  # Higher discount = higher demand
                expected_revenue_impact = (
                    current_price
                    * (1 - proposed_discount / 100)
                    * expected_demand_increase
                    / 100
                )

                campaign_recommendation = {
                    "product_id": product_id,
                    "discount_context": {
                        "current_price": current_price,
                        "proposed_discount": proposed_discount,
                        "discount_reason": discount_reason,
                    },
                    "recommended_campaign": {
                        "type": campaign_type,
                        "duration_days": campaign_duration,
                        "target_audience": target_audience,
                        "promotional_channels": self._get_optimal_channels(
                            campaign_type
                        ),
                        "messaging_strategy": self._get_messaging_strategy(
                            campaign_type, proposed_discount
                        ),
                    },
                    "expected_impact": {
                        "demand_increase_percentage": expected_demand_increase,
                        "revenue_impact": round(expected_revenue_impact, 2),
                        "inventory_turnover": "accelerated",
                    },
                    "coordination_requirements": {
                        "pricing_confirmation": "required",
                        "inventory_validation": "required",
                        "campaign_timing": "coordinate_with_pricing_implementation",
                    },
                }

                campaign_recommendations.append(campaign_recommendation)

            # Create collaboration message for promotion agent
            collaboration_message = {
                "message_type": "discount_coordination_request",
                "from_agent": pricing_agent_id,
                "to_agent": promotion_agent_id,
                "collaboration_id": collaboration_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "discount_opportunities_count": len(discount_opportunities),
                "total_potential_impact": sum(
                    rec["expected_impact"]["revenue_impact"]
                    for rec in campaign_recommendations
                ),
                "campaign_recommendations": campaign_recommendations,
                "coordination_request": {
                    "action_required": "create_coordinated_promotional_campaigns",
                    "timeline": "coordinate_with_pricing_changes",
                    "success_metrics": [
                        "campaign_conversion_rate",
                        "inventory_turnover_acceleration",
                        "revenue_optimization",
                    ],
                },
            }

            # Store collaboration in memory
            memory_context = {
                "collaboration_type": CollaborationType.PRICING_TO_PROMOTION.value,
                "discount_opportunities_count": len(discount_opportunities),
                "campaign_recommendations_count": len(campaign_recommendations),
                "total_potential_impact": collaboration_message[
                    "total_potential_impact"
                ],
            }

            memory_id = agent_memory.store_decision(
                agent_id=pricing_agent_id,
                decision=AgentDecision(
                    agent_id=pricing_agent_id,
                    action_type=ActionType.COLLABORATION_REQUEST,
                    parameters={
                        "collaboration_id": collaboration_id,
                        "target_agent": promotion_agent_id,
                        "opportunities_count": len(discount_opportunities),
                    },
                    rationale=f"Initiated promotional coordination for {len(discount_opportunities)} discount opportunities",
                    confidence_score=0.85,
                    expected_outcome={
                        "promotional_campaigns": len(campaign_recommendations),
                        "revenue_optimization": "significant",
                    },
                ),
                context=memory_context,
            )

            # Update collaboration status
            collaboration_context["status"] = "message_sent"
            collaboration_context["memory_id"] = memory_id
            collaboration_context["message"] = collaboration_message

            result = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.PRICING_TO_PROMOTION.value,
                "status": "initiated",
                "message_sent": True,
                "opportunities_processed": len(discount_opportunities),
                "campaign_recommendations": len(campaign_recommendations),
                "total_potential_impact": collaboration_message[
                    "total_potential_impact"
                ],
                "collaboration_message": collaboration_message,
                "analysis": f"Initiated promotional coordination for {len(discount_opportunities)} discount opportunities with ${collaboration_message['total_potential_impact']:,.2f} potential impact",
            }

            logger.info(
                "collaboration_id=<%s>, campaigns=<%d> | pricing-to-promotion coordination initiated",
                collaboration_id,
                len(campaign_recommendations),
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "collaboration_id": collaboration_id,
                    "opportunities_processed": len(discount_opportunities),
                    "campaign_recommendations_count": len(campaign_recommendations),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "error=<%s> | failed to initiate pricing-to-promotion coordination",
                str(e),
            )

            # End Langfuse span with error outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "success": False,
                    "error": str(e),
                },
                error=e,
            )

            return {
                "collaboration_id": str(uuid4()),
                "status": "failed",
                "analysis": f"Error initiating coordination: {str(e)}",
            }

    async def promotion_to_inventory_stock_validation(
        self,
        campaign_requests: List[Dict[str, Any]],
        promotion_agent_id: str = "promotion_agent",
        inventory_agent_id: str = "inventory_agent",
    ) -> Dict[str, Any]:
        """
        Add Promotion-to-Inventory agent validation for stock availability.

        Args:
            campaign_requests: List of promotional campaigns requiring stock validation
            promotion_agent_id: ID of the promotion agent
            inventory_agent_id: ID of the inventory agent

        Returns:
            Dictionary containing stock validation results
        """
        # Start Langfuse span for collaboration
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.workflow_id,
            operation="promotion_to_inventory_stock_validation",
            input_data={
                "campaign_requests_count": len(campaign_requests),
                "promotion_agent_id": promotion_agent_id,
                "inventory_agent_id": inventory_agent_id,
            },
        )

        try:
            collaboration_id = str(uuid4())
            logger.info(
                "collaboration_id=<%s>, campaigns=<%d> | initiating promotion-to-inventory stock validation",
                collaboration_id,
                len(campaign_requests),
            )

            # Create collaboration context
            collaboration_context = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.PROMOTION_TO_INVENTORY.value,
                "initiating_agent": promotion_agent_id,
                "target_agent": inventory_agent_id,
                "timestamp": datetime.now(timezone.utc),
                "campaign_requests": campaign_requests,
            }

            # Store collaboration in active workflows
            self.active_collaborations[collaboration_id] = collaboration_context

            # Validate stock availability for each campaign
            validation_results = []
            total_stock_required = 0

            for campaign in campaign_requests:
                campaign_id = campaign.get("campaign_id", str(uuid4()))
                product_ids = campaign.get("product_ids", [])
                expected_demand_increase = campaign.get("expected_demand_increase", 100)
                campaign_duration = campaign.get("duration_days", 7)

                # Calculate stock requirements for campaign
                campaign_stock_requirements = []
                campaign_feasible = True

                for product_id in product_ids:
                    # Get current inventory data (simulated)
                    current_stock = campaign.get("current_inventory", {}).get(
                        product_id, 50
                    )
                    daily_demand = campaign.get("daily_demand", {}).get(product_id, 5)

                    # Calculate expected demand during campaign
                    base_campaign_demand = daily_demand * campaign_duration
                    increased_demand = base_campaign_demand * (
                        1 + expected_demand_increase / 100
                    )

                    # Check stock sufficiency
                    stock_sufficient = current_stock >= increased_demand
                    if not stock_sufficient:
                        campaign_feasible = False

                    # Calculate recommended actions
                    recommended_actions = []
                    if not stock_sufficient:
                        shortage = increased_demand - current_stock
                        recommended_actions.append(
                            f"Emergency restock: {int(shortage)} units"
                        )
                        recommended_actions.append("Consider reducing campaign scope")
                    elif current_stock < increased_demand * 1.2:  # Less than 20% buffer
                        recommended_actions.append("Monitor stock levels closely")
                        recommended_actions.append("Prepare contingency restock")
                    else:
                        recommended_actions.append("Stock levels adequate")

                    stock_requirement = {
                        "product_id": product_id,
                        "current_stock": current_stock,
                        "expected_campaign_demand": int(increased_demand),
                        "stock_sufficient": stock_sufficient,
                        "shortage_amount": max(
                            0, int(increased_demand - current_stock)
                        ),
                        "buffer_percentage": round(
                            (current_stock - increased_demand) / increased_demand * 100,
                            1,
                        )
                        if increased_demand > 0
                        else 100,
                        "recommended_actions": recommended_actions,
                        "risk_level": "high"
                        if not stock_sufficient
                        else "medium"
                        if current_stock < increased_demand * 1.2
                        else "low",
                    }

                    campaign_stock_requirements.append(stock_requirement)
                    total_stock_required += increased_demand

                # Generate overall campaign validation
                campaign_validation = {
                    "campaign_id": campaign_id,
                    "campaign_type": campaign.get("type", "general"),
                    "products_count": len(product_ids),
                    "campaign_feasible": campaign_feasible,
                    "stock_requirements": campaign_stock_requirements,
                    "total_expected_demand": int(
                        sum(
                            req["expected_campaign_demand"]
                            for req in campaign_stock_requirements
                        )
                    ),
                    "products_with_shortage": sum(
                        1
                        for req in campaign_stock_requirements
                        if not req["stock_sufficient"]
                    ),
                    "overall_risk_level": "high"
                    if not campaign_feasible
                    else "medium"
                    if any(
                        req["risk_level"] == "medium"
                        for req in campaign_stock_requirements
                    )
                    else "low",
                    "recommendations": self._generate_campaign_stock_recommendations(
                        campaign_feasible, campaign_stock_requirements
                    ),
                }

                validation_results.append(campaign_validation)

            # Create collaboration message for inventory agent
            collaboration_message = {
                "message_type": "stock_validation_request",
                "from_agent": promotion_agent_id,
                "to_agent": inventory_agent_id,
                "collaboration_id": collaboration_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "campaigns_count": len(campaign_requests),
                "total_stock_required": int(total_stock_required),
                "feasible_campaigns": sum(
                    1 for result in validation_results if result["campaign_feasible"]
                ),
                "validation_results": validation_results,
                "coordination_request": {
                    "action_required": "validate_and_reserve_inventory",
                    "timeline": "immediate_for_campaign_planning",
                    "success_metrics": [
                        "campaign_stock_availability",
                        "inventory_reservation_accuracy",
                        "stockout_prevention",
                    ],
                },
            }

            # Store collaboration in memory
            memory_context = {
                "collaboration_type": CollaborationType.PROMOTION_TO_INVENTORY.value,
                "campaigns_validated": len(campaign_requests),
                "feasible_campaigns": collaboration_message["feasible_campaigns"],
                "total_stock_required": total_stock_required,
            }

            memory_id = agent_memory.store_decision(
                agent_id=promotion_agent_id,
                decision=AgentDecision(
                    agent_id=promotion_agent_id,
                    action_type=ActionType.COLLABORATION_REQUEST,
                    parameters={
                        "collaboration_id": collaboration_id,
                        "target_agent": inventory_agent_id,
                        "campaigns_count": len(campaign_requests),
                    },
                    rationale=f"Initiated stock validation for {len(campaign_requests)} promotional campaigns",
                    confidence_score=0.9,
                    expected_outcome={
                        "validated_campaigns": len(validation_results),
                        "inventory_coordination": "improved",
                    },
                ),
                context=memory_context,
            )

            # Update collaboration status
            collaboration_context["status"] = "validation_completed"
            collaboration_context["memory_id"] = memory_id
            collaboration_context["message"] = collaboration_message

            result = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.PROMOTION_TO_INVENTORY.value,
                "status": "completed",
                "campaigns_validated": len(campaign_requests),
                "feasible_campaigns": collaboration_message["feasible_campaigns"],
                "total_stock_required": int(total_stock_required),
                "validation_results": validation_results,
                "collaboration_message": collaboration_message,
                "analysis": f"Validated stock availability for {len(campaign_requests)} campaigns, {collaboration_message['feasible_campaigns']} are feasible",
            }

            logger.info(
                "collaboration_id=<%s>, feasible=<%d> | promotion-to-inventory validation completed",
                collaboration_id,
                collaboration_message["feasible_campaigns"],
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "collaboration_id": collaboration_id,
                    "campaigns_validated": len(campaign_requests),
                    "feasible_campaigns": collaboration_message["feasible_campaigns"],
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error("error=<%s> | failed to validate stock availability", str(e))

            # End Langfuse span with error outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "success": False,
                    "error": str(e),
                },
                error=e,
            )

            return {
                "collaboration_id": str(uuid4()),
                "status": "failed",
                "analysis": f"Error validating stock: {str(e)}",
            }

    async def cross_agent_learning_from_outcomes(
        self,
        decision_outcomes: List[Dict[str, Any]],
        learning_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create cross-agent learning from shared decision outcomes.

        Args:
            decision_outcomes: List of decision outcomes from multiple agents
            learning_context: Additional context for learning analysis

        Returns:
            Dictionary containing cross-agent learning insights
        """
        # Start Langfuse span for collaboration
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.workflow_id,
            operation="cross_agent_learning_from_outcomes",
            input_data={
                "decision_outcomes_count": len(decision_outcomes),
                "learning_context": learning_context,
            },
        )

        try:
            learning_id = str(uuid4())
            logger.info(
                "learning_id=<%s>, outcomes=<%d> | initiating cross-agent learning",
                learning_id,
                len(decision_outcomes),
            )

            # Analyze outcomes by agent and decision type
            agent_performance = {}
            collaboration_patterns = {}
            shared_insights = []

            for outcome in decision_outcomes:
                agent_id = outcome.get("agent_id", "unknown")
                decision_type = outcome.get("decision_type", "unknown")
                success = outcome.get("success", False)
                collaboration_id = outcome.get("collaboration_id")

                # Track agent performance
                if agent_id not in agent_performance:
                    agent_performance[agent_id] = {
                        "total_decisions": 0,
                        "successful_decisions": 0,
                        "decision_types": {},
                        "collaboration_success": 0,
                        "solo_success": 0,
                    }

                agent_performance[agent_id]["total_decisions"] += 1
                if success:
                    agent_performance[agent_id]["successful_decisions"] += 1

                # Track decision types
                if decision_type not in agent_performance[agent_id]["decision_types"]:
                    agent_performance[agent_id]["decision_types"][decision_type] = {
                        "total": 0,
                        "successful": 0,
                    }

                agent_performance[agent_id]["decision_types"][decision_type][
                    "total"
                ] += 1
                if success:
                    agent_performance[agent_id]["decision_types"][decision_type][
                        "successful"
                    ] += 1

                # Track collaboration vs solo performance
                if collaboration_id:
                    if success:
                        agent_performance[agent_id]["collaboration_success"] += 1
                else:
                    if success:
                        agent_performance[agent_id]["solo_success"] += 1

                # Track collaboration patterns
                if collaboration_id:
                    if collaboration_id not in collaboration_patterns:
                        collaboration_patterns[collaboration_id] = {
                            "agents_involved": set(),
                            "decision_types": set(),
                            "success_rate": 0,
                            "outcomes": [],
                        }

                    collaboration_patterns[collaboration_id]["agents_involved"].add(
                        agent_id
                    )
                    collaboration_patterns[collaboration_id]["decision_types"].add(
                        decision_type
                    )
                    collaboration_patterns[collaboration_id]["outcomes"].append(success)

            # Calculate collaboration success rates
            for collab_id, pattern in collaboration_patterns.items():
                pattern["success_rate"] = (
                    sum(pattern["outcomes"]) / len(pattern["outcomes"])
                    if pattern["outcomes"]
                    else 0
                )
                pattern["agents_involved"] = list(pattern["agents_involved"])
                pattern["decision_types"] = list(pattern["decision_types"])

            # Generate shared insights
            # Insight 1: Best performing collaboration patterns
            best_collaborations = sorted(
                [(k, v) for k, v in collaboration_patterns.items()],
                key=lambda x: x[1]["success_rate"],
                reverse=True,
            )[:3]

            if best_collaborations:
                shared_insights.append(
                    {
                        "insight_type": "best_collaboration_patterns",
                        "description": "Most successful collaboration patterns identified",
                        "details": [
                            {
                                "collaboration_type": f"{'-'.join(sorted(collab[1]['agents_involved']))}",
                                "success_rate": round(collab[1]["success_rate"], 2),
                                "decision_types": collab[1]["decision_types"],
                            }
                            for collab in best_collaborations
                        ],
                    }
                )

            # Insight 2: Agent specialization patterns
            agent_specializations = {}
            for agent_id, performance in agent_performance.items():
                best_decision_type = (
                    max(
                        performance["decision_types"].items(),
                        key=lambda x: x[1]["successful"] / max(1, x[1]["total"]),
                    )
                    if performance["decision_types"]
                    else ("unknown", {"successful": 0, "total": 1})
                )

                agent_specializations[agent_id] = {
                    "best_decision_type": best_decision_type[0],
                    "success_rate": best_decision_type[1]["successful"]
                    / max(1, best_decision_type[1]["total"]),
                    "collaboration_advantage": (
                        performance["collaboration_success"]
                        / max(1, performance["total_decisions"])
                        - performance["solo_success"]
                        / max(1, performance["total_decisions"])
                    ),
                }

            shared_insights.append(
                {
                    "insight_type": "agent_specializations",
                    "description": "Agent specialization and collaboration advantages identified",
                    "details": agent_specializations,
                }
            )

            # Insight 3: Decision timing patterns
            timing_insights = self._analyze_decision_timing_patterns(decision_outcomes)
            if timing_insights:
                shared_insights.append(
                    {
                        "insight_type": "timing_patterns",
                        "description": "Optimal decision timing patterns discovered",
                        "details": timing_insights,
                    }
                )

            # Generate learning recommendations
            learning_recommendations = []

            # Recommendation 1: Enhance successful collaboration patterns
            if best_collaborations:
                learning_recommendations.append(
                    {
                        "recommendation": "enhance_successful_collaborations",
                        "description": f"Focus on {best_collaborations[0][1]['agents_involved']} collaboration pattern",
                        "expected_impact": "15-25% improvement in decision success rate",
                    }
                )

            # Recommendation 2: Agent specialization optimization
            high_performing_agents = [
                agent_id
                for agent_id, perf in agent_performance.items()
                if perf["successful_decisions"] / max(1, perf["total_decisions"]) > 0.8
            ]

            if high_performing_agents:
                learning_recommendations.append(
                    {
                        "recommendation": "leverage_agent_specializations",
                        "description": f"Route decisions to specialized agents: {', '.join(high_performing_agents)}",
                        "expected_impact": "10-20% improvement in overall system performance",
                    }
                )

            # Store learning insights in memory for all agents
            for agent_id in agent_performance.keys():
                learning_memory_context = {
                    "learning_type": "cross_agent_insights",
                    "insights_count": len(shared_insights),
                    "agent_performance": agent_performance.get(agent_id, {}),
                    "collaboration_patterns": len(collaboration_patterns),
                }

                agent_memory.store_decision(
                    agent_id=agent_id,
                    decision=AgentDecision(
                        agent_id=agent_id,
                        action_type=ActionType.LEARNING_UPDATE,
                        parameters={
                            "learning_id": learning_id,
                            "insights_received": len(shared_insights),
                        },
                        rationale=f"Received cross-agent learning insights from {len(decision_outcomes)} decision outcomes",
                        confidence_score=0.8,
                        expected_outcome={
                            "decision_improvement": "measurable",
                            "collaboration_optimization": "enhanced",
                        },
                    ),
                    context=learning_memory_context,
                )

            result = {
                "learning_id": learning_id,
                "type": CollaborationType.CROSS_AGENT_LEARNING.value,
                "status": "completed",
                "outcomes_analyzed": len(decision_outcomes),
                "agents_involved": list(agent_performance.keys()),
                "collaboration_patterns_identified": len(collaboration_patterns),
                "shared_insights": shared_insights,
                "learning_recommendations": learning_recommendations,
                "agent_performance_summary": {
                    agent_id: {
                        "success_rate": round(
                            perf["successful_decisions"]
                            / max(1, perf["total_decisions"]),
                            2,
                        ),
                        "total_decisions": perf["total_decisions"],
                        "collaboration_advantage": round(
                            agent_specializations.get(agent_id, {}).get(
                                "collaboration_advantage", 0
                            ),
                            2,
                        ),
                    }
                    for agent_id, perf in agent_performance.items()
                },
                "analysis": f"Generated {len(shared_insights)} cross-agent insights from {len(decision_outcomes)} decision outcomes",
            }

            logger.info(
                "learning_id=<%s>, insights=<%d>, agents=<%d> | cross-agent learning completed",
                learning_id,
                len(shared_insights),
                len(agent_performance),
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "learning_id": learning_id,
                    "outcomes_analyzed": len(decision_outcomes),
                    "insights_generated": len(shared_insights),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error("error=<%s> | failed to perform cross-agent learning", str(e))

            # End Langfuse span with error outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "success": False,
                    "error": str(e),
                },
                error=e,
            )

            return {
                "learning_id": str(uuid4()),
                "status": "failed",
                "analysis": f"Error in cross-agent learning: {str(e)}",
            }

    async def collaborative_market_event_response(
        self,
        market_event: Dict[str, Any],
        participating_agents: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Implement collaborative response to market events and demand spikes.

        Args:
            market_event: Market event data requiring coordinated response
            participating_agents: List of agent IDs to coordinate (default: all agents)

        Returns:
            Dictionary containing collaborative response results
        """
        # Start Langfuse span for collaboration
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.workflow_id,
            operation="collaborative_market_event_response",
            input_data={
                "market_event": market_event,
                "participating_agents": participating_agents,
            },
        )

        try:
            collaboration_id = str(uuid4())
            event_type = market_event.get("event_type", "unknown")
            event_severity = market_event.get("severity", "medium")

            logger.info(
                "collaboration_id=<%s>, event_type=<%s>, severity=<%s> | initiating collaborative market event response",
                collaboration_id,
                event_type,
                event_severity,
            )

            if not participating_agents:
                participating_agents = [
                    "inventory_agent",
                    "pricing_agent",
                    "promotion_agent",
                ]

            # Create collaboration context
            collaboration_context = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.MARKET_EVENT_RESPONSE.value,
                "market_event": market_event,
                "participating_agents": participating_agents,
                "timestamp": datetime.now(timezone.utc),
                "event_severity": event_severity,
            }

            # Store collaboration in active workflows
            self.active_collaborations[collaboration_id] = collaboration_context

            # Generate coordinated response strategy based on event type
            response_strategy = self._generate_market_event_strategy(
                market_event, participating_agents
            )

            # Create agent-specific action plans
            agent_action_plans = {}

            for agent_id in participating_agents:
                if agent_id == "inventory_agent":
                    action_plan = self._generate_inventory_event_actions(
                        market_event, response_strategy
                    )
                elif agent_id == "pricing_agent":
                    action_plan = self._generate_pricing_event_actions(
                        market_event, response_strategy
                    )
                elif agent_id == "promotion_agent":
                    action_plan = self._generate_promotion_event_actions(
                        market_event, response_strategy
                    )
                else:
                    action_plan = {"actions": [], "priority": "low"}

                agent_action_plans[agent_id] = action_plan

            # Calculate expected collaborative impact
            expected_impact = self._calculate_collaborative_impact(
                market_event, agent_action_plans
            )

            # Create coordination timeline
            coordination_timeline = self._create_coordination_timeline(
                agent_action_plans, event_severity
            )

            # Generate collaboration message
            collaboration_message = {
                "message_type": "market_event_coordination",
                "collaboration_id": collaboration_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_event": market_event,
                "response_strategy": response_strategy,
                "participating_agents": participating_agents,
                "agent_action_plans": agent_action_plans,
                "coordination_timeline": coordination_timeline,
                "expected_impact": expected_impact,
                "coordination_request": {
                    "action_required": "execute_coordinated_response",
                    "timeline": f"immediate"
                    if event_severity == "high"
                    else "within_2_hours",
                    "success_metrics": [
                        "response_time",
                        "coordination_effectiveness",
                        "market_impact_mitigation",
                    ],
                },
            }

            # Store collaboration in memory for each participating agent
            for agent_id in participating_agents:
                memory_context = {
                    "collaboration_type": CollaborationType.MARKET_EVENT_RESPONSE.value,
                    "event_type": event_type,
                    "event_severity": event_severity,
                    "participating_agents_count": len(participating_agents),
                    "agent_actions_count": len(
                        agent_action_plans[agent_id].get("actions", [])
                    ),
                }

                agent_memory.store_decision(
                    agent_id=agent_id,
                    decision=AgentDecision(
                        agent_id=agent_id,
                        action_type=ActionType.COLLABORATION_REQUEST,
                        parameters={
                            "collaboration_id": collaboration_id,
                            "event_type": event_type,
                            "coordinated_agents": len(participating_agents),
                        },
                        rationale=f"Participating in collaborative response to {event_type} market event",
                        confidence_score=0.85,
                        expected_outcome={
                            "coordinated_response": "effective",
                            "market_impact_mitigation": expected_impact.get(
                                "mitigation_score", 0.7
                            ),
                        },
                    ),
                    context=memory_context,
                )

            # Update collaboration status
            collaboration_context["status"] = "coordination_initiated"
            collaboration_context["message"] = collaboration_message

            result = {
                "collaboration_id": collaboration_id,
                "type": CollaborationType.MARKET_EVENT_RESPONSE.value,
                "status": "initiated",
                "event_type": event_type,
                "event_severity": event_severity,
                "participating_agents": participating_agents,
                "response_strategy": response_strategy,
                "agent_action_plans": agent_action_plans,
                "coordination_timeline": coordination_timeline,
                "expected_impact": expected_impact,
                "collaboration_message": collaboration_message,
                "analysis": f"Initiated collaborative response to {event_type} event with {len(participating_agents)} agents",
            }

            logger.info(
                "collaboration_id=<%s>, agents=<%d>, actions=<%d> | collaborative market event response initiated",
                collaboration_id,
                len(participating_agents),
                sum(
                    len(plan.get("actions", [])) for plan in agent_action_plans.values()
                ),
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "collaboration_id": collaboration_id,
                    "event_type": event_type,
                    "participating_agents_count": len(participating_agents),
                    "total_actions": sum(
                        len(plan.get("actions", []))
                        for plan in agent_action_plans.values()
                    ),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "error=<%s> | failed to initiate collaborative market event response",
                str(e),
            )

            # End Langfuse span with error outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "success": False,
                    "error": str(e),
                },
                error=e,
            )

            return {
                "collaboration_id": str(uuid4()),
                "status": "failed",
                "analysis": f"Error initiating market event response: {str(e)}",
            }

    # Helper methods for collaboration workflows

    def _calculate_urgency_level(self, slow_moving_items: List[Dict[str, Any]]) -> str:
        """Calculate urgency level based on slow-moving items characteristics."""
        if not slow_moving_items:
            return "low"

        avg_days_without_sale = sum(
            item.get("days_without_sale", 0) for item in slow_moving_items
        ) / len(slow_moving_items)
        total_value = sum(item.get("inventory_value", 0) for item in slow_moving_items)

        if avg_days_without_sale > 45 or total_value > 10000:
            return "critical"
        elif avg_days_without_sale > 30 or total_value > 5000:
            return "high"
        elif avg_days_without_sale > 14 or total_value > 2000:
            return "medium"
        else:
            return "low"

    def _get_optimal_channels(self, campaign_type: str) -> List[str]:
        """Get optimal promotional channels based on campaign type."""
        channel_mapping = {
            "clearance_sale": ["email", "website_banner", "social_media"],
            "price_match_promotion": ["website", "mobile_app", "customer_service"],
            "flash_sale": ["push_notifications", "social_media", "sms"],
            "general_promotion": ["email", "website", "social_media"],
        }
        return channel_mapping.get(campaign_type, ["email", "website"])

    def _get_messaging_strategy(
        self, campaign_type: str, discount_percentage: float
    ) -> Dict[str, str]:
        """Get messaging strategy based on campaign type and discount."""
        if campaign_type == "clearance_sale":
            return {
                "primary_message": f"Final Clearance - {discount_percentage:.0f}% Off!",
                "urgency": "Limited quantities available",
                "call_to_action": "Shop now before it's gone",
            }
        elif campaign_type == "flash_sale":
            return {
                "primary_message": f"Flash Sale - {discount_percentage:.0f}% Off!",
                "urgency": "24 hours only",
                "call_to_action": "Don't miss out - shop now",
            }
        else:
            return {
                "primary_message": f"Special Offer - {discount_percentage:.0f}% Off",
                "urgency": "Limited time offer",
                "call_to_action": "Shop the sale",
            }

    def _generate_campaign_stock_recommendations(
        self, campaign_feasible: bool, stock_requirements: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate stock recommendations for campaign feasibility."""
        recommendations = []

        if not campaign_feasible:
            recommendations.append("Campaign requires immediate inventory action")
            recommendations.append("Consider emergency restocking for shortage items")
            recommendations.append(
                "Alternative: Reduce campaign scope to available inventory"
            )

        high_risk_items = [
            req for req in stock_requirements if req["risk_level"] == "high"
        ]
        if high_risk_items:
            recommendations.append(
                f"Monitor {len(high_risk_items)} high-risk items closely"
            )

        if not recommendations:
            recommendations.append("Stock levels adequate for planned campaign")

        return recommendations

    def _analyze_decision_timing_patterns(
        self, decision_outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze timing patterns in decision outcomes."""
        # Simplified timing analysis
        timing_data = {}
        for outcome in decision_outcomes:
            timestamp = outcome.get("timestamp")
            success = outcome.get("success", False)

            if timestamp:
                # Extract hour of day for pattern analysis
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    hour = dt.hour

                    if hour not in timing_data:
                        timing_data[hour] = {"total": 0, "successful": 0}

                    timing_data[hour]["total"] += 1
                    if success:
                        timing_data[hour]["successful"] += 1
                except:
                    continue

        if not timing_data:
            return {}

        # Find best performing hours
        best_hours = sorted(
            timing_data.items(),
            key=lambda x: x[1]["successful"] / max(1, x[1]["total"]),
            reverse=True,
        )[:3]

        return {
            "optimal_decision_hours": [hour for hour, _ in best_hours],
            "peak_success_rate": best_hours[0][1]["successful"]
            / max(1, best_hours[0][1]["total"])
            if best_hours
            else 0,
        }

    def _generate_market_event_strategy(
        self, market_event: Dict[str, Any], participating_agents: List[str]
    ) -> Dict[str, Any]:
        """Generate coordinated response strategy for market events."""
        event_type = market_event.get("event_type", "unknown")
        event_severity = market_event.get("severity", "medium")

        if event_type == "demand_spike":
            return {
                "strategy_type": "demand_surge_response",
                "primary_objective": "maximize_revenue_capture",
                "coordination_priority": "high",
                "response_timeline": "immediate",
                "key_actions": [
                    "increase_prices",
                    "validate_inventory",
                    "create_urgency_campaigns",
                ],
            }
        elif event_type == "competitor_price_drop":
            return {
                "strategy_type": "competitive_response",
                "primary_objective": "maintain_market_position",
                "coordination_priority": "medium",
                "response_timeline": "within_4_hours",
                "key_actions": [
                    "analyze_competitor_move",
                    "adjust_pricing",
                    "enhance_value_proposition",
                ],
            }
        elif event_type == "supply_disruption":
            return {
                "strategy_type": "supply_mitigation",
                "primary_objective": "minimize_stockouts",
                "coordination_priority": "critical",
                "response_timeline": "immediate",
                "key_actions": [
                    "inventory_reallocation",
                    "demand_management",
                    "customer_communication",
                ],
            }
        else:
            return {
                "strategy_type": "general_response",
                "primary_objective": "maintain_stability",
                "coordination_priority": "medium",
                "response_timeline": "within_2_hours",
                "key_actions": ["monitor_situation", "prepare_contingencies"],
            }

    def _generate_inventory_event_actions(
        self, market_event: Dict[str, Any], strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate inventory agent actions for market events."""
        event_type = market_event.get("event_type", "unknown")

        if event_type == "demand_spike":
            return {
                "actions": [
                    "validate_current_stock_levels",
                    "calculate_surge_demand_forecast",
                    "identify_potential_stockouts",
                    "recommend_emergency_restocking",
                ],
                "priority": "critical",
                "timeline": "immediate",
            }
        elif event_type == "supply_disruption":
            return {
                "actions": [
                    "assess_affected_inventory",
                    "reallocate_available_stock",
                    "identify_alternative_suppliers",
                    "adjust_safety_buffers",
                ],
                "priority": "critical",
                "timeline": "immediate",
            }
        else:
            return {
                "actions": ["monitor_inventory_levels", "update_demand_forecasts"],
                "priority": "medium",
                "timeline": "within_2_hours",
            }

    def _generate_pricing_event_actions(
        self, market_event: Dict[str, Any], strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate pricing agent actions for market events."""
        event_type = market_event.get("event_type", "unknown")

        if event_type == "demand_spike":
            return {
                "actions": [
                    "analyze_demand_elasticity",
                    "calculate_optimal_surge_pricing",
                    "implement_dynamic_price_adjustments",
                    "monitor_competitor_responses",
                ],
                "priority": "high",
                "timeline": "within_1_hour",
            }
        elif event_type == "competitor_price_drop":
            return {
                "actions": [
                    "analyze_competitor_pricing_strategy",
                    "evaluate_price_match_necessity",
                    "calculate_margin_impact",
                    "implement_strategic_price_response",
                ],
                "priority": "high",
                "timeline": "within_2_hours",
            }
        else:
            return {
                "actions": ["monitor_pricing_environment", "maintain_current_strategy"],
                "priority": "low",
                "timeline": "within_4_hours",
            }

    def _generate_promotion_event_actions(
        self, market_event: Dict[str, Any], strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate promotion agent actions for market events."""
        event_type = market_event.get("event_type", "unknown")

        if event_type == "demand_spike":
            return {
                "actions": [
                    "create_urgency_campaigns",
                    "amplify_social_media_presence",
                    "implement_scarcity_messaging",
                    "coordinate_with_pricing_changes",
                ],
                "priority": "high",
                "timeline": "within_1_hour",
            }
        elif event_type == "competitor_price_drop":
            return {
                "actions": [
                    "develop_value_proposition_campaigns",
                    "highlight_product_differentiators",
                    "create_loyalty_retention_offers",
                    "monitor_social_sentiment",
                ],
                "priority": "medium",
                "timeline": "within_3_hours",
            }
        else:
            return {
                "actions": [
                    "monitor_market_sentiment",
                    "prepare_contingency_campaigns",
                ],
                "priority": "low",
                "timeline": "within_4_hours",
            }

    def _calculate_collaborative_impact(
        self,
        market_event: Dict[str, Any],
        agent_action_plans: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate expected impact of collaborative response."""
        total_actions = sum(
            len(plan.get("actions", [])) for plan in agent_action_plans.values()
        )
        high_priority_agents = sum(
            1
            for plan in agent_action_plans.values()
            if plan.get("priority") == "critical"
        )

        # Simplified impact calculation
        base_impact = 0.6
        coordination_bonus = min(0.3, len(agent_action_plans) * 0.1)
        urgency_bonus = high_priority_agents * 0.1

        mitigation_score = min(1.0, base_impact + coordination_bonus + urgency_bonus)

        return {
            "mitigation_score": round(mitigation_score, 2),
            "total_coordinated_actions": total_actions,
            "high_priority_responses": high_priority_agents,
            "expected_response_effectiveness": "high"
            if mitigation_score > 0.8
            else "medium"
            if mitigation_score > 0.6
            else "low",
        }

    def _create_coordination_timeline(
        self, agent_action_plans: Dict[str, Dict[str, Any]], event_severity: str
    ) -> List[Dict[str, Any]]:
        """Create coordination timeline for agent actions."""
        timeline = []

        # Sort actions by priority and timeline
        all_actions = []
        for agent_id, plan in agent_action_plans.items():
            for action in plan.get("actions", []):
                all_actions.append(
                    {
                        "agent_id": agent_id,
                        "action": action,
                        "priority": plan.get("priority", "medium"),
                        "timeline": plan.get("timeline", "within_2_hours"),
                    }
                )

        # Group by timeline
        timeline_groups = {}
        for action in all_actions:
            timeline_key = action["timeline"]
            if timeline_key not in timeline_groups:
                timeline_groups[timeline_key] = []
            timeline_groups[timeline_key].append(action)

        # Create ordered timeline
        timeline_order = [
            "immediate",
            "within_1_hour",
            "within_2_hours",
            "within_3_hours",
            "within_4_hours",
        ]

        for timeline_key in timeline_order:
            if timeline_key in timeline_groups:
                timeline.append(
                    {
                        "phase": timeline_key,
                        "actions": timeline_groups[timeline_key],
                        "coordination_required": len(
                            set(
                                action["agent_id"]
                                for action in timeline_groups[timeline_key]
                            )
                        )
                        > 1,
                    }
                )

        return timeline


# Global collaboration workflow instance
collaboration_workflow = CollaborationWorkflow()
