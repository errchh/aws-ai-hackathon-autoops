"""
Pricing Agent implementation using AWS Strands framework.

This module implements the Pricing Agent that handles dynamic pricing decisions,
markdown applications, and price optimization using demand elasticity analysis.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from strands import Agent, tool
from strands.models import BedrockModel
from langfuse import observe

from agents.memory import agent_memory
from config.settings import get_settings
from config.langfuse_integration import get_langfuse_integration
from models.core import AgentDecision, ActionType, Product


# Configure logging
logger = logging.getLogger(__name__)


class PricingAgent:
    """
    Pricing Agent for dynamic pricing decisions and markdown applications.

    This agent uses AWS Strands framework with Anthropic Claude to make
    intelligent pricing decisions based on demand elasticity, competitor
    analysis, and inventory levels.
    """

    def __init__(self):
        """Initialize the Pricing Agent with Strands framework."""
        self.settings = get_settings()
        self.agent_id = "pricing_agent"

        # Initialize Bedrock model for Anthropic Claude
        self.model = BedrockModel(
            model_id=self.settings.bedrock.model_id,
            temperature=self.settings.bedrock.temperature,
            max_tokens=self.settings.bedrock.max_tokens,
            streaming=False,
        )

        # Initialize Strands Agent with custom tools
        self.agent = Agent(
            model=self.model,
            tools=[
                self.analyze_demand_elasticity,
                self.calculate_optimal_price,
                self.apply_markdown_strategy,
                self.evaluate_price_impact,
                self.get_competitor_prices,
                self.retrieve_pricing_history,
            ],
        )

        # Initialize Langfuse integration service
        self.langfuse_integration = get_langfuse_integration()

        logger.info(
            "agent_id=<%s> | Pricing Agent initialized with Strands framework",
            self.agent_id,
        )

    @observe(name="pricing_agent_analyze_demand_elasticity")
    @tool
    def analyze_demand_elasticity(
        self, product_id: str, price_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze demand elasticity for a product based on historical price and sales data.

        Args:
            product_id: Product identifier
            price_history: List of historical price and sales data points

        Returns:
            Dictionary containing elasticity analysis results
        """
        # Start Langfuse span for elasticity analysis
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="analyze_demand_elasticity",
            input_data={
                "product_id": product_id,
                "price_history_count": len(price_history),
            },
         )
 
        try:
            logger.debug("product_id=<%s> | analyzing demand elasticity", product_id)

            if len(price_history) < 2:
                return {
                    "elasticity_coefficient": -1.0,  # Default elastic assumption
                    "confidence": 0.3,
                    "recommendation": "insufficient_data",
                    "analysis": "Need more historical data for accurate elasticity calculation",
                }

            # Calculate price elasticity using percentage changes
            total_elasticity = 0
            valid_calculations = 0

            for i in range(1, len(price_history)):
                prev_data = price_history[i - 1]
                curr_data = price_history[i]

                price_change_pct = (
                    curr_data["price"] - prev_data["price"]
                ) / prev_data["price"]
                demand_change_pct = (
                    curr_data["quantity"] - prev_data["quantity"]
                ) / prev_data["quantity"]

                if price_change_pct != 0:
                    elasticity = demand_change_pct / price_change_pct
                    total_elasticity += elasticity
                    valid_calculations += 1

            if valid_calculations == 0:
                avg_elasticity = -1.0
                confidence = 0.3
            else:
                avg_elasticity = total_elasticity / valid_calculations
                confidence = min(0.9, 0.5 + (valid_calculations * 0.1))

            # Determine product elasticity category
            if avg_elasticity > -0.5:
                category = "inelastic"
                recommendation = "price_increase_opportunity"
            elif avg_elasticity < -1.5:
                category = "highly_elastic"
                recommendation = "price_decrease_for_volume"
            else:
                category = "moderately_elastic"
                recommendation = "balanced_pricing"

            result = {
                "elasticity_coefficient": round(avg_elasticity, 3),
                "confidence": round(confidence, 2),
                "category": category,
                "recommendation": recommendation,
                "data_points": len(price_history),
                "analysis": f"Product shows {category} demand with elasticity of {avg_elasticity:.2f}",
            }

            logger.info(
                "product_id=<%s>, elasticity=<%f>, category=<%s> | demand elasticity analyzed",
                product_id,
                avg_elasticity,
                category,
            )

            # End Langfuse span and log decision
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "elasticity_coefficient": round(avg_elasticity, 3),
                    "category": category,
                    "recommendation": recommendation,
                    "confidence": round(confidence, 2),
                },
            )

            # Log pricing decision
            self.langfuse_integration.log_agent_decision(
                agent_id=self.agent_id,
                decision_data={
                    "type": "elasticity_analysis",
                    "inputs": {
                        "product_id": product_id,
                        "price_history_count": len(price_history),
                    },
                    "outputs": {
                        "elasticity_coefficient": round(avg_elasticity, 3),
                        "category": category,
                        "recommendation": recommendation,
                    },
                    "confidence": round(confidence, 2),
                    "reasoning": f"Product shows {category} demand with elasticity of {avg_elasticity:.2f}",
                },
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to analyze demand elasticity",
                product_id,
                str(e),
            )

            # End Langfuse span with error
            self.langfuse_integration.end_agent_span(span_id=span_id, error=e)

            return {
                "elasticity_coefficient": -1.0,
                "confidence": 0.1,
                "recommendation": "analysis_failed",
                "analysis": f"Error in elasticity analysis: {str(e)}",
            }

    @observe(name="pricing_agent_calculate_optimal_price")
    @tool
    def calculate_optimal_price(
        self, product_data: Dict[str, Any], market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate optimal pricing for profit maximization.

        Args:
            product_data: Product information including cost, current price, inventory
            market_conditions: Market data including competitor prices, demand trends

        Returns:
            Dictionary containing optimal pricing recommendation
        """
        # Start Langfuse span for optimal price calculation
        span_id = self.langfuse_integration.start_agent_span(
             agent_id=self.agent_id,
             operation="calculate_optimal_price",
             input_data={
                 "product_id": product_data.get("id", "unknown"),
                 "current_price": product_data.get("current_price", 0),
                 "cost": product_data.get("cost", 0)
             }
         )

        try:
            logger.debug(
                "product_id=<%s> | calculating optimal price",
                product_data.get("id", "unknown"),
            )

            current_price = product_data["current_price"]
            cost = product_data["cost"]
            inventory_level = product_data.get("inventory_level", 0)

            # Get elasticity from market conditions or use default
            elasticity = market_conditions.get("elasticity_coefficient", -1.2)
            competitor_avg = market_conditions.get(
                "competitor_average_price", current_price
            )

            # Calculate profit-maximizing price using elasticity
            # Optimal price = cost / (1 + 1/elasticity) for elastic demand
            if elasticity < -1:  # Elastic demand
                optimal_base = cost / (1 + 1 / elasticity)
            else:  # Inelastic demand
                optimal_base = cost * 1.5  # Conservative markup for inelastic products

            # Adjust for market positioning
            market_adjustment = 1.0
            if competitor_avg > 0:
                if current_price > competitor_avg * 1.1:  # We're expensive
                    market_adjustment = 0.95  # Slight decrease
                elif current_price < competitor_avg * 0.9:  # We're cheap
                    market_adjustment = 1.05  # Slight increase

            # Adjust for inventory levels
            inventory_adjustment = 1.0
            if (
                inventory_level > product_data.get("reorder_point", 50) * 2
            ):  # High inventory
                inventory_adjustment = 0.92  # Reduce price to move inventory
            elif inventory_level < product_data.get(
                "reorder_point", 50
            ):  # Low inventory
                inventory_adjustment = 1.08  # Increase price due to scarcity

            # Calculate final optimal price
            optimal_price = optimal_base * market_adjustment * inventory_adjustment

            # Ensure price is above cost with minimum margin
            min_price = cost * 1.1  # 10% minimum margin
            optimal_price = max(optimal_price, min_price)

            # Calculate expected impact
            price_change_pct = (optimal_price - current_price) / current_price
            expected_demand_change = elasticity * price_change_pct

            # Estimate profit impact
            current_margin = current_price - cost
            new_margin = optimal_price - cost
            base_volume = 100  # Assumed base volume
            new_volume = base_volume * (1 + expected_demand_change)

            current_profit = current_margin * base_volume
            new_profit = new_margin * new_volume
            profit_impact = new_profit - current_profit

            result = {
                "optimal_price": round(optimal_price, 2),
                "current_price": current_price,
                "price_change_percentage": round(price_change_pct * 100, 1),
                "expected_demand_change": round(expected_demand_change * 100, 1),
                "profit_impact_estimate": round(profit_impact, 2),
                "confidence_score": 0.8,
                "rationale": f"Optimal price calculated using elasticity {elasticity:.2f}, market positioning, and inventory levels",
            }

            logger.info(
                "product_id=<%s>, optimal_price=<%f>, profit_impact=<%f> | optimal price calculated",
                product_data.get("id", "unknown"),
                optimal_price,
                profit_impact,
            )

            # End Langfuse span and log decision
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "optimal_price": round(optimal_price, 2),
                    "price_change_percentage": round(price_change_pct * 100, 1),
                    "profit_impact_estimate": round(profit_impact, 2),
                    "confidence_score": 0.8
                }
            )

            # Log pricing decision
            self.langfuse_integration.log_agent_decision(
                agent_id=self.agent_id,
                decision_data={
                    "type": "optimal_price_calculation",
                    "inputs": {
                        "product_id": product_data.get("id", "unknown"),
                        "current_price": current_price,
                        "cost": cost,
                        "elasticity": elasticity
                    },
                    "outputs": {
                        "optimal_price": round(optimal_price, 2),
                        "price_change_percentage": round(price_change_pct * 100, 1),
                        "profit_impact_estimate": round(profit_impact, 2)
                    },
                    "confidence": 0.8,
                    "reasoning": f"Optimal price calculated using elasticity {elasticity:.2f}, market positioning, and inventory levels"
                }
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to calculate optimal price",
                product_data.get("id", "unknown"),
                str(e),
            )

            # End Langfuse span with error
            self.langfuse_integration.end_agent_span(span_id=span_id, error=e)

            return {
                "optimal_price": product_data["current_price"],
                "confidence_score": 0.1,
                "rationale": f"Error in price calculation: {str(e)}",
            }

    @observe(name="pricing_agent_apply_markdown_strategy")
    @tool
    def apply_markdown_strategy(
        self, product_data: Dict[str, Any], urgency_level: str
    ) -> Dict[str, Any]:
        """
        Apply markdown strategy for slow-moving inventory.

        Args:
            product_data: Product information
            urgency_level: Urgency of markdown (low, medium, high, critical)

         Returns:
             Dictionary containing markdown strategy recommendation
         """
        # Start Langfuse span for markdown strategy
        span_id = self.langfuse_integration.start_agent_span(
             agent_id=self.agent_id,
             operation="apply_markdown_strategy",
             input_data={
                 "product_id": product_data.get("id", "unknown"),
                 "urgency_level": urgency_level,
                 "current_price": product_data.get("current_price", 0)
             }
         )

        try:
            logger.debug(
                "product_id=<%s>, urgency=<%s> | applying markdown strategy",
                product_data.get("id", "unknown"),
                urgency_level,
            )

            current_price = product_data["current_price"]
            cost = product_data["cost"]
            inventory_level = product_data.get("inventory_level", 0)
            days_without_sale = product_data.get("days_without_sale", 0)

            # Determine markdown percentage based on urgency and inventory age
            markdown_percentages = {"low": 10, "medium": 20, "high": 35, "critical": 50}

            base_markdown = markdown_percentages.get(urgency_level, 20)

            # Adjust markdown based on days without sale
            if days_without_sale > 30:
                base_markdown += 10
            elif days_without_sale > 14:
                base_markdown += 5

            # Ensure markdown doesn't go below cost
            max_markdown = (
                (current_price - cost * 1.05) / current_price
            ) * 100  # Keep 5% margin
            final_markdown = min(base_markdown, max_markdown)

            markdown_price = current_price * (1 - final_markdown / 100)

            # Calculate expected impact
            elasticity = -1.5  # Assume elastic demand for markdowns
            price_change_pct = -final_markdown / 100
            expected_demand_increase = abs(elasticity * price_change_pct) * 100

            # Estimate inventory turnover improvement
            current_turnover_days = max(1, days_without_sale)
            improved_turnover_days = max(
                1, current_turnover_days * (1 - expected_demand_increase / 200)
            )

            result = {
                "markdown_percentage": round(final_markdown, 1),
                "markdown_price": round(markdown_price, 2),
                "original_price": current_price,
                "expected_demand_increase": round(expected_demand_increase, 1),
                "estimated_turnover_days": round(improved_turnover_days, 1),
                "urgency_level": urgency_level,
                "rationale": f"Applied {final_markdown:.1f}% markdown for {urgency_level} urgency with {days_without_sale} days without sale",
            }

            logger.info(
                "product_id=<%s>, markdown_pct=<%f>, new_price=<%f> | markdown strategy applied",
                product_data.get("id", "unknown"),
                final_markdown,
                markdown_price,
            )

            # End Langfuse span and log decision
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "markdown_percentage": round(final_markdown, 1),
                    "markdown_price": round(markdown_price, 2),
                    "expected_demand_increase": round(expected_demand_increase, 1),
                    "urgency_level": urgency_level
                }
            )

            # Log pricing decision
            self.langfuse_integration.log_agent_decision(
                agent_id=self.agent_id,
                decision_data={
                    "type": "markdown_strategy",
                    "inputs": {
                        "product_id": product_data.get("id", "unknown"),
                        "urgency_level": urgency_level,
                        "days_without_sale": days_without_sale
                    },
                    "outputs": {
                        "markdown_percentage": round(final_markdown, 1),
                        "markdown_price": round(markdown_price, 2),
                        "expected_demand_increase": round(expected_demand_increase, 1)
                    },
                    "confidence": 0.85,
                    "reasoning": f"Applied {final_markdown:.1f}% markdown for {urgency_level} urgency with {days_without_sale} days without sale"
                }
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to apply markdown strategy",
                product_data.get("id", "unknown"),
                str(e),
            )

            # End Langfuse span with error
            self.langfuse_integration.end_agent_span(span_id=span_id, error=e)

            return {
                "markdown_percentage": 0,
                "markdown_price": product_data["current_price"],
                "rationale": f"Error in markdown calculation: {str(e)}",
            }

    @observe(name="pricing_agent_evaluate_price_impact")
    @tool
    def evaluate_price_impact(
        self, product_id: str, proposed_price: float, current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the potential impact of a price change before implementation.

        Args:
            product_id: Product identifier
            proposed_price: Proposed new price
            current_context: Current market and product context

         Returns:
             Dictionary containing price impact evaluation
         """
        # Start Langfuse span for price impact evaluation
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="evaluate_price_impact",
            input_data={
                "product_id": product_id,
                "proposed_price": proposed_price,
                "current_price": current_context.get("current_price", 0)
            }
        )

        try:
            logger.debug(
                "product_id=<%s>, proposed_price=<%f> | evaluating price impact",
                product_id,
                proposed_price,
            )

            current_price = current_context.get("current_price", 0)
            cost = current_context.get("cost", 0)
            elasticity = current_context.get("elasticity_coefficient", -1.2)

            if current_price == 0:
                return {
                    "impact_score": 0.0,
                    "recommendation": "insufficient_data",
                    "analysis": "Missing current price data",
                }

            # Calculate price change impact
            price_change_pct = (proposed_price - current_price) / current_price
            expected_demand_change = elasticity * price_change_pct

            # Calculate revenue and profit impacts
            base_volume = 100  # Assumed base volume
            new_volume = base_volume * (1 + expected_demand_change)

            current_revenue = current_price * base_volume
            new_revenue = proposed_price * new_volume
            revenue_impact = new_revenue - current_revenue

            current_profit = (current_price - cost) * base_volume
            new_profit = (proposed_price - cost) * new_volume
            profit_impact = new_profit - current_profit

            # Calculate impact score (0-1, higher is better)
            if profit_impact > 0:
                impact_score = min(1.0, 0.7 + (profit_impact / current_profit) * 0.3)
            else:
                impact_score = max(0.0, 0.7 + (profit_impact / current_profit) * 0.3)

            # Generate recommendation
            if impact_score > 0.8:
                recommendation = "highly_recommended"
            elif impact_score > 0.6:
                recommendation = "recommended"
            elif impact_score > 0.4:
                recommendation = "neutral"
            else:
                recommendation = "not_recommended"

            result = {
                "impact_score": round(impact_score, 2),
                "price_change_percentage": round(price_change_pct * 100, 1),
                "expected_demand_change": round(expected_demand_change * 100, 1),
                "revenue_impact": round(revenue_impact, 2),
                "profit_impact": round(profit_impact, 2),
                "recommendation": recommendation,
                "analysis": f"Price change of {price_change_pct * 100:.1f}% expected to change demand by {expected_demand_change * 100:.1f}%",
            }

            logger.info(
                "product_id=<%s>, impact_score=<%f>, recommendation=<%s> | price impact evaluated",
                product_id,
                impact_score,
                recommendation,
            )

            # End Langfuse span and log decision
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "impact_score": round(impact_score, 2),
                    "recommendation": recommendation,
                    "price_change_percentage": round(price_change_pct * 100, 1),
                    "profit_impact": round(profit_impact, 2)
                }
            )

            # Log pricing decision
            self.langfuse_integration.log_agent_decision(
                agent_id=self.agent_id,
                decision_data={
                    "type": "price_impact_evaluation",
                    "inputs": {
                        "product_id": product_id,
                        "proposed_price": proposed_price,
                        "current_price": current_price
                    },
                    "outputs": {
                        "impact_score": round(impact_score, 2),
                        "recommendation": recommendation,
                        "profit_impact": round(profit_impact, 2)
                    },
                    "confidence": 0.8,
                    "reasoning": f"Price change of {price_change_pct * 100:.1f}% expected to change demand by {expected_demand_change * 100:.1f}%"
                }
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to evaluate price impact",
                product_id,
                str(e),
            )

            # End Langfuse span with error
            self.langfuse_integration.end_agent_span(span_id=span_id, error=e)

            return {
                "impact_score": 0.0,
                "recommendation": "evaluation_failed",
                "analysis": f"Error in impact evaluation: {str(e)}",
            }

    @observe(name="pricing_agent_get_competitor_prices")
    @tool
    def get_competitor_prices(self, product_id: str, category: str) -> Dict[str, Any]:
        """
        Retrieve competitor pricing data for market analysis.

        Args:
            product_id: Product identifier
            category: Product category

        Returns:
            Dictionary containing competitor pricing analysis
        """
        # Start Langfuse span for competitor price analysis
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="get_competitor_prices",
            input_data={
                "product_id": product_id,
                "category": category
            }
        )

        try:
            logger.debug(
                "product_id=<%s>, category=<%s> | retrieving competitor prices",
                product_id,
                category,
            )

            # Simulate healthcare/wellness competitor data
            import random

            # Healthcare/wellness-specific competitor mapping
            healthcare_competitors = [
                "Johnson & Johnson",
                "Pfizer Consumer",
                "CVS Health",
                "Walgreens",
                "Nature Made",
            ]
            wellness_competitors = [
                "Garden of Life",
                "doTERRA",
                "Young Living",
                "Thorne Health",
                "NOW Foods",
            ]
            fitness_competitors = [
                "Gaiam",
                "Manduka",
                "Liforme",
                "Jade Yoga",
                "Hugger Mugger",
            ]

            # Select appropriate competitors based on category
            if (
                "vitamin" in category.lower()
                or "supplement" in category.lower()
                or "health" in category.lower()
            ):
                competitor_pool = healthcare_competitors
                base_price = 22.0  # Healthcare product base price
            elif (
                "essential_oil" in category.lower()
                or "wellness" in category.lower()
                or "organic" in category.lower()
            ):
                competitor_pool = wellness_competitors
                base_price = 28.0  # Premium wellness base price
            elif (
                "fitness" in category.lower()
                or "yoga" in category.lower()
                or "exercise" in category.lower()
            ):
                competitor_pool = fitness_competitors
                base_price = 35.0  # Fitness accessory base price
            else:
                competitor_pool = healthcare_competitors + wellness_competitors
                base_price = 25.0  # General healthcare/wellness

            competitors = []
            selected_competitors = random.sample(
                competitor_pool, min(4, len(competitor_pool))
            )

            for competitor_name in selected_competitors:
                # Healthcare/wellness pricing tends to be more stable, smaller variations
                competitor_price = base_price * (
                    0.85 + random.random() * 0.3
                )  # ±15% variation
                competitors.append(
                    {
                        "name": competitor_name,
                        "price": round(competitor_price, 2),
                        "availability": random.choice(
                            [True, True, True, False]
                        ),  # 75% available
                        "last_updated": datetime.now(timezone.utc).isoformat(),
                        "category_focus": "healthcare"
                        if competitor_name in healthcare_competitors
                        else "wellness",
                    }
                )

            # Calculate market statistics
            available_prices = [c["price"] for c in competitors if c["availability"]]
            if available_prices:
                avg_price = sum(available_prices) / len(available_prices)
                min_price = min(available_prices)
                max_price = max(available_prices)
            else:
                avg_price = min_price = max_price = base_price

            result = {
                "product_id": product_id,
                "category": category,
                "competitors": competitors,
                "market_stats": {
                    "average_price": round(avg_price, 2),
                    "min_price": round(min_price, 2),
                    "max_price": round(max_price, 2),
                    "available_competitors": len(available_prices),
                },
                "analysis": f"Found {len(competitors)} competitors with average price of ${avg_price:.2f}",
            }

            logger.info(
                "product_id=<%s>, competitors=<%d>, avg_price=<%f> | competitor prices retrieved",
                product_id,
                len(competitors),
                avg_price,
            )

            # End Langfuse span and log decision
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "competitor_count": len(competitors),
                    "average_price": round(avg_price, 2),
                    "category": category
                }
            )

            # Log pricing decision
            self.langfuse_integration.log_agent_decision(
                agent_id=self.agent_id,
                decision_data={
                    "type": "competitor_analysis",
                    "inputs": {
                        "product_id": product_id,
                        "category": category
                    },
                    "outputs": {
                        "competitor_count": len(competitors),
                        "average_price": round(avg_price, 2),
                        "min_price": round(min_price, 2),
                        "max_price": round(max_price, 2)
                    },
                    "confidence": 0.8,
                    "reasoning": f"Found {len(competitors)} competitors with average price of ${avg_price:.2f}"
                }
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to retrieve competitor prices",
                product_id,
                str(e),
            )

            # End Langfuse span with error
            self.langfuse_integration.end_agent_span(span_id=span_id, error=e)

            return {
                "product_id": product_id,
                "competitors": [],
                "analysis": f"Error retrieving competitor data: {str(e)}",
            }

    @observe(name="pricing_agent_retrieve_pricing_history")
    @tool
    def retrieve_pricing_history(
         self, product_id: str, days: int = 30
     ) -> Dict[str, Any]:
         """
         Retrieve historical pricing decisions and outcomes from agent memory.

         Args:
             product_id: Product identifier
             days: Number of days of history to retrieve

         Returns:
             Dictionary containing pricing history and insights
         """
         # Start Langfuse span for pricing history retrieval
         span_id = self.langfuse_integration.start_agent_span(
             agent_id=self.agent_id,
             operation="retrieve_pricing_history",
             input_data={
                 "product_id": product_id,
                 "days": days
             }
         )

        try:
        try:
            logger.debug(
                "product_id=<%s>, days=<%d> | retrieving pricing history",
                product_id,
                days,
            )

            # Retrieve similar pricing decisions from memory
            context = {"product_id": product_id}
            similar_decisions = agent_memory.retrieve_similar_decisions(
                agent_id=self.agent_id,
                current_context=context,
                action_type=ActionType.PRICE_ADJUSTMENT.value,
                limit=10,
            )

            # Get agent decision history
            decision_history = agent_memory.get_agent_decision_history(
                agent_id=self.agent_id,
                action_type=ActionType.PRICE_ADJUSTMENT.value,
                limit=20,
                include_outcomes=True,
            )

            # Analyze pricing patterns
            price_changes = []
            successful_changes = 0
            total_changes = 0

            for decision_data, similarity in similar_decisions:
                decision = decision_data.get("decision", {})
                outcome = decision_data.get("outcome", {})

                if decision and outcome:
                    total_changes += 1
                    if outcome.get("success", False):
                        successful_changes += 1

                    price_changes.append(
                        {
                            "timestamp": decision.get("timestamp"),
                            "price_change": decision.get("parameters", {}).get(
                                "new_price", 0
                            )
                            - decision.get("parameters", {}).get("previous_price", 0),
                            "success": outcome.get("success", False),
                            "similarity": similarity,
                        }
                    )

            success_rate = (
                (successful_changes / total_changes) if total_changes > 0 else 0
            )

            result = {
                "product_id": product_id,
                "total_decisions": len(decision_history),
                "similar_decisions": len(similar_decisions),
                "success_rate": round(success_rate, 2),
                "price_changes": price_changes[:5],  # Last 5 changes
                "insights": {
                    "most_successful_strategy": "moderate_adjustments"
                    if success_rate > 0.7
                    else "conservative_pricing",
                    "average_similarity": round(
                        sum(s for _, s in similar_decisions) / len(similar_decisions), 2
                    )
                    if similar_decisions
                    else 0,
                },
                "analysis": f"Found {len(decision_history)} historical decisions with {success_rate:.1%} success rate",
            }

            logger.info(
                "product_id=<%s>, decisions=<%d>, success_rate=<%f> | pricing history retrieved",
                product_id,
                len(decision_history),
                success_rate,
            )

            # End Langfuse span and log decision
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "total_decisions": len(decision_history),
                    "success_rate": round(success_rate, 2),
                    "similar_decisions": len(similar_decisions)
                }
            )

            # Log pricing decision
            self.langfuse_integration.log_agent_decision(
                agent_id=self.agent_id,
                decision_data={
                    "type": "pricing_history_analysis",
                    "inputs": {
                        "product_id": product_id,
                        "days": days
                    },
                    "outputs": {
                        "total_decisions": len(decision_history),
                        "success_rate": round(success_rate, 2),
                        "most_successful_strategy": "moderate_adjustments" if success_rate > 0.7 else "conservative_pricing"
                    },
                    "confidence": 0.8,
                    "reasoning": f"Found {len(decision_history)} historical decisions with {success_rate:.1%} success rate"
                }
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to retrieve pricing history",
                product_id,
                str(e),
            )

            # End Langfuse span with error
            self.langfuse_integration.end_agent_span(span_id=span_id, error=e)

            return {
                "product_id": product_id,
                "total_decisions": 0,
                "analysis": f"Error retrieving pricing history: {str(e)}",
            }

    @observe(name="pricing_agent_make_pricing_decision")
    def make_pricing_decision(
        self, product_data: Dict[str, Any], market_context: Dict[str, Any]
    ) -> AgentDecision:
        """
        Make a comprehensive pricing decision using all available tools and analysis.

        Args:
            product_data: Complete product information
            market_context: Market conditions and external factors

        Returns:
            AgentDecision object with the pricing recommendation
        """
        try:
            logger.info(
                "product_id=<%s> | making pricing decision",
                product_data.get("id", "unknown"),
            )

            # Construct healthcare/wellness-specific prompt for the agent
            product_category = product_data.get("category", "general")
            health_necessity = product_data.get("health_necessity_score", 0.5)
            expiration_date = product_data.get("expiration_date", "N/A")

            prompt = f"""
            Analyze the following healthcare/wellness product and market data to make an optimal pricing decision:
            
            Product Data:
            - ID: {product_data.get("id")}
            - Category: {product_category} (Healthcare/Wellness)
            - Current Price: ${product_data.get("current_price", 0):.2f}
            - Cost: ${product_data.get("cost", 0):.2f}
            - Inventory Level: {product_data.get("inventory_level", 0)}
            - Days Without Sale: {product_data.get("days_without_sale", 0)}
            - Health Necessity Score: {health_necessity} (0=luxury wellness, 1=essential healthcare)
            - Expiration Date: {expiration_date}
            
            Market Context:
            - Demand Trend: {market_context.get("demand_trend", "stable")}
            - Competitor Activity: {market_context.get("competitor_activity", "normal")}
            - Seasonal Factor: {market_context.get("seasonal_factor", 1.0)}
            - Season: {market_context.get("current_season", "unknown")}
            
            Healthcare/Wellness Pricing Considerations:
            - Essential healthcare products (vitamins, first aid) are typically inelastic (-0.3 to -0.8)
            - Wellness/fitness products are more elastic (-0.8 to -2.0)
            - Winter immune support products can command 15-25% premium
            - Summer fitness gear sees 10-20% demand increase
            - Supplements nearing expiration need accelerated markdowns
            - Regulatory compliance requires minimum 15% margin for healthcare products
            
            Please use the available tools to:
            1. Analyze demand elasticity considering healthcare/wellness category
            2. Get current competitor prices from healthcare/wellness brands
            3. Calculate optimal pricing considering health necessity and seasonality
            4. Evaluate price impact with healthcare compliance constraints
            5. Consider category-appropriate markdown strategies
            6. Review historical pricing decisions for similar healthcare/wellness products
            
            Provide a comprehensive pricing recommendation with healthcare/wellness-specific rationale.
            """

            # Use the Strands agent to process the request
            response = self.agent(prompt)

            # Extract decision parameters from the agent's response
            # In a real implementation, this would parse the structured response
            decision_params = {
                "product_id": product_data.get("id"),
                "analysis_performed": True,
                "tools_used": [
                    "demand_elasticity",
                    "competitor_analysis",
                    "optimal_pricing",
                ],
                "recommendation_source": "strands_agent_analysis",
            }

            # Create agent decision record
            decision = AgentDecision(
                agent_id=self.agent_id,
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters=decision_params,
                rationale=f"Comprehensive pricing analysis using Strands agent with multiple tools: {response[:200]}...",
                confidence_score=0.85,
                expected_outcome={
                    "analysis_type": "comprehensive",
                    "tools_utilized": len(decision_params.get("tools_used", [])),
                    "agent_response_length": len(response),
                },
                context={
                    "product_data": product_data,
                    "market_context": market_context,
                    "agent_response": response,
                },
            )

            # Store decision in memory
            memory_id = agent_memory.store_decision(
                agent_id=self.agent_id,
                decision=decision,
                context={
                    "product_data": product_data,
                    "market_context": market_context,
                },
            )

            logger.info(
                "product_id=<%s>, decision_id=<%s>, memory_id=<%s> | pricing decision completed",
                product_data.get("id", "unknown"),
                str(decision.id),
                memory_id,
            )

            return decision

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to make pricing decision",
                product_data.get("id", "unknown"),
                str(e),
            )

            # Return fallback decision
            return AgentDecision(
                agent_id=self.agent_id,
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={"error": str(e)},
                rationale=f"Failed to complete pricing analysis: {str(e)}",
                confidence_score=0.1,
                expected_outcome={"status": "error"},
                context={"error": str(e)},
            )

    def update_decision_outcome(self, decision_id: str, outcome_data: Dict[str, Any]):
        """
        Update the outcome of a previous pricing decision for learning.

        Args:
            decision_id: ID of the decision to update
            outcome_data: Actual outcome data
        """
        try:
            # Find the memory entry for this decision
            decision_history = agent_memory.get_agent_decision_history(
                agent_id=self.agent_id, limit=100
            )

            memory_id = None
            for decision_data in decision_history:
                if decision_data.get("decision", {}).get("id") == decision_id:
                    memory_id = decision_data.get("metadata", {}).get("memory_id")
                    break

            if memory_id:
                agent_memory.update_outcome(memory_id, outcome_data)
                logger.info(
                    "decision_id=<%s>, memory_id=<%s> | decision outcome updated",
                    decision_id,
                    memory_id,
                )
            else:
                logger.warning(
                    "decision_id=<%s> | decision not found in memory", decision_id
                )

        except Exception as e:
            logger.error(
                "decision_id=<%s>, error=<%s> | failed to update decision outcome",
                decision_id,
                str(e),
            )


# Global pricing agent instance
pricing_agent = PricingAgent()
