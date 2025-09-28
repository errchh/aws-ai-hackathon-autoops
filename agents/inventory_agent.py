"""
Inventory Agent implementation using AWS Strands framework.

This module implements the Inventory Agent that handles demand forecasting,
safety buffer calculations, restock alert generation, and slow-moving inventory
identification using probabilistic models and historical data analysis.
"""

import logging
import math
import statistics
from datetime import datetime, timezone, timedelta
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


class InventoryAgent:
    """
    Inventory Agent for demand forecasting and inventory optimization.

    This agent uses AWS Strands framework with Anthropic Claude to make
    intelligent inventory decisions based on probabilistic demand forecasting,
    safety buffer calculations, and historical data analysis.
    """

    def __init__(self):
        """Initialize the Inventory Agent with Strands framework."""
        self.settings = get_settings()
        self.agent_id = "inventory_agent"

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
                self.forecast_demand_probabilistic,
                self.calculate_safety_buffer,
                self.generate_restock_alert,
                self.identify_slow_moving_inventory,
                self.analyze_demand_patterns,
                self.retrieve_inventory_history,
            ],
        )

        logger.info(
            "agent_id=<%s> | Inventory Agent initialized with Strands framework",
            self.agent_id,
        )

    @tool
    @observe(name="inventory_agent_forecast_demand_probabilistic")
    def forecast_demand_probabilistic(
        self, product_data: Dict[str, Any], forecast_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate probabilistic demand forecast using historical data analysis.

        Args:
            product_data: Product information including historical sales data
            forecast_days: Number of days to forecast ahead

        Returns:
            Dictionary containing probabilistic demand forecast
        """
        try:
            logger.debug(
                "product_id=<%s>, forecast_days=<%d> | generating probabilistic demand forecast",
                product_data.get("id", "unknown"),
                forecast_days,
            )

            product_id = product_data.get("id", "unknown")
            historical_sales = product_data.get("historical_sales", [])

            if len(historical_sales) < 7:
                # Insufficient data - use conservative estimates
                base_demand = product_data.get("average_daily_sales", 5.0)
                demand_variance = base_demand * 0.3  # 30% variance assumption

                return {
                    "product_id": product_id,
                    "forecast_days": forecast_days,
                    "total_expected_demand": round(base_demand * forecast_days, 2),
                    "daily_expected_demand": round(base_demand, 2),
                    "demand_variance": round(demand_variance, 2),
                    "confidence_level": 0.4,
                    "forecast_method": "insufficient_data_fallback",
                    "analysis": "Used conservative estimates due to insufficient historical data",
                }

            # Calculate historical statistics
            daily_sales = [
                sale["quantity"] for sale in historical_sales[-30:]
            ]  # Last 30 days
            mean_demand = statistics.mean(daily_sales)
            demand_variance = (
                statistics.variance(daily_sales)
                if len(daily_sales) > 1
                else mean_demand * 0.2
            )
            demand_std_dev = math.sqrt(demand_variance)

            # Detect seasonality patterns
            seasonality_factor = self._detect_seasonality(daily_sales)

            # Detect trend
            trend_factor = self._calculate_trend(daily_sales)

            # Apply healthcare/wellness seasonal adjustments
            product_category = product_data.get("category", "general")
            seasonal_category = product_data.get("seasonal_category", "year_round")
            current_month = datetime.now().month

            # Healthcare/wellness seasonal multipliers
            seasonal_multiplier = 1.0
            if "immune" in seasonal_category or "vitamin" in product_category.lower():
                # Winter immune support surge (Oct-Mar)
                if current_month in [10, 11, 12, 1, 2, 3]:
                    seasonal_multiplier = 2.5  # 250% increase in winter
                elif current_month in [4, 5]:
                    seasonal_multiplier = 1.2  # Moderate spring demand
                else:
                    seasonal_multiplier = 0.8  # Lower summer demand
            elif (
                "fitness" in seasonal_category or "exercise" in product_category.lower()
            ):
                # New Year fitness surge (Nov-Apr)
                if current_month in [11, 12, 1, 2, 3, 4]:
                    seasonal_multiplier = 1.8  # 180% increase for fitness
                else:
                    seasonal_multiplier = 0.9  # Moderate off-season
            elif (
                "stress" in seasonal_category
                or "essential_oil" in product_category.lower()
            ):
                # Consistent year-round demand with slight winter increase
                if current_month in [11, 12, 1, 2]:
                    seasonal_multiplier = 1.15  # 15% winter stress increase
                else:
                    seasonal_multiplier = 1.0

            # Generate probabilistic forecast
            forecast_points = []
            total_expected_demand = 0

            for day in range(1, forecast_days + 1):
                # Base forecast with trend
                base_forecast = mean_demand * (
                    1 + trend_factor * day / 30
                )  # Monthly trend application

                # Apply seasonality (simplified weekly pattern)
                current_date = datetime.now(timezone.utc) + timedelta(days=day)
                weekly_seasonal = seasonality_factor.get(current_date.weekday(), 1.0)

                # Apply healthcare/wellness seasonal adjustment
                daily_forecast = base_forecast * weekly_seasonal * seasonal_multiplier
                total_expected_demand += daily_forecast

                # Calculate confidence intervals (95% confidence)
                confidence_margin = 1.96 * demand_std_dev  # 95% confidence interval
                lower_bound = max(0, daily_forecast - confidence_margin)
                upper_bound = daily_forecast + confidence_margin

                forecast_points.append(
                    {
                        "day": day,
                        "date": current_date.date().isoformat(),
                        "expected_demand": round(daily_forecast, 2),
                        "lower_bound": round(lower_bound, 2),
                        "upper_bound": round(upper_bound, 2),
                        "confidence_level": 0.95,
                    }
                )

            # Calculate forecast accuracy based on historical performance
            forecast_accuracy = self._calculate_forecast_accuracy(
                daily_sales, mean_demand, demand_std_dev
            )

            result = {
                "product_id": product_id,
                "forecast_days": forecast_days,
                "total_expected_demand": round(total_expected_demand, 2),
                "daily_expected_demand": round(mean_demand, 2),
                "demand_variance": round(demand_variance, 2),
                "demand_std_dev": round(demand_std_dev, 2),
                "trend_factor": round(trend_factor, 4),
                "seasonality_detected": len(seasonality_factor) > 1,
                "forecast_accuracy": round(forecast_accuracy, 3),
                "confidence_level": 0.85,
                "forecast_points": forecast_points[:7],  # Return first week for brevity
                "forecast_method": "probabilistic_with_trend_seasonality",
                "analysis": f"Generated {forecast_days}-day probabilistic forecast with {forecast_accuracy:.1%} historical accuracy",
            }

            logger.info(
                "product_id=<%s>, total_demand=<%f>, accuracy=<%f> | probabilistic demand forecast generated",
                product_id,
                total_expected_demand,
                forecast_accuracy,
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to generate probabilistic demand forecast",
                product_data.get("id", "unknown"),
                str(e),
            )
            return {
                "product_id": product_data.get("id", "unknown"),
                "forecast_days": forecast_days,
                "total_expected_demand": 0,
                "confidence_level": 0.1,
                "forecast_method": "error_fallback",
                "analysis": f"Error in demand forecasting: {str(e)}",
            }

    @observe(name="inventory_agent_calculate_safety_buffer")
    @tool
    def calculate_safety_buffer(
        self, product_data: Dict[str, Any], service_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate optimal safety buffer based on demand variance and lead time.

        Args:
            product_data: Product information including demand statistics
            service_level: Desired service level (0.0 to 1.0)

        Returns:
            Dictionary containing safety buffer calculations
        """
        try:
            logger.debug(
                "product_id=<%s>, service_level=<%f> | calculating safety buffer",
                product_data.get("id", "unknown"),
                service_level,
            )

            product_id = product_data.get("id", "unknown")
            product_category = product_data.get("category", "general")
            health_necessity = product_data.get("health_necessity_score", 0.5)
            daily_demand = product_data.get("daily_expected_demand", 5.0)
            demand_variance = product_data.get("demand_variance", 2.0)
            lead_time_days = product_data.get("supplier_lead_time", 7)

            # Determine healthcare/wellness-specific service level if not provided
            if service_level == 0.95:  # Default case, adjust based on category
                if (
                    health_necessity >= 0.8
                    or "vitamin" in product_category.lower()
                    or "first_aid" in product_category.lower()
                ):
                    service_level = 0.98  # 98% for essential healthcare
                elif (
                    health_necessity >= 0.6 or "supplement" in product_category.lower()
                ):
                    service_level = 0.95  # 95% for specialty supplements
                elif (
                    "fitness" in product_category.lower()
                    or "wellness" in product_category.lower()
                ):
                    service_level = 0.90  # 90% for fitness/wellness accessories
                else:
                    service_level = 0.85  # 85% for luxury wellness items

            # Calculate demand standard deviation
            demand_std_dev = math.sqrt(demand_variance)

            # Extended Z-score mapping for healthcare service levels
            z_scores = {0.85: 1.04, 0.90: 1.28, 0.95: 1.65, 0.98: 2.05, 0.99: 2.33}
            z_score = z_scores.get(service_level, 1.65)  # Default to 95%

            # Safety stock calculation: Z * sqrt(lead_time) * demand_std_dev
            # This accounts for demand variability during lead time
            safety_stock = z_score * math.sqrt(lead_time_days) * demand_std_dev

            # Calculate reorder point: (average demand * lead time) + safety stock
            reorder_point = (daily_demand * lead_time_days) + safety_stock

            # Calculate maximum inventory level (for ABC analysis)
            # Assuming EOQ-based ordering
            annual_demand = daily_demand * 365
            order_cost = product_data.get("order_cost", 50)  # Fixed ordering cost
            carrying_cost_rate = product_data.get(
                "carrying_cost_rate", 0.20
            )  # 20% annual carrying cost
            unit_cost = product_data.get("cost", 10)

            # Economic Order Quantity (EOQ)
            if carrying_cost_rate > 0 and unit_cost > 0:
                eoq = math.sqrt(
                    (2 * annual_demand * order_cost) / (carrying_cost_rate * unit_cost)
                )
            else:
                eoq = daily_demand * 30  # 30-day supply fallback

            max_inventory = reorder_point + eoq

            # Calculate buffer effectiveness metrics
            stockout_probability = 1 - service_level
            expected_stockouts_per_year = (365 / lead_time_days) * stockout_probability

            # Calculate carrying cost impact
            safety_stock_carrying_cost = safety_stock * unit_cost * carrying_cost_rate

            result = {
                "product_id": product_id,
                "service_level": service_level,
                "safety_stock": round(safety_stock, 0),
                "reorder_point": round(reorder_point, 0),
                "economic_order_quantity": round(eoq, 0),
                "max_inventory_level": round(max_inventory, 0),
                "demand_std_dev": round(demand_std_dev, 2),
                "lead_time_days": lead_time_days,
                "z_score": z_score,
                "expected_stockouts_per_year": round(expected_stockouts_per_year, 2),
                "safety_stock_carrying_cost": round(safety_stock_carrying_cost, 2),
                "buffer_efficiency": round(safety_stock / reorder_point, 3),
                "analysis": f"Safety buffer of {safety_stock:.0f} units provides {service_level:.1%} service level",
            }

            logger.info(
                "product_id=<%s>, safety_stock=<%f>, reorder_point=<%f> | safety buffer calculated",
                product_id,
                safety_stock,
                reorder_point,
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to calculate safety buffer",
                product_data.get("id", "unknown"),
                str(e),
            )
            return {
                "product_id": product_data.get("id", "unknown"),
                "safety_stock": 0,
                "reorder_point": 0,
                "analysis": f"Error in safety buffer calculation: {str(e)}",
            }

    @observe(name="inventory_agent_generate_restock_alert")
    @tool
    def generate_restock_alert(
        self, product_data: Dict[str, Any], urgency_level: str = "medium"
    ) -> Dict[str, Any]:
        """
        Generate restock alert with recommended quantities based on demand forecast.

        Args:
            product_data: Product information including current inventory
            urgency_level: Alert urgency (low, medium, high, critical)

        Returns:
            Dictionary containing restock alert details
        """
        try:
            logger.debug(
                "product_id=<%s>, urgency=<%s> | generating restock alert",
                product_data.get("id", "unknown"),
                urgency_level,
            )

            product_id = product_data.get("id", "unknown")
            current_stock = product_data.get("inventory_level", 0)
            reorder_point = product_data.get("reorder_point", 25)
            daily_demand = product_data.get("daily_expected_demand", 5.0)
            lead_time_days = product_data.get("supplier_lead_time", 7)
            eoq = product_data.get("economic_order_quantity", 100)

            # Determine recommended quantity based on urgency
            urgency_multipliers = {
                "low": 1.0,
                "medium": 1.2,
                "high": 1.5,
                "critical": 2.0,
            }

            base_quantity = eoq
            urgency_multiplier = urgency_multipliers.get(urgency_level, 1.2)
            recommended_quantity = int(base_quantity * urgency_multiplier)

            # Calculate days until stockout
            if daily_demand > 0:
                days_until_stockout = max(0, current_stock / daily_demand)
            else:
                days_until_stockout = float("inf")

            # Calculate stockout risk
            if current_stock <= reorder_point:
                stockout_risk = min(
                    1.0, (reorder_point - current_stock) / reorder_point
                )
            else:
                stockout_risk = max(
                    0.0, (reorder_point - current_stock) / (reorder_point * 2)
                )

            # Estimate costs
            unit_cost = product_data.get("cost", 10)
            order_cost = product_data.get("order_cost", 50)
            total_cost = (recommended_quantity * unit_cost) + order_cost

            # Calculate expected delivery date
            expected_delivery = datetime.now(timezone.utc) + timedelta(
                days=lead_time_days
            )

            # Generate priority score (0-100)
            priority_score = min(
                100,
                int(
                    (stockout_risk * 40)  # Risk component
                    + (urgency_multipliers[urgency_level] * 20)  # Urgency component
                    + (
                        max(0, (reorder_point - current_stock) / reorder_point) * 40
                    )  # Stock level component
                ),
            )

            result = {
                "product_id": product_id,
                "alert_id": str(uuid4()),
                "urgency_level": urgency_level,
                "current_stock": current_stock,
                "reorder_point": reorder_point,
                "recommended_quantity": recommended_quantity,
                "total_cost": round(total_cost, 2),
                "unit_cost": unit_cost,
                "days_until_stockout": round(days_until_stockout, 1),
                "stockout_risk": round(stockout_risk, 3),
                "priority_score": priority_score,
                "expected_delivery": expected_delivery.isoformat(),
                "lead_time_days": lead_time_days,
                "rationale": f"Stock level {current_stock} below reorder point {reorder_point}. Recommend ordering {recommended_quantity} units.",
                "analysis": f"Generated {urgency_level} urgency restock alert with {stockout_risk:.1%} stockout risk",
            }

            logger.info(
                "product_id=<%s>, recommended_qty=<%d>, priority=<%d> | restock alert generated",
                product_id,
                recommended_quantity,
                priority_score,
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to generate restock alert",
                product_data.get("id", "unknown"),
                str(e),
            )
            return {
                "product_id": product_data.get("id", "unknown"),
                "recommended_quantity": 0,
                "analysis": f"Error generating restock alert: {str(e)}",
            }

    @observe(name="inventory_agent_identify_slow_moving_inventory")
    @tool
    def identify_slow_moving_inventory(
        self, inventory_data: List[Dict[str, Any]], threshold_days: int = 30
    ) -> Dict[str, Any]:
        """
        Identify slow-moving inventory items based on sales velocity and aging.

        Args:
            inventory_data: List of product inventory information
            threshold_days: Days without sale to consider slow-moving

        Returns:
            Dictionary containing slow-moving inventory analysis
        """
        try:
            logger.debug(
                "inventory_items=<%d>, threshold_days=<%d> | identifying slow-moving inventory",
                len(inventory_data),
                threshold_days,
            )

            slow_moving_items = []
            total_slow_moving_value = 0

            for product in inventory_data:
                product_id = product.get("id", "unknown")
                current_stock = product.get("inventory_level", 0)
                days_without_sale = product.get("days_without_sale", 0)
                daily_demand = product.get("daily_expected_demand", 0)
                unit_cost = product.get("cost", 0)
                current_price = product.get("current_price", 0)

                # Calculate inventory value
                inventory_value = current_stock * unit_cost

                # Determine if item is slow-moving
                is_slow_moving = False
                slow_moving_reasons = []

                if days_without_sale >= threshold_days:
                    is_slow_moving = True
                    slow_moving_reasons.append(f"No sales for {days_without_sale} days")

                if daily_demand > 0:
                    days_of_supply = current_stock / daily_demand
                    if days_of_supply > 90:  # More than 3 months supply
                        is_slow_moving = True
                        slow_moving_reasons.append(
                            f"Excessive supply: {days_of_supply:.0f} days"
                        )
                elif current_stock > 0:
                    is_slow_moving = True
                    slow_moving_reasons.append("Zero demand with inventory")

                if is_slow_moving:
                    # Calculate recommended actions
                    recommended_actions = []

                    # Markdown recommendation
                    if current_price > unit_cost * 1.1:  # Has margin for markdown
                        markdown_percentage = min(30, max(10, days_without_sale // 10))
                        recommended_actions.append(
                            f"Apply {markdown_percentage}% markdown"
                        )

                    # Bundle recommendation
                    if current_stock > 20:
                        recommended_actions.append(
                            "Consider bundling with fast-moving items"
                        )

                    # Liquidation recommendation
                    if days_without_sale > 90:
                        recommended_actions.append("Consider liquidation or donation")

                    slow_moving_item = {
                        "product_id": product_id,
                        "current_stock": current_stock,
                        "days_without_sale": days_without_sale,
                        "daily_demand": daily_demand,
                        "inventory_value": round(inventory_value, 2),
                        "days_of_supply": round(days_of_supply, 1)
                        if daily_demand > 0
                        else float("inf"),
                        "reasons": slow_moving_reasons,
                        "recommended_actions": recommended_actions,
                        "urgency_score": min(
                            100,
                            int(
                                days_without_sale / threshold_days * 50
                                + (inventory_value / 1000) * 25
                            ),
                        ),
                    }

                    slow_moving_items.append(slow_moving_item)
                    total_slow_moving_value += inventory_value

            # Sort by urgency score (highest first)
            slow_moving_items.sort(key=lambda x: x["urgency_score"], reverse=True)

            # Calculate summary statistics
            total_inventory_value = sum(
                p.get("inventory_level", 0) * p.get("cost", 0) for p in inventory_data
            )
            slow_moving_percentage = (
                (total_slow_moving_value / total_inventory_value * 100)
                if total_inventory_value > 0
                else 0
            )

            result = {
                "analysis_date": datetime.now(timezone.utc).isoformat(),
                "threshold_days": threshold_days,
                "total_products_analyzed": len(inventory_data),
                "slow_moving_count": len(slow_moving_items),
                "slow_moving_percentage": round(slow_moving_percentage, 1),
                "total_slow_moving_value": round(total_slow_moving_value, 2),
                "total_inventory_value": round(total_inventory_value, 2),
                "slow_moving_items": slow_moving_items[:10],  # Top 10 by urgency
                "summary_recommendations": self._generate_slow_moving_summary(
                    slow_moving_items
                ),
                "analysis": f"Identified {len(slow_moving_items)} slow-moving items worth ${total_slow_moving_value:,.2f}",
            }

            logger.info(
                "slow_moving_count=<%d>, total_value=<%f> | slow-moving inventory identified",
                len(slow_moving_items),
                total_slow_moving_value,
            )

            return result

        except Exception as e:
            logger.error(
                "error=<%s> | failed to identify slow-moving inventory", str(e)
            )
            return {
                "slow_moving_count": 0,
                "analysis": f"Error identifying slow-moving inventory: {str(e)}",
            }

    @tool
    def analyze_demand_patterns(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze demand patterns to identify trends, seasonality, and anomalies.

        Args:
            product_data: Product information with historical sales data

        Returns:
            Dictionary containing demand pattern analysis
        """
        try:
            logger.debug(
                "product_id=<%s> | analyzing demand patterns",
                product_data.get("id", "unknown"),
            )

            product_id = product_data.get("id", "unknown")
            historical_sales = product_data.get("historical_sales", [])

            if len(historical_sales) < 14:
                return {
                    "product_id": product_id,
                    "pattern_analysis": "insufficient_data",
                    "analysis": "Need at least 14 days of data for pattern analysis",
                }

            # Extract daily sales quantities
            daily_sales = [
                sale["quantity"] for sale in historical_sales[-60:]
            ]  # Last 60 days
            dates = [sale["date"] for sale in historical_sales[-60:]]

            # Calculate basic statistics
            mean_demand = statistics.mean(daily_sales)
            median_demand = statistics.median(daily_sales)
            std_dev = statistics.stdev(daily_sales) if len(daily_sales) > 1 else 0
            coefficient_of_variation = std_dev / mean_demand if mean_demand > 0 else 0

            # Detect trend
            trend_analysis = self._analyze_trend(daily_sales)

            # Detect seasonality
            seasonality_analysis = self._analyze_seasonality(daily_sales, dates)

            # Detect anomalies
            anomalies = self._detect_anomalies(daily_sales)

            # Calculate demand stability
            if coefficient_of_variation < 0.2:
                stability = "very_stable"
            elif coefficient_of_variation < 0.5:
                stability = "stable"
            elif coefficient_of_variation < 1.0:
                stability = "moderate"
            else:
                stability = "volatile"

            result = {
                "product_id": product_id,
                "analysis_period_days": len(daily_sales),
                "mean_daily_demand": round(mean_demand, 2),
                "median_daily_demand": round(median_demand, 2),
                "demand_std_dev": round(std_dev, 2),
                "coefficient_of_variation": round(coefficient_of_variation, 3),
                "demand_stability": stability,
                "trend_analysis": trend_analysis,
                "seasonality_analysis": seasonality_analysis,
                "anomalies_detected": len(anomalies),
                "anomaly_details": anomalies[:5],  # Top 5 anomalies
                "forecasting_recommendations": self._get_forecasting_recommendations(
                    stability, trend_analysis, seasonality_analysis
                ),
                "analysis": f"Demand shows {stability} pattern with {trend_analysis['direction']} trend",
            }

            logger.info(
                "product_id=<%s>, stability=<%s>, trend=<%s> | demand patterns analyzed",
                product_id,
                stability,
                trend_analysis["direction"],
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to analyze demand patterns",
                product_data.get("id", "unknown"),
                str(e),
            )
            return {
                "product_id": product_data.get("id", "unknown"),
                "pattern_analysis": "error",
                "analysis": f"Error analyzing demand patterns: {str(e)}",
            }

    @tool
    def retrieve_inventory_history(
        self, product_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Retrieve historical inventory decisions and outcomes from agent memory.

        Args:
            product_id: Product identifier
            days: Number of days of history to retrieve

        Returns:
            Dictionary containing inventory history and insights
        """
        try:
            logger.debug(
                "product_id=<%s>, days=<%d> | retrieving inventory history",
                product_id,
                days,
            )

            # Retrieve similar inventory decisions from memory
            context = {"product_id": product_id}
            similar_decisions = agent_memory.retrieve_similar_decisions(
                agent_id=self.agent_id,
                current_context=context,
                action_type=ActionType.INVENTORY_RESTOCK.value,
                limit=10,
            )

            # Get agent decision history
            decision_history = agent_memory.get_agent_decision_history(
                agent_id=self.agent_id,
                action_type=ActionType.INVENTORY_RESTOCK.value,
                limit=20,
                include_outcomes=True,
            )

            # Analyze inventory management patterns
            restock_decisions = []
            successful_decisions = 0
            total_decisions = 0

            for decision_data, similarity in similar_decisions:
                decision = decision_data.get("decision", {})
                outcome = decision_data.get("outcome", {})

                if decision and outcome:
                    total_decisions += 1
                    if outcome.get("success", False):
                        successful_decisions += 1

                    restock_decisions.append(
                        {
                            "timestamp": decision.get("timestamp"),
                            "recommended_quantity": decision.get("parameters", {}).get(
                                "recommended_quantity", 0
                            ),
                            "actual_quantity": outcome.get("actual_quantity", 0),
                            "success": outcome.get("success", False),
                            "similarity": similarity,
                            "stockout_prevented": outcome.get(
                                "stockout_prevented", False
                            ),
                        }
                    )

            success_rate = (
                (successful_decisions / total_decisions) if total_decisions > 0 else 0
            )

            # Calculate forecast accuracy from past decisions
            forecast_accuracy = self._calculate_historical_forecast_accuracy(
                decision_history
            )

            result = {
                "product_id": product_id,
                "total_decisions": len(decision_history),
                "similar_decisions": len(similar_decisions),
                "success_rate": round(success_rate, 2),
                "forecast_accuracy": round(forecast_accuracy, 2),
                "restock_decisions": restock_decisions[:5],  # Last 5 decisions
                "insights": {
                    "most_successful_strategy": "proactive_restocking"
                    if success_rate > 0.7
                    else "reactive_restocking",
                    "average_similarity": round(
                        sum(s for _, s in similar_decisions) / len(similar_decisions), 2
                    )
                    if similar_decisions
                    else 0,
                    "stockout_prevention_rate": self._calculate_stockout_prevention_rate(
                        restock_decisions
                    ),
                },
                "learning_recommendations": self._generate_learning_recommendations(
                    success_rate, forecast_accuracy
                ),
                "analysis": f"Found {len(decision_history)} historical decisions with {success_rate:.1%} success rate",
            }

            logger.info(
                "product_id=<%s>, decisions=<%d>, success_rate=<%f> | inventory history retrieved",
                product_id,
                len(decision_history),
                success_rate,
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to retrieve inventory history",
                product_id,
                str(e),
            )
            return {
                "product_id": product_id,
                "total_decisions": 0,
                "analysis": f"Error retrieving inventory history: {str(e)}",
            }

    # Helper methods for analysis
    def _detect_seasonality(self, daily_sales: List[float]) -> Dict[int, float]:
        """Detect weekly seasonality patterns in sales data."""
        if len(daily_sales) < 14:
            return {}

        # Group by day of week (simplified seasonality detection)
        weekday_sales = {}
        for i, sales in enumerate(daily_sales):
            weekday = i % 7
            if weekday not in weekday_sales:
                weekday_sales[weekday] = []
            weekday_sales[weekday].append(sales)

        # Calculate average for each weekday
        weekday_averages = {}
        overall_average = statistics.mean(daily_sales)

        for weekday, sales_list in weekday_sales.items():
            if sales_list:
                avg = statistics.mean(sales_list)
                weekday_averages[weekday] = (
                    avg / overall_average if overall_average > 0 else 1.0
                )

        return weekday_averages

    def _calculate_trend(self, daily_sales: List[float]) -> float:
        """Calculate trend factor using simple linear regression."""
        if len(daily_sales) < 7:
            return 0.0

        n = len(daily_sales)
        x_values = list(range(n))

        # Calculate linear regression slope
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(daily_sales)

        numerator = sum(
            (x - x_mean) * (y - y_mean) for x, y in zip(x_values, daily_sales)
        )
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        return slope / y_mean if y_mean > 0 else 0.0  # Normalize by mean

    def _calculate_forecast_accuracy(
        self, daily_sales: List[float], mean_demand: float, std_dev: float
    ) -> float:
        """Calculate forecast accuracy based on how well mean predicts actual values."""
        if len(daily_sales) < 7:
            return 0.5

        # Calculate Mean Absolute Percentage Error (MAPE)
        errors = []
        for actual in daily_sales:
            if actual > 0:
                error = abs(actual - mean_demand) / actual
                errors.append(error)

        if not errors:
            return 0.5

        mape = statistics.mean(errors)
        accuracy = max(0.1, 1 - mape)  # Convert MAPE to accuracy
        return min(0.95, accuracy)  # Cap at 95%

    def _generate_slow_moving_summary(
        self, slow_moving_items: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate summary recommendations for slow-moving inventory."""
        if not slow_moving_items:
            return ["No slow-moving inventory detected"]

        recommendations = []

        # High-value items
        high_value_items = [
            item for item in slow_moving_items if item["inventory_value"] > 1000
        ]
        if high_value_items:
            recommendations.append(
                f"Priority: {len(high_value_items)} high-value items need immediate attention"
            )

        # Long-stagnant items
        very_old_items = [
            item for item in slow_moving_items if item["days_without_sale"] > 90
        ]
        if very_old_items:
            recommendations.append(
                f"Consider liquidation for {len(very_old_items)} items stagnant >90 days"
            )

        # Markdown opportunities
        markdown_items = [
            item
            for item in slow_moving_items
            if any(
                "markdown" in action.lower() for action in item["recommended_actions"]
            )
        ]
        if markdown_items:
            recommendations.append(
                f"Apply markdowns to {len(markdown_items)} items to accelerate turnover"
            )

        return recommendations

    def _analyze_trend(self, daily_sales: List[float]) -> Dict[str, Any]:
        """Analyze trend direction and strength."""
        if len(daily_sales) < 7:
            return {"direction": "insufficient_data", "strength": 0}

        trend_factor = self._calculate_trend(daily_sales)

        if abs(trend_factor) < 0.01:
            direction = "stable"
            strength = "weak"
        elif trend_factor > 0.05:
            direction = "increasing"
            strength = "strong" if trend_factor > 0.1 else "moderate"
        elif trend_factor < -0.05:
            direction = "decreasing"
            strength = "strong" if trend_factor < -0.1 else "moderate"
        else:
            direction = "stable"
            strength = "weak"

        return {
            "direction": direction,
            "strength": strength,
            "trend_factor": round(trend_factor, 4),
            "confidence": min(
                0.9, len(daily_sales) / 30
            ),  # More data = higher confidence
        }

    def _analyze_seasonality(
        self, daily_sales: List[float], dates: List[str]
    ) -> Dict[str, Any]:
        """Analyze seasonality patterns in demand."""
        if len(daily_sales) < 14:
            return {"detected": False, "pattern": "insufficient_data"}

        seasonality_factors = self._detect_seasonality(daily_sales)

        if not seasonality_factors:
            return {"detected": False, "pattern": "none"}

        # Check if there's significant variation
        factor_values = list(seasonality_factors.values())
        if max(factor_values) - min(factor_values) < 0.2:
            return {"detected": False, "pattern": "minimal_variation"}

        # Identify pattern
        weekend_factor = (
            seasonality_factors.get(5, 1.0) + seasonality_factors.get(6, 1.0)
        ) / 2
        weekday_factor = sum(seasonality_factors.get(i, 1.0) for i in range(5)) / 5

        if weekend_factor > weekday_factor * 1.2:
            pattern = "weekend_peak"
        elif weekday_factor > weekend_factor * 1.2:
            pattern = "weekday_peak"
        else:
            pattern = "mixed"

        return {
            "detected": True,
            "pattern": pattern,
            "weekend_factor": round(weekend_factor, 2),
            "weekday_factor": round(weekday_factor, 2),
            "seasonality_strength": round(max(factor_values) - min(factor_values), 2),
        }

    def _detect_anomalies(self, daily_sales: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in sales data using statistical methods."""
        if len(daily_sales) < 7:
            return []

        mean_sales = statistics.mean(daily_sales)
        std_dev = statistics.stdev(daily_sales) if len(daily_sales) > 1 else 0

        if std_dev == 0:
            return []

        anomalies = []
        threshold = 2.5  # 2.5 standard deviations

        for i, sales in enumerate(daily_sales):
            z_score = abs(sales - mean_sales) / std_dev
            if z_score > threshold:
                anomalies.append(
                    {
                        "day_index": i,
                        "sales_value": sales,
                        "z_score": round(z_score, 2),
                        "type": "spike" if sales > mean_sales else "drop",
                        "severity": "high" if z_score > 3.0 else "moderate",
                    }
                )

        return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)

    def _get_forecasting_recommendations(
        self, stability: str, trend_analysis: Dict, seasonality_analysis: Dict
    ) -> List[str]:
        """Generate forecasting recommendations based on demand patterns."""
        recommendations = []

        if stability == "volatile":
            recommendations.append(
                "Use wider confidence intervals due to high demand volatility"
            )
            recommendations.append(
                "Consider shorter forecast horizons for better accuracy"
            )

        if trend_analysis["direction"] == "increasing" and trend_analysis[
            "strength"
        ] in ["moderate", "strong"]:
            recommendations.append(
                "Incorporate growth trend into safety stock calculations"
            )
        elif trend_analysis["direction"] == "decreasing":
            recommendations.append(
                "Consider reducing order quantities due to declining demand"
            )

        if seasonality_analysis.get("detected", False):
            recommendations.append(
                f"Apply {seasonality_analysis['pattern']} seasonality adjustments"
            )

        if not recommendations:
            recommendations.append(
                "Standard forecasting methods appropriate for stable demand"
            )

        return recommendations

    def _calculate_historical_forecast_accuracy(
        self, decision_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate historical forecast accuracy from past decisions."""
        if not decision_history:
            return 0.7  # Default assumption

        accurate_forecasts = 0
        total_forecasts = 0

        for decision_data in decision_history:
            outcome = decision_data.get("outcome", {})
            if "forecast_accuracy" in outcome:
                total_forecasts += 1
                if outcome["forecast_accuracy"] > 0.8:
                    accurate_forecasts += 1

        if total_forecasts == 0:
            return 0.7

        return accurate_forecasts / total_forecasts

    def _calculate_stockout_prevention_rate(
        self, restock_decisions: List[Dict[str, Any]]
    ) -> float:
        """Calculate rate of successful stockout prevention."""
        if not restock_decisions:
            return 0.0

        prevented = sum(
            1
            for decision in restock_decisions
            if decision.get("stockout_prevented", False)
        )
        return prevented / len(restock_decisions)

    def _generate_learning_recommendations(
        self, success_rate: float, forecast_accuracy: float
    ) -> List[str]:
        """Generate recommendations for improving agent performance."""
        recommendations = []

        if success_rate < 0.7:
            recommendations.append("Review restock timing - consider earlier alerts")
            recommendations.append(
                "Analyze failed decisions to improve decision criteria"
            )

        if forecast_accuracy < 0.7:
            recommendations.append(
                "Incorporate more historical data for better forecasting"
            )
            recommendations.append(
                "Consider external factors (promotions, seasonality)"
            )

        if success_rate > 0.8 and forecast_accuracy > 0.8:
            recommendations.append(
                "Performance is excellent - maintain current strategies"
            )

        return (
            recommendations
            if recommendations
            else ["Continue monitoring and learning from outcomes"]
        )

    def make_inventory_decision(
        self, product_data: Dict[str, Any], context: Dict[str, Any]
    ) -> AgentDecision:
        """
        Make a comprehensive inventory decision using all available tools and analysis.

        Args:
            product_data: Complete product information
            context: Additional context for the decision

        Returns:
            AgentDecision object with the inventory recommendation
        """
        try:
            logger.info(
                "product_id=<%s> | making inventory decision",
                product_data.get("id", "unknown"),
            )

            # Construct healthcare/wellness-specific prompt for the agent
            product_category = product_data.get("category", "general")
            health_necessity = product_data.get("health_necessity_score", 0.5)
            expiration_date = product_data.get("expiration_date", "N/A")
            seasonal_category = product_data.get("seasonal_category", "year_round")
            current_month = datetime.now().month

            prompt = f"""
            Analyze the following healthcare/wellness product and context data to make an optimal inventory decision:
            
            Product Data:
            - ID: {product_data.get("id")}
            - Category: {product_category} (Healthcare/Wellness)
            - Current Stock: {product_data.get("inventory_level", 0)}
            - Reorder Point: {product_data.get("reorder_point", 25)}
            - Daily Demand: {product_data.get("daily_expected_demand", 5.0)}
            - Lead Time: {product_data.get("supplier_lead_time", 7)} days
            - Cost: ${product_data.get("cost", 0):.2f}
            - Health Necessity Score: {health_necessity} (0=luxury wellness, 1=essential healthcare)
            - Expiration Date: {expiration_date}
            - Seasonal Category: {seasonal_category}
            - Current Month: {current_month}
            
            Context:
            - Analysis Type: {context.get("analysis_type", "routine_check")}
            - Urgency: {context.get("urgency", "medium")}
            - Market Conditions: {context.get("market_conditions", "normal")}
            - Season: {context.get("current_season", "unknown")}
            
            Healthcare/Wellness Inventory Considerations:
            - Essential healthcare products need 95-98% service levels (vitamins, first aid)
            - Wellness products can operate at 85-95% service levels (fitness, aromatherapy)
            - Winter months (Oct-Mar) see 200-300% increase in immune support demand
            - Summer months (May-Aug) see 150% increase in fitness gear demand
            - Supplements within 90-180 days of expiration need accelerated turnover
            - Healthcare suppliers are more reliable (3-7 days) than wellness (5-14 days)
            - Regulatory compliance requires continuous availability of essential healthcare items
            
            Please use the available tools to:
            1. Generate probabilistic demand forecast considering seasonal health patterns
            2. Calculate optimal safety buffer based on health necessity and expiration dates
            3. Generate restock alert with healthcare/wellness priority levels
            4. Analyze demand patterns for health trends and seasonal variations
            5. Review historical inventory decisions for similar healthcare/wellness products
            6. Check for slow-moving inventory with expiration and seasonal considerations
            
            Provide a comprehensive healthcare/wellness inventory recommendation with clear rationale.
            """

            # Use the Strands agent to process the request
            response = self.agent(prompt)

            # Extract decision parameters from the agent's response
            decision_params = {
                "product_id": product_data.get("id"),
                "analysis_performed": True,
                "tools_used": [
                    "demand_forecast",
                    "safety_buffer",
                    "restock_alert",
                    "pattern_analysis",
                ],
                "recommendation_source": "strands_agent_analysis",
            }

            # Create agent decision record
            decision = AgentDecision(
                agent_id=self.agent_id,
                action_type=ActionType.INVENTORY_RESTOCK,
                parameters=decision_params,
                rationale=f"Comprehensive inventory analysis using Strands agent: {response[:200]}...",
                confidence_score=0.88,
                expected_outcome={
                    "analysis_type": "comprehensive",
                    "tools_utilized": len(decision_params.get("tools_used", [])),
                    "agent_response_length": len(response),
                },
                context={
                    "product_data": product_data,
                    "decision_context": context,
                    "agent_response": response,
                },
            )

            # Store decision in memory
            memory_id = agent_memory.store_decision(
                agent_id=self.agent_id,
                decision=decision,
                context={"product_data": product_data, "decision_context": context},
            )

            logger.info(
                "product_id=<%s>, decision_id=<%s>, memory_id=<%s> | inventory decision completed",
                product_data.get("id", "unknown"),
                str(decision.id),
                memory_id,
            )

            return decision

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to make inventory decision",
                product_data.get("id", "unknown"),
                str(e),
            )

            # Return fallback decision
            return AgentDecision(
                agent_id=self.agent_id,
                action_type=ActionType.INVENTORY_RESTOCK,
                parameters={"error": str(e)},
                rationale=f"Failed to complete inventory analysis: {str(e)}",
                confidence_score=0.1,
                expected_outcome={"status": "error"},
                context={"error": str(e)},
            )

    def update_decision_outcome(self, decision_id: str, outcome_data: Dict[str, Any]):
        """
        Update the outcome of a previous inventory decision for learning.

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


# Global inventory agent instance
inventory_agent = InventoryAgent()
