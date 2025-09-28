"""
Promotion Agent implementation using AWS Strands framework.

This module implements the Promotion Agent that handles promotional campaign creation,
bundle recommendations, social sentiment analysis, and campaign orchestration using
intelligent decision-making and coordination with other agents.
"""

import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from strands import Agent, tool
from strands.models import BedrockModel

from agents.memory import agent_memory
from config.settings import get_settings
from config.langfuse_integration import get_langfuse_integration
from models.core import AgentDecision, ActionType, Product


# Configure logging
logger = logging.getLogger(__name__)


class PromotionAgent:
    """
    Promotion Agent for campaign orchestration and promotional strategy.

    This agent uses AWS Strands framework with Anthropic Claude to make
    intelligent promotional decisions including flash sales, bundle creation,
    sentiment analysis, and campaign coordination.
    """

    def __init__(self):
        """Initialize the Promotion Agent with Strands framework."""
        self.settings = get_settings()
        self.agent_id = "promotion_agent"

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
                self.create_flash_sale,
                self.generate_bundle_recommendation,
                self.analyze_social_sentiment,
                self.schedule_promotional_campaign,
                self.evaluate_campaign_effectiveness,
                self.coordinate_with_pricing_agent,
                self.validate_inventory_availability,
                self.retrieve_promotion_history,
            ],
        )

        # Initialize Langfuse integration service
        self.langfuse_integration = get_langfuse_integration()

        logger.info(
            "agent_id=<%s> | Promotion Agent initialized with Strands framework",
            self.agent_id,
        )

    @tool
    def create_flash_sale(
        self,
        product_data: Dict[str, Any],
        duration_hours: int = 24,
        target_audience: str = "general",
    ) -> Dict[str, Any]:
        """
        Create a flash sale with duration and targeting logic.

        Args:
            product_data: Product information including inventory and pricing
            duration_hours: Duration of the flash sale in hours
            target_audience: Target customer segment (general, premium, budget, loyalty)

        Returns:
            Dictionary containing flash sale details and recommendations
        """
        # Start Langfuse span for flash sale creation
        try:
            span_id = self.langfuse_integration.start_agent_span(
                agent_id=self.agent_id,
                operation="create_flash_sale",
                input_data={
                    "product_id": product_data.get("id", "unknown"),
                    "duration_hours": duration_hours,
                    "target_audience": target_audience,
                },
            )
        except Exception as e:
            logger.warning(
                "product_id=<%s> | failed to start tracing span: %s",
                product_data.get("id", "unknown"),
                str(e),
            )
            span_id = None

        try:
            logger.debug(
                "product_id=<%s>, duration=<%d>, audience=<%s> | creating flash sale",
                product_data.get("id", "unknown"),
                duration_hours,
                target_audience,
            )

            product_id = product_data.get("id", "unknown")
            current_price = product_data.get("current_price", 0)
            cost = product_data.get("cost", 0)
            inventory_level = product_data.get("inventory_level", 0)
            category = product_data.get("category", "general")

            # Determine optimal discount based on target audience and inventory
            audience_discounts = {
                "general": (15, 25),  # 15-25% discount
                "premium": (
                    10,
                    20,
                ),  # 10-20% discount (premium customers less price sensitive)
                "budget": (
                    20,
                    35,
                ),  # 20-35% discount (budget customers more price sensitive)
                "loyalty": (25, 40),  # 25-40% discount (reward loyal customers)
            }

            min_discount, max_discount = audience_discounts.get(
                target_audience, (15, 25)
            )

            # Adjust discount based on inventory levels
            if inventory_level > 100:  # High inventory
                discount_percentage = max_discount
            elif inventory_level > 50:  # Medium inventory
                discount_percentage = (min_discount + max_discount) / 2
            else:  # Low inventory
                discount_percentage = min_discount

            # Ensure minimum margin is maintained
            max_allowable_discount = (
                (current_price - cost * 1.05) / current_price
            ) * 100
            final_discount = min(discount_percentage, max_allowable_discount)

            flash_sale_price = current_price * (1 - final_discount / 100)

            # Calculate expected impact
            # Flash sales typically increase demand by 2-5x normal rate
            demand_multiplier = (
                2.0 + (final_discount / 100) * 6
            )  # Higher discount = higher demand
            expected_units_sold = min(
                inventory_level * 0.6, int(10 * demand_multiplier)
            )  # Cap at 60% of inventory

            # Calculate revenue and profit impact
            normal_daily_revenue = current_price * 5  # Assume 5 units/day normal
            flash_sale_revenue = flash_sale_price * expected_units_sold
            revenue_impact = flash_sale_revenue - (
                normal_daily_revenue * duration_hours / 24
            )

            profit_per_unit = flash_sale_price - cost
            total_profit_impact = profit_per_unit * expected_units_sold

            # Generate campaign timing
            start_time = datetime.now(timezone.utc) + timedelta(
                hours=1
            )  # Start in 1 hour
            end_time = start_time + timedelta(hours=duration_hours)

            # Create campaign messaging based on audience
            audience_messaging = {
                "general": "Limited Time Flash Sale! Don't Miss Out!",
                "premium": "Exclusive Flash Sale for Our Valued Customers",
                "budget": "Massive Savings Alert! Flash Sale Now Live!",
                "loyalty": "VIP Flash Sale - Thank You for Your Loyalty!",
            }

            campaign_message = audience_messaging.get(
                target_audience, "Flash Sale Now Live!"
            )

            result = {
                "flash_sale_id": str(uuid4()),
                "product_id": product_id,
                "campaign_name": f"Flash Sale - {product_data.get('name', 'Product')}",
                "campaign_message": campaign_message,
                "original_price": current_price,
                "flash_sale_price": round(flash_sale_price, 2),
                "discount_percentage": round(final_discount, 1),
                "target_audience": target_audience,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_hours": duration_hours,
                "expected_units_sold": expected_units_sold,
                "revenue_impact": round(revenue_impact, 2),
                "profit_impact": round(total_profit_impact, 2),
                "inventory_allocation": expected_units_sold,
                "urgency_score": min(100, int(final_discount * 2 + duration_hours)),
                "success_probability": 0.75
                + (final_discount / 100) * 0.2,  # Higher discount = higher success
                "rationale": f"Created {duration_hours}h flash sale with {final_discount:.1f}% discount targeting {target_audience} audience",
                "analysis": f"Flash sale expected to sell {expected_units_sold} units with ${revenue_impact:.2f} revenue impact",
            }

            logger.info(
                "product_id=<%s>, discount=<%f>, expected_units=<%d> | flash sale created",
                product_id,
                final_discount,
                expected_units_sold,
            )

            # End Langfuse span with success outcome
            if span_id:
                self.langfuse_integration.end_agent_span(
                    span_id=span_id,
                    outcome={
                        "flash_sale_id": result["flash_sale_id"],
                        "discount_percentage": result["discount_percentage"],
                        "expected_units_sold": result["expected_units_sold"],
                        "revenue_impact": result["revenue_impact"],
                        "success": True,
                    },
                )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to create flash sale",
                product_data.get("id", "unknown"),
                str(e),
            )

            # End Langfuse span with error outcome
            if span_id:
                self.langfuse_integration.end_agent_span(
                    span_id=span_id,
                    outcome={
                        "success": False,
                        "error": str(e),
                    },
                    error=e,
                )

            return {
                "flash_sale_id": str(uuid4()),
                "product_id": product_data.get("id", "unknown"),
                "success": False,
                "analysis": f"Error creating flash sale: {str(e)}",
            }

    @tool
    def generate_bundle_recommendation(
        self, anchor_product: Dict[str, Any], available_products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create bundle recommendation engine using product affinity analysis.

        Args:
            anchor_product: Main product for the bundle
            available_products: List of products available for bundling

        Returns:
            Dictionary containing bundle recommendation with affinity analysis
        """
        # Start Langfuse span for bundle recommendation
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="generate_bundle_recommendation",
            input_data={
                "anchor_product_id": anchor_product.get("id", "unknown"),
                "available_products_count": len(available_products),
            },
        )

        try:
            logger.debug(
                "anchor_product=<%s>, available_count=<%d> | generating bundle recommendation",
                anchor_product.get("id", "unknown"),
                len(available_products),
            )

            anchor_id = anchor_product.get("id", "unknown")
            anchor_category = anchor_product.get("category", "general")
            anchor_price = anchor_product.get("current_price", 0)

            # Calculate product affinity scores
            complementary_products = []

            for product in available_products:
                if product.get("id") == anchor_id:
                    continue  # Skip the anchor product itself

                # Calculate affinity score based on multiple factors
                affinity_score = self._calculate_product_affinity(
                    anchor_product, product
                )

                if (
                    affinity_score > 0.3
                ):  # Only include products with reasonable affinity
                    complementary_products.append(
                        {
                            "product": product,
                            "affinity_score": affinity_score,
                            "affinity_reasons": self._get_affinity_reasons(
                                anchor_product, product
                            ),
                        }
                    )

            # Sort by affinity score and select top candidates
            complementary_products.sort(key=lambda x: x["affinity_score"], reverse=True)
            top_complementary = complementary_products[
                :4
            ]  # Max 4 complementary products

            if not top_complementary:
                return {
                    "bundle_id": str(uuid4()),
                    "anchor_product_id": anchor_id,
                    "complementary_products": [],
                    "bundle_feasible": False,
                    "analysis": "No suitable complementary products found for bundling",
                }

            # Calculate optimal bundle pricing
            total_individual_price = anchor_price + sum(
                p["product"]["current_price"] for p in top_complementary
            )

            # Bundle discount should be attractive but profitable
            # Higher affinity = higher discount possible
            avg_affinity = sum(p["affinity_score"] for p in top_complementary) / len(
                top_complementary
            )
            bundle_discount_percentage = 10 + (
                avg_affinity * 15
            )  # 10-25% discount based on affinity

            bundle_price = total_individual_price * (
                1 - bundle_discount_percentage / 100
            )
            savings_amount = total_individual_price - bundle_price

            # Calculate expected impact
            bundle_conversion_rate = 0.15 + (
                avg_affinity * 0.1
            )  # 15-25% conversion rate
            expected_bundles_sold = int(
                10 * bundle_conversion_rate
            )  # Assume 10 potential customers

            # Revenue impact calculation
            individual_revenue = (
                anchor_price * 1
            )  # Assume 1 anchor product sold normally
            bundle_revenue = bundle_price * expected_bundles_sold
            revenue_uplift = bundle_revenue - individual_revenue

            # Generate bundle name and description
            bundle_name = self._generate_bundle_name(
                anchor_product, [p["product"] for p in top_complementary]
            )
            bundle_description = self._generate_bundle_description(
                anchor_product, top_complementary
            )

            result = {
                "bundle_id": str(uuid4()),
                "bundle_name": bundle_name,
                "bundle_description": bundle_description,
                "anchor_product_id": anchor_id,
                "anchor_product_name": anchor_product.get("name", "Unknown"),
                "complementary_products": [
                    {
                        "product_id": p["product"]["id"],
                        "product_name": p["product"]["name"],
                        "affinity_score": round(p["affinity_score"], 2),
                        "affinity_reasons": p["affinity_reasons"],
                        "individual_price": p["product"]["current_price"],
                    }
                    for p in top_complementary
                ],
                "total_products": len(top_complementary) + 1,
                "individual_total_price": round(total_individual_price, 2),
                "bundle_price": round(bundle_price, 2),
                "discount_percentage": round(bundle_discount_percentage, 1),
                "savings_amount": round(savings_amount, 2),
                "average_affinity_score": round(avg_affinity, 2),
                "expected_conversion_rate": round(bundle_conversion_rate, 2),
                "expected_bundles_sold": expected_bundles_sold,
                "revenue_uplift": round(revenue_uplift, 2),
                "bundle_feasible": True,
                "recommendation_strength": "high"
                if avg_affinity > 0.7
                else "medium"
                if avg_affinity > 0.5
                else "low",
                "rationale": f"Bundle combines {len(top_complementary) + 1} products with {avg_affinity:.1%} average affinity",
                "analysis": f"Bundle recommendation with {bundle_discount_percentage:.1f}% discount and ${revenue_uplift:.2f} revenue uplift",
            }

            logger.info(
                "anchor_product=<%s>, complementary_count=<%d>, avg_affinity=<%f> | bundle recommendation generated",
                anchor_id,
                len(top_complementary),
                avg_affinity,
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "bundle_id": result["bundle_id"],
                    "bundle_name": result["bundle_name"],
                    "complementary_products_count": len(
                        result["complementary_products"]
                    ),
                    "average_affinity_score": result["average_affinity_score"],
                    "bundle_feasible": result["bundle_feasible"],
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "anchor_product=<%s>, error=<%s> | failed to generate bundle recommendation",
                anchor_product.get("id", "unknown"),
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
                "bundle_id": str(uuid4()),
                "anchor_product_id": anchor_product.get("id", "unknown"),
                "bundle_feasible": False,
                "analysis": f"Error generating bundle recommendation: {str(e)}",
            }

    @tool
    def analyze_social_sentiment(
        self,
        product_category: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        time_period_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Add social sentiment analysis integration for trend detection.

        Args:
            product_category: Product category to analyze sentiment for
            keywords: Specific keywords to track sentiment
            time_period_hours: Time period for sentiment analysis

        Returns:
            Dictionary containing sentiment analysis and promotional opportunities
        """
        # Start Langfuse span for social sentiment analysis
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="analyze_social_sentiment",
            input_data={
                "product_category": product_category,
                "keywords": keywords,
                "time_period_hours": time_period_hours,
            },
        )

        try:
            logger.debug(
                "category=<%s>, keywords=<%s>, period=<%d> | analyzing social sentiment",
                product_category,
                keywords,
                time_period_hours,
            )

            # Simulate social media sentiment analysis
            # In real implementation, this would integrate with social media APIs
            platforms = ["twitter", "instagram", "facebook", "tiktok"]

            platform_sentiments = []
            total_mentions = 0
            sentiment_scores = []

            for platform in platforms:
                # Generate realistic sentiment data
                base_sentiment = (
                    hash(platform + str(time_period_hours)) % 200 - 100
                ) / 100  # -1 to 1
                mention_count = (
                    hash(platform + (product_category or "general")) % 500 + 50
                )
                engagement_rate = (hash(platform + "engagement") % 40 + 20) / 10  # 2-6%

                # Generate trending keywords based on category
                if product_category:
                    category_keywords = self._get_category_keywords(product_category)
                else:
                    category_keywords = [
                        "trending",
                        "popular",
                        "sale",
                        "discount",
                        "new",
                    ]

                if keywords:
                    trending_keywords = keywords + category_keywords[:2]
                else:
                    trending_keywords = category_keywords[:3]

                platform_sentiments.append(
                    {
                        "platform": platform,
                        "sentiment_score": round(base_sentiment, 2),
                        "mention_count": mention_count,
                        "engagement_rate": round(engagement_rate, 1),
                        "trending_keywords": trending_keywords,
                    }
                )

                total_mentions += mention_count
                sentiment_scores.append(base_sentiment)

            # Calculate overall sentiment metrics
            overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)

            # Determine sentiment trend
            if overall_sentiment > 0.3:
                sentiment_trend = "very_positive"
            elif overall_sentiment > 0.1:
                sentiment_trend = "positive"
            elif overall_sentiment > -0.1:
                sentiment_trend = "neutral"
            elif overall_sentiment > -0.3:
                sentiment_trend = "negative"
            else:
                sentiment_trend = "very_negative"

            # Generate promotional opportunities based on sentiment
            promotional_opportunities = []
            risk_factors = []

            if overall_sentiment > 0.2:
                promotional_opportunities.append(
                    "High positive sentiment - ideal for premium product launches"
                )
                promotional_opportunities.append(
                    "Strong engagement - consider influencer partnerships"
                )

            if any("sale" in ps["trending_keywords"] for ps in platform_sentiments):
                promotional_opportunities.append(
                    "Sale keywords trending - launch flash sale campaigns"
                )

            if total_mentions > 1000:
                promotional_opportunities.append(
                    "High mention volume - leverage with targeted ads"
                )

            if overall_sentiment < -0.2:
                risk_factors.append(
                    "Negative sentiment detected - avoid aggressive promotions"
                )
                risk_factors.append(
                    "Consider addressing customer concerns before campaigns"
                )

            if any(
                "complaint" in ps["trending_keywords"] for ps in platform_sentiments
            ):
                risk_factors.append(
                    "Customer complaints trending - focus on service improvements"
                )

            # Default recommendations if none generated
            if not promotional_opportunities:
                promotional_opportunities.append(
                    "Moderate sentiment - standard promotional activities recommended"
                )

            if not risk_factors:
                risk_factors.append("No significant risk factors detected")

            result = {
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "analysis_period_hours": time_period_hours,
                "product_category": product_category,
                "keywords_tracked": keywords or [],
                "overall_sentiment": round(overall_sentiment, 2),
                "sentiment_trend": sentiment_trend,
                "total_mentions": total_mentions,
                "platform_data": platform_sentiments,
                "promotional_opportunities": promotional_opportunities,
                "risk_factors": risk_factors,
                "confidence_score": 0.75
                + (
                    abs(overall_sentiment) * 0.2
                ),  # Higher confidence with stronger sentiment
                "recommendation": self._generate_sentiment_recommendation(
                    overall_sentiment, total_mentions
                ),
                "analysis": f"Social sentiment analysis shows {sentiment_trend} trend with {total_mentions} total mentions",
            }

            logger.info(
                "category=<%s>, sentiment=<%f>, mentions=<%d> | social sentiment analyzed",
                product_category,
                overall_sentiment,
                total_mentions,
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "overall_sentiment": result["overall_sentiment"],
                    "sentiment_trend": result["sentiment_trend"],
                    "total_mentions": result["total_mentions"],
                    "promotional_opportunities_count": len(
                        result["promotional_opportunities"]
                    ),
                    "risk_factors_count": len(result["risk_factors"]),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "category=<%s>, error=<%s> | failed to analyze social sentiment",
                product_category,
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
                "overall_sentiment": 0.0,
                "sentiment_trend": "unknown",
                "analysis": f"Error analyzing social sentiment: {str(e)}",
            }

    @tool
    def schedule_promotional_campaign(
        self, campaign_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Implement promotional campaign scheduling and coordination.

        Args:
            campaign_data: Campaign details including timing, products, and targeting

        Returns:
            Dictionary containing campaign schedule and coordination details
        """
        # Start Langfuse span for promotional campaign scheduling
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="schedule_promotional_campaign",
            input_data={
                "campaign_name": campaign_data.get("name", "unknown"),
                "campaign_type": campaign_data.get("type", "unknown"),
                "product_ids_count": len(campaign_data.get("product_ids", [])),
            },
        )

        try:
            logger.debug(
                "campaign_name=<%s> | scheduling promotional campaign",
                campaign_data.get("name", "unknown"),
            )

            campaign_name = campaign_data.get("name", "Promotional Campaign")
            campaign_type = campaign_data.get("type", "general")
            product_ids = campaign_data.get("product_ids", [])
            start_date = campaign_data.get("start_date")
            end_date = campaign_data.get("end_date")
            budget = campaign_data.get("budget", 1000.0)
            target_audience = campaign_data.get("target_audience", "general")

            # Validate campaign timing
            if start_date and end_date:
                start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

                if end_dt <= start_dt:
                    raise ValueError("Campaign end date must be after start date")

                campaign_duration = (end_dt - start_dt).days
            else:
                # Default to 7-day campaign starting tomorrow
                start_dt = datetime.now(timezone.utc) + timedelta(days=1)
                end_dt = start_dt + timedelta(days=7)
                campaign_duration = 7

            # Generate campaign schedule phases
            phases = self._generate_campaign_phases(start_dt, end_dt, campaign_type)

            # Calculate resource allocation
            daily_budget = (
                budget / campaign_duration if campaign_duration > 0 else budget
            )

            # Coordinate with other agents
            coordination_requirements = []

            if product_ids:
                coordination_requirements.append(
                    {
                        "agent": "inventory_agent",
                        "action": "reserve_inventory",
                        "products": product_ids,
                        "reason": "Ensure sufficient stock for campaign",
                    }
                )

                coordination_requirements.append(
                    {
                        "agent": "pricing_agent",
                        "action": "coordinate_pricing",
                        "products": product_ids,
                        "reason": "Align pricing strategy with campaign",
                    }
                )

            # Generate campaign metrics and KPIs
            expected_metrics = {
                "target_impressions": int(budget * 100),  # $1 = 100 impressions
                "target_clicks": int(budget * 3),  # 3% CTR
                "target_conversions": int(budget * 0.15),  # 5% conversion rate
                "expected_revenue": budget * 3.5,  # 3.5x ROAS target
                "expected_roi": 2.5,
            }

            # Create campaign timeline
            timeline = []
            current_date = start_dt
            while current_date < end_dt:
                timeline.append(
                    {
                        "date": current_date.date().isoformat(),
                        "phase": self._get_campaign_phase(current_date, phases),
                        "daily_budget": round(daily_budget, 2),
                        "activities": self._get_daily_activities(
                            current_date, campaign_type
                        ),
                    }
                )
                current_date += timedelta(days=1)

            result = {
                "campaign_id": str(uuid4()),
                "campaign_name": campaign_name,
                "campaign_type": campaign_type,
                "status": "scheduled",
                "start_date": start_dt.isoformat(),
                "end_date": end_dt.isoformat(),
                "duration_days": campaign_duration,
                "total_budget": budget,
                "daily_budget": round(daily_budget, 2),
                "target_audience": target_audience,
                "product_ids": product_ids,
                "phases": phases,
                "timeline": timeline[:7],  # Show first week
                "coordination_requirements": coordination_requirements,
                "expected_metrics": expected_metrics,
                "success_criteria": {
                    "min_roi": 1.5,
                    "min_conversion_rate": 0.03,
                    "max_cost_per_acquisition": budget
                    / max(1, expected_metrics["target_conversions"]),
                },
                "monitoring_schedule": self._generate_monitoring_schedule(
                    start_dt, end_dt
                ),
                "rationale": f"Scheduled {campaign_duration}-day {campaign_type} campaign with ${budget} budget",
                "analysis": f"Campaign scheduled with {len(phases)} phases and {len(coordination_requirements)} coordination requirements",
            }

            logger.info(
                "campaign_name=<%s>, duration=<%d>, budget=<%f> | promotional campaign scheduled",
                campaign_name,
                campaign_duration,
                budget,
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "campaign_id": result["campaign_id"],
                    "campaign_name": result["campaign_name"],
                    "status": result["status"],
                    "duration_days": result["duration_days"],
                    "total_budget": result["total_budget"],
                    "phases_count": len(result["phases"]),
                    "coordination_requirements_count": len(
                        result["coordination_requirements"]
                    ),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "campaign_name=<%s>, error=<%s> | failed to schedule promotional campaign",
                campaign_data.get("name", "unknown"),
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
                "campaign_id": str(uuid4()),
                "status": "failed",
                "analysis": f"Error scheduling campaign: {str(e)}",
            }

    @tool
    def evaluate_campaign_effectiveness(
        self, campaign_id: str, performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Configure agent memory for campaign effectiveness tracking.

        Args:
            campaign_id: Campaign identifier to evaluate
            performance_data: Actual campaign performance metrics

        Returns:
            Dictionary containing campaign effectiveness evaluation
        """
        # Start Langfuse span for campaign effectiveness evaluation
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="evaluate_campaign_effectiveness",
            input_data={
                "campaign_id": campaign_id,
                "performance_data_keys": list(performance_data.keys()),
            },
        )

        try:
            logger.debug(
                "campaign_id=<%s> | evaluating campaign effectiveness", campaign_id
            )

            # Extract performance metrics
            actual_impressions = performance_data.get("impressions", 0)
            actual_clicks = performance_data.get("clicks", 0)
            actual_conversions = performance_data.get("conversions", 0)
            actual_revenue = performance_data.get("revenue", 0)
            actual_cost = performance_data.get("cost", 0)

            # Calculate performance ratios
            ctr = (
                (actual_clicks / actual_impressions * 100)
                if actual_impressions > 0
                else 0
            )
            conversion_rate = (
                (actual_conversions / actual_clicks * 100) if actual_clicks > 0 else 0
            )
            cpa = (
                actual_cost / actual_conversions
                if actual_conversions > 0
                else float("inf")
            )
            roi = (actual_revenue - actual_cost) / actual_cost if actual_cost > 0 else 0
            roas = actual_revenue / actual_cost if actual_cost > 0 else 0

            # Retrieve expected metrics from campaign memory
            # In real implementation, this would fetch from campaign database
            expected_metrics = {
                "target_impressions": 10000,
                "target_clicks": 300,
                "target_conversions": 15,
                "expected_revenue": 3500,
                "expected_roi": 2.5,
            }

            # Calculate performance vs expectations
            performance_ratios = {
                "impressions_ratio": actual_impressions
                / expected_metrics["target_impressions"]
                if expected_metrics["target_impressions"] > 0
                else 0,
                "clicks_ratio": actual_clicks / expected_metrics["target_clicks"]
                if expected_metrics["target_clicks"] > 0
                else 0,
                "conversions_ratio": actual_conversions
                / expected_metrics["target_conversions"]
                if expected_metrics["target_conversions"] > 0
                else 0,
                "revenue_ratio": actual_revenue / expected_metrics["expected_revenue"]
                if expected_metrics["expected_revenue"] > 0
                else 0,
                "roi_ratio": roi / expected_metrics["expected_roi"]
                if expected_metrics["expected_roi"] > 0
                else 0,
            }

            # Calculate overall effectiveness score
            effectiveness_score = (
                performance_ratios["impressions_ratio"] * 0.1
                + performance_ratios["clicks_ratio"] * 0.2
                + performance_ratios["conversions_ratio"] * 0.3
                + performance_ratios["revenue_ratio"] * 0.3
                + performance_ratios["roi_ratio"] * 0.1
            )

            # Determine campaign success level
            if effectiveness_score >= 1.2:
                success_level = "excellent"
            elif effectiveness_score >= 1.0:
                success_level = "successful"
            elif effectiveness_score >= 0.8:
                success_level = "moderate"
            elif effectiveness_score >= 0.6:
                success_level = "poor"
            else:
                success_level = "failed"

            # Generate insights and recommendations
            insights = []
            recommendations = []

            if ctr < 2.0:
                insights.append(
                    "Low click-through rate indicates poor ad creative or targeting"
                )
                recommendations.append(
                    "Improve ad creative and refine audience targeting"
                )

            if conversion_rate < 3.0:
                insights.append(
                    "Low conversion rate suggests landing page or offer issues"
                )
                recommendations.append(
                    "Optimize landing page and strengthen value proposition"
                )

            if roi < 1.0:
                insights.append("Negative ROI indicates campaign is not profitable")
                recommendations.append(
                    "Reduce costs or improve conversion optimization"
                )

            if performance_ratios["impressions_ratio"] < 0.8:
                insights.append(
                    "Low impression delivery may indicate budget or targeting constraints"
                )
                recommendations.append("Increase budget or expand targeting parameters")

            # Store effectiveness data in agent memory
            effectiveness_data = {
                "campaign_id": campaign_id,
                "effectiveness_score": effectiveness_score,
                "success_level": success_level,
                "performance_metrics": {
                    "ctr": ctr,
                    "conversion_rate": conversion_rate,
                    "cpa": cpa,
                    "roi": roi,
                    "roas": roas,
                },
                "performance_ratios": performance_ratios,
                "insights": insights,
                "recommendations": recommendations,
            }

            # In real implementation: store in agent memory for future learning
            # agent_memory.store_campaign_effectiveness(self.agent_id, effectiveness_data)

            result = {
                "campaign_id": campaign_id,
                "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
                "effectiveness_score": round(effectiveness_score, 2),
                "success_level": success_level,
                "performance_metrics": {
                    "impressions": actual_impressions,
                    "clicks": actual_clicks,
                    "conversions": actual_conversions,
                    "revenue": round(actual_revenue, 2),
                    "cost": round(actual_cost, 2),
                    "ctr": round(ctr, 2),
                    "conversion_rate": round(conversion_rate, 2),
                    "cpa": round(cpa, 2) if cpa != float("inf") else "N/A",
                    "roi": round(roi, 2),
                    "roas": round(roas, 2),
                },
                "performance_vs_expected": performance_ratios,
                "insights": insights,
                "recommendations": recommendations,
                "learning_points": self._generate_learning_points(effectiveness_data),
                "future_optimization": self._suggest_future_optimizations(
                    effectiveness_data
                ),
                "analysis": f"Campaign achieved {success_level} performance with {effectiveness_score:.1f} effectiveness score",
            }

            logger.info(
                "campaign_id=<%s>, effectiveness=<%f>, success=<%s> | campaign effectiveness evaluated",
                campaign_id,
                effectiveness_score,
                success_level,
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "campaign_id": result["campaign_id"],
                    "effectiveness_score": result["effectiveness_score"],
                    "success_level": result["success_level"],
                    "insights_count": len(result["insights"]),
                    "recommendations_count": len(result["recommendations"]),
                    "success": True,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "campaign_id=<%s>, error=<%s> | failed to evaluate campaign effectiveness",
                campaign_id,
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
                "campaign_id": campaign_id,
                "effectiveness_score": 0.0,
                "success_level": "evaluation_failed",
                "analysis": f"Error evaluating campaign effectiveness: {str(e)}",
            }

    @tool
    def coordinate_with_pricing_agent(
        self, coordination_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Coordinate promotional activities with the pricing agent.

        Args:
            coordination_request: Details of coordination needed with pricing agent

        Returns:
            Dictionary containing coordination results and recommendations
        """
        try:
            logger.debug(
                "request_type=<%s> | coordinating with pricing agent",
                coordination_request.get("type", "unknown"),
            )

            request_type = coordination_request.get("type", "general")
            product_ids = coordination_request.get("product_ids", [])
            campaign_details = coordination_request.get("campaign_details", {})

            # Simulate coordination with pricing agent
            coordination_results = []

            for product_id in product_ids:
                if request_type == "flash_sale_pricing":
                    # Request optimal flash sale pricing
                    coordination_results.append(
                        {
                            "product_id": product_id,
                            "coordination_type": "flash_sale_pricing",
                            "pricing_recommendation": {
                                "suggested_discount": 20.0,
                                "minimum_price": 15.99,
                                "expected_demand_increase": 150.0,
                                "profit_impact": -50.0,
                            },
                            "pricing_agent_confidence": 0.82,
                            "coordination_status": "approved",
                        }
                    )

                elif request_type == "bundle_pricing":
                    # Request bundle pricing coordination
                    coordination_results.append(
                        {
                            "product_id": product_id,
                            "coordination_type": "bundle_pricing",
                            "pricing_recommendation": {
                                "individual_price": 24.99,
                                "bundle_contribution": 22.49,
                                "margin_impact": "acceptable",
                                "cross_sell_potential": "high",
                            },
                            "pricing_agent_confidence": 0.78,
                            "coordination_status": "approved",
                        }
                    )

                elif request_type == "campaign_pricing":
                    # Request campaign-wide pricing coordination
                    coordination_results.append(
                        {
                            "product_id": product_id,
                            "coordination_type": "campaign_pricing",
                            "pricing_recommendation": {
                                "campaign_price": 21.99,
                                "duration_limit": "7_days",
                                "inventory_consideration": "medium_stock",
                                "competitive_position": "competitive",
                            },
                            "pricing_agent_confidence": 0.85,
                            "coordination_status": "approved",
                        }
                    )

            # Calculate overall coordination success
            approved_count = sum(
                1
                for r in coordination_results
                if r["coordination_status"] == "approved"
            )
            coordination_success_rate = (
                approved_count / len(coordination_results)
                if coordination_results
                else 0
            )

            # Generate coordination summary
            if coordination_success_rate >= 0.8:
                coordination_status = "successful"
                coordination_message = "Strong alignment with pricing strategy achieved"
            elif coordination_success_rate >= 0.6:
                coordination_status = "partial"
                coordination_message = "Some pricing conflicts require resolution"
            else:
                coordination_status = "failed"
                coordination_message = "Significant pricing strategy conflicts detected"

            result = {
                "coordination_id": str(uuid4()),
                "coordination_timestamp": datetime.now(timezone.utc).isoformat(),
                "request_type": request_type,
                "products_coordinated": len(product_ids),
                "coordination_results": coordination_results,
                "overall_status": coordination_status,
                "success_rate": round(coordination_success_rate, 2),
                "coordination_message": coordination_message,
                "next_steps": self._generate_coordination_next_steps(
                    coordination_results
                ),
                "follow_up_required": coordination_success_rate < 1.0,
                "analysis": f"Coordinated pricing for {len(product_ids)} products with {coordination_success_rate:.1%} success rate",
            }

            logger.info(
                "request_type=<%s>, products=<%d>, success_rate=<%f> | pricing coordination completed",
                request_type,
                len(product_ids),
                coordination_success_rate,
            )

            return result

        except Exception as e:
            logger.error(
                "request_type=<%s>, error=<%s> | failed to coordinate with pricing agent",
                coordination_request.get("type", "unknown"),
                str(e),
            )
            return {
                "coordination_id": str(uuid4()),
                "overall_status": "failed",
                "analysis": f"Error coordinating with pricing agent: {str(e)}",
            }

    @tool
    def validate_inventory_availability(
        self, product_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate inventory availability for promotional campaigns.

        Args:
            product_requirements: Product inventory requirements for campaigns

        Returns:
            Dictionary containing inventory validation results
        """
        try:
            logger.debug(
                "products=<%d> | validating inventory availability",
                len(product_requirements.get("products", [])),
            )

            products = product_requirements.get("products", [])
            campaign_duration = product_requirements.get("campaign_duration_days", 7)
            expected_demand_multiplier = product_requirements.get(
                "demand_multiplier", 2.0
            )

            validation_results = []
            overall_availability = True

            for product in products:
                product_id = product.get("id", "unknown")
                current_inventory = product.get("inventory_level", 0)
                normal_daily_demand = product.get("daily_demand", 5)
                reorder_point = product.get("reorder_point", 25)

                # Calculate expected demand during campaign
                campaign_daily_demand = normal_daily_demand * expected_demand_multiplier
                total_campaign_demand = campaign_daily_demand * campaign_duration

                # Calculate inventory sufficiency
                available_for_campaign = max(0, current_inventory - reorder_point)
                inventory_sufficiency = (
                    available_for_campaign / total_campaign_demand
                    if total_campaign_demand > 0
                    else 1.0
                )

                # Determine availability status
                if inventory_sufficiency >= 1.0:
                    availability_status = "sufficient"
                elif inventory_sufficiency >= 0.7:
                    availability_status = "adequate"
                elif inventory_sufficiency >= 0.5:
                    availability_status = "limited"
                else:
                    availability_status = "insufficient"
                    overall_availability = False

                # Generate recommendations
                recommendations = []
                if availability_status == "insufficient":
                    recommendations.append("Emergency restock required before campaign")
                    recommendations.append(
                        "Consider reducing campaign scope or duration"
                    )
                elif availability_status == "limited":
                    recommendations.append("Monitor inventory closely during campaign")
                    recommendations.append(
                        "Prepare backup inventory or alternative products"
                    )

                validation_results.append(
                    {
                        "product_id": product_id,
                        "product_name": product.get("name", "Unknown"),
                        "current_inventory": current_inventory,
                        "available_for_campaign": available_for_campaign,
                        "expected_campaign_demand": int(total_campaign_demand),
                        "inventory_sufficiency": round(inventory_sufficiency, 2),
                        "availability_status": availability_status,
                        "days_of_supply": round(
                            available_for_campaign / campaign_daily_demand, 1
                        )
                        if campaign_daily_demand > 0
                        else float("inf"),
                        "recommendations": recommendations,
                        "risk_level": "high"
                        if availability_status == "insufficient"
                        else "medium"
                        if availability_status == "limited"
                        else "low",
                    }
                )

            # Generate overall assessment
            insufficient_count = sum(
                1
                for r in validation_results
                if r["availability_status"] == "insufficient"
            )
            limited_count = sum(
                1 for r in validation_results if r["availability_status"] == "limited"
            )

            if insufficient_count > 0:
                overall_assessment = "high_risk"
                assessment_message = (
                    f"{insufficient_count} products have insufficient inventory"
                )
            elif limited_count > 0:
                overall_assessment = "medium_risk"
                assessment_message = f"{limited_count} products have limited inventory"
            else:
                overall_assessment = "low_risk"
                assessment_message = "All products have sufficient inventory"

            result = {
                "validation_id": str(uuid4()),
                "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                "campaign_duration_days": campaign_duration,
                "demand_multiplier": expected_demand_multiplier,
                "products_validated": len(products),
                "overall_availability": overall_availability,
                "overall_assessment": overall_assessment,
                "assessment_message": assessment_message,
                "validation_results": validation_results,
                "summary_stats": {
                    "sufficient_products": sum(
                        1
                        for r in validation_results
                        if r["availability_status"] == "sufficient"
                    ),
                    "adequate_products": sum(
                        1
                        for r in validation_results
                        if r["availability_status"] == "adequate"
                    ),
                    "limited_products": limited_count,
                    "insufficient_products": insufficient_count,
                },
                "campaign_recommendations": self._generate_inventory_campaign_recommendations(
                    validation_results
                ),
                "monitoring_requirements": self._generate_inventory_monitoring_requirements(
                    validation_results
                ),
                "analysis": f"Validated inventory for {len(products)} products with {overall_assessment} assessment",
            }

            logger.info(
                "products=<%d>, overall_availability=<%s>, assessment=<%s> | inventory availability validated",
                len(products),
                overall_availability,
                overall_assessment,
            )

            return result

        except Exception as e:
            logger.error(
                "products=<%d>, error=<%s> | failed to validate inventory availability",
                len(product_requirements.get("products", [])),
                str(e),
            )
            return {
                "validation_id": str(uuid4()),
                "overall_availability": False,
                "overall_assessment": "validation_failed",
                "analysis": f"Error validating inventory availability: {str(e)}",
            }

    @tool
    def retrieve_promotion_history(
        self,
        product_id: Optional[str] = None,
        campaign_type: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Retrieve historical promotional decisions and outcomes from agent memory.

        Args:
            product_id: Specific product to retrieve history for
            campaign_type: Type of campaigns to retrieve
            days: Number of days of history to retrieve

        Returns:
            Dictionary containing promotional history and insights
        """
        try:
            logger.debug(
                "product_id=<%s>, campaign_type=<%s>, days=<%d> | retrieving promotion history",
                product_id,
                campaign_type,
                days,
            )

            # Retrieve similar promotional decisions from memory
            context = {}
            if product_id:
                context["product_id"] = product_id
            if campaign_type:
                context["campaign_type"] = campaign_type

            # Simulate retrieval from agent memory
            similar_decisions = []
            decision_history = []

            # Generate simulated historical data
            for i in range(5):  # Simulate 5 historical campaigns
                campaign_id = str(uuid4())
                campaign_date = datetime.now(timezone.utc) - timedelta(
                    days=random.randint(1, days)
                )

                decision_data = {
                    "decision": {
                        "id": str(uuid4()),
                        "timestamp": campaign_date.isoformat(),
                        "campaign_id": campaign_id,
                        "campaign_type": campaign_type
                        or random.choice(["flash_sale", "bundle", "seasonal"]),
                        "product_id": product_id or f"PROD_{i:03d}",
                        "parameters": {
                            "discount_percentage": random.uniform(10, 40),
                            "duration_hours": random.randint(12, 168),
                            "budget": random.uniform(500, 5000),
                        },
                    },
                    "outcome": {
                        "success": random.choice(
                            [True, True, True, False]
                        ),  # 75% success rate
                        "revenue_generated": random.uniform(1000, 8000),
                        "roi": random.uniform(0.5, 4.0),
                        "conversion_rate": random.uniform(0.02, 0.08),
                    },
                }

                similarity_score = random.uniform(0.6, 0.95)
                similar_decisions.append((decision_data, similarity_score))
                decision_history.append(decision_data)

            # Analyze promotional patterns
            successful_campaigns = [
                d for d, _ in similar_decisions if d["outcome"]["success"]
            ]
            success_rate = (
                len(successful_campaigns) / len(similar_decisions)
                if similar_decisions
                else 0
            )

            # Calculate average performance metrics
            if successful_campaigns:
                avg_roi = sum(d["outcome"]["roi"] for d in successful_campaigns) / len(
                    successful_campaigns
                )
                avg_conversion_rate = sum(
                    d["outcome"]["conversion_rate"] for d in successful_campaigns
                ) / len(successful_campaigns)
                avg_discount = sum(
                    d["decision"]["parameters"]["discount_percentage"]
                    for d in successful_campaigns
                ) / len(successful_campaigns)
            else:
                avg_roi = 0
                avg_conversion_rate = 0
                avg_discount = 0

            # Generate insights
            insights = []
            if success_rate > 0.8:
                insights.append(
                    "High success rate indicates effective promotional strategy"
                )
            elif success_rate < 0.5:
                insights.append(
                    "Low success rate suggests need for strategy refinement"
                )

            if avg_roi > 2.0:
                insights.append("Strong ROI performance across campaigns")
            elif avg_roi < 1.0:
                insights.append("ROI below break-even - review cost structure")

            if avg_discount > 30:
                insights.append("High discount rates may be eroding margins")
            elif avg_discount < 15:
                insights.append("Conservative discounting may limit campaign impact")

            result = {
                "retrieval_timestamp": datetime.now(timezone.utc).isoformat(),
                "product_id": product_id,
                "campaign_type": campaign_type,
                "analysis_period_days": days,
                "total_campaigns": len(decision_history),
                "similar_campaigns": len(similar_decisions),
                "success_rate": round(success_rate, 2),
                "performance_metrics": {
                    "average_roi": round(avg_roi, 2),
                    "average_conversion_rate": round(avg_conversion_rate, 3),
                    "average_discount_percentage": round(avg_discount, 1),
                },
                "campaign_history": [
                    {
                        "campaign_id": d["decision"]["campaign_id"],
                        "campaign_type": d["decision"]["campaign_type"],
                        "date": d["decision"]["timestamp"],
                        "success": d["outcome"]["success"],
                        "roi": round(d["outcome"]["roi"], 2),
                        "discount_percentage": round(
                            d["decision"]["parameters"]["discount_percentage"], 1
                        ),
                    }
                    for d, _ in similar_decisions[:5]  # Show top 5
                ],
                "insights": insights,
                "learning_recommendations": self._generate_promotion_learning_recommendations(
                    success_rate, avg_roi, avg_discount
                ),
                "best_practices": self._extract_best_practices(successful_campaigns),
                "analysis": f"Retrieved {len(decision_history)} promotional campaigns with {success_rate:.1%} success rate",
            }

            logger.info(
                "product_id=<%s>, campaigns=<%d>, success_rate=<%f> | promotion history retrieved",
                product_id,
                len(decision_history),
                success_rate,
            )

            return result

        except Exception as e:
            logger.error(
                "product_id=<%s>, error=<%s> | failed to retrieve promotion history",
                product_id,
                str(e),
            )
            return {
                "total_campaigns": 0,
                "success_rate": 0.0,
                "analysis": f"Error retrieving promotion history: {str(e)}",
            }

    # Helper methods for internal calculations
    def _calculate_product_affinity(
        self, anchor_product: Dict[str, Any], candidate_product: Dict[str, Any]
    ) -> float:
        """Calculate affinity score between two products."""
        affinity_score = 0.0

        # Category affinity (same category = higher affinity)
        if anchor_product.get("category") == candidate_product.get("category"):
            affinity_score += 0.4

        # Price range affinity (similar price ranges work well together)
        anchor_price = anchor_product.get("current_price", 0)
        candidate_price = candidate_product.get("current_price", 0)
        if anchor_price > 0 and candidate_price > 0:
            price_ratio = min(anchor_price, candidate_price) / max(
                anchor_price, candidate_price
            )
            affinity_score += price_ratio * 0.3

        # Complementary product logic (simplified)
        anchor_name = anchor_product.get("name", "").lower()
        candidate_name = candidate_product.get("name", "").lower()

        complementary_pairs = [
            ("coffee", "cream"),
            ("coffee", "sugar"),
            ("coffee", "mug"),
            ("shirt", "pants"),
            ("laptop", "mouse"),
            ("phone", "case"),
        ]

        for pair in complementary_pairs:
            if (pair[0] in anchor_name and pair[1] in candidate_name) or (
                pair[1] in anchor_name and pair[0] in candidate_name
            ):
                affinity_score += 0.3
                break

        return min(1.0, affinity_score)

    def _get_affinity_reasons(
        self, anchor_product: Dict[str, Any], candidate_product: Dict[str, Any]
    ) -> List[str]:
        """Get reasons for product affinity."""
        reasons = []

        if anchor_product.get("category") == candidate_product.get("category"):
            reasons.append("Same product category")

        anchor_price = anchor_product.get("current_price", 0)
        candidate_price = candidate_product.get("current_price", 0)
        if anchor_price > 0 and candidate_price > 0:
            price_ratio = min(anchor_price, candidate_price) / max(
                anchor_price, candidate_price
            )
            if price_ratio > 0.7:
                reasons.append("Similar price range")

        return reasons or ["General product compatibility"]

    def _generate_bundle_name(
        self,
        anchor_product: Dict[str, Any],
        complementary_products: List[Dict[str, Any]],
    ) -> str:
        """Generate an attractive bundle name."""
        anchor_name = anchor_product.get("name", "Product")
        category = anchor_product.get("category", "Essential")

        if len(complementary_products) == 1:
            return f"{anchor_name} + {complementary_products[0].get('name', 'Accessory')} Bundle"
        else:
            return f"{category} Complete Bundle"

    def _generate_bundle_description(
        self,
        anchor_product: Dict[str, Any],
        complementary_products: List[Dict[str, Any]],
    ) -> str:
        """Generate bundle description."""
        product_count = len(complementary_products) + 1
        return f"Complete {product_count}-piece bundle featuring {anchor_product.get('name', 'premium product')} and carefully selected complementary items for maximum value."

    def _get_category_keywords(self, category: str) -> List[str]:
        """Get trending keywords for healthcare/wellness product categories."""
        category_keywords = {
            "vitamins": [
                "immunity",
                "boost",
                "health",
                "wellness",
                "natural",
                "essential",
            ],
            "supplements": [
                "nutrition",
                "support",
                "daily",
                "premium",
                "pure",
                "effective",
            ],
            "fitness": [
                "workout",
                "exercise",
                "strength",
                "flexibility",
                "active",
                "goals",
            ],
            "wellness": [
                "mindfulness",
                "balance",
                "calm",
                "stress-relief",
                "natural",
                "holistic",
            ],
            "essential_oils": [
                "aromatherapy",
                "pure",
                "therapeutic",
                "relaxation",
                "natural",
                "organic",
            ],
            "probiotics": [
                "digestive",
                "gut health",
                "microbiome",
                "daily",
                "support",
                "balance",
            ],
            "immune_support": [
                "immunity",
                "defense",
                "protection",
                "seasonal",
                "boost",
                "strengthen",
            ],
            "stress_relief": [
                "calm",
                "relaxation",
                "peace",
                "mindfulness",
                "balance",
                "zen",
            ],
            "energy": [
                "vitality",
                "focus",
                "performance",
                "endurance",
                "natural",
                "sustained",
            ],
            "sleep": [
                "rest",
                "recovery",
                "peaceful",
                "natural",
                "deep sleep",
                "relaxation",
            ],
        }
        return category_keywords.get(
            category.lower(),
            ["health", "wellness", "natural", "quality", "effective", "trusted"],
        )

    def _generate_sentiment_recommendation(
        self, sentiment_score: float, mention_count: int
    ) -> str:
        """Generate healthcare/wellness-specific recommendation based on sentiment analysis."""
        if sentiment_score > 0.4 and mention_count > 800:
            return "Excellent health/wellness sentiment - ideal for premium wellness campaigns and influencer partnerships"
        elif sentiment_score > 0.2 and mention_count > 400:
            return "Strong positive health sentiment - good opportunity for seasonal wellness promotions and bundles"
        elif sentiment_score > 0.0:
            return "Moderate positive sentiment - suitable for educational wellness content and gentle promotional activities"
        elif sentiment_score < -0.3:
            return "Negative health sentiment detected - focus on customer education and compliance before promotional campaigns"
        elif sentiment_score < -0.1:
            return "Slight negative sentiment - emphasize safety, quality, and regulatory compliance in messaging"
        else:
            return "Neutral health sentiment - standard wellness promotional activities with educational focus recommended"

    def _generate_campaign_phases(
        self, start_date: datetime, end_date: datetime, campaign_type: str
    ) -> List[Dict[str, Any]]:
        """Generate campaign phases based on type and duration."""
        duration = (end_date - start_date).days

        if campaign_type == "flash_sale":
            return [
                {
                    "name": "Launch",
                    "start": start_date,
                    "duration_hours": 6,
                    "focus": "Maximum visibility",
                },
                {
                    "name": "Peak",
                    "start": start_date + timedelta(hours=6),
                    "duration_hours": 12,
                    "focus": "Conversion optimization",
                },
                {
                    "name": "Final Push",
                    "start": start_date + timedelta(hours=18),
                    "duration_hours": 6,
                    "focus": "Urgency messaging",
                },
            ]
        else:
            phases = []
            phase_duration = max(1, duration // 3)
            phases.append(
                {
                    "name": "Launch",
                    "start": start_date,
                    "duration_days": phase_duration,
                    "focus": "Awareness building",
                }
            )
            phases.append(
                {
                    "name": "Optimization",
                    "start": start_date + timedelta(days=phase_duration),
                    "duration_days": phase_duration,
                    "focus": "Performance optimization",
                }
            )
            phases.append(
                {
                    "name": "Final Push",
                    "start": start_date + timedelta(days=phase_duration * 2),
                    "duration_days": duration - phase_duration * 2,
                    "focus": "Conversion maximization",
                }
            )
            return phases

    def _get_campaign_phase(
        self, current_date: datetime, phases: List[Dict[str, Any]]
    ) -> str:
        """Determine which campaign phase a date falls into."""
        for phase in phases:
            phase_start = phase["start"]
            if "duration_hours" in phase:
                phase_end = phase_start + timedelta(hours=phase["duration_hours"])
            else:
                phase_end = phase_start + timedelta(days=phase["duration_days"])

            if phase_start <= current_date < phase_end:
                return phase["name"]

        return "Unknown"

    def _get_daily_activities(self, date: datetime, campaign_type: str) -> List[str]:
        """Get daily activities for a campaign."""
        activities = ["Monitor performance metrics", "Adjust targeting if needed"]

        if campaign_type == "flash_sale":
            activities.extend(["Send urgency notifications", "Update inventory levels"])
        elif campaign_type == "bundle":
            activities.extend(
                ["Track bundle conversion rates", "Optimize product recommendations"]
            )

        return activities

    def _generate_monitoring_schedule(
        self, start_date: datetime, end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Generate monitoring schedule for campaign."""
        schedule = []
        current = start_date

        while current < end_date:
            schedule.append(
                {
                    "date": current.date().isoformat(),
                    "frequency": "hourly"
                    if (current - start_date).days < 1
                    else "daily",
                    "metrics": ["impressions", "clicks", "conversions", "cost"],
                }
            )
            current += timedelta(days=1)

        return schedule[:7]  # Return first week

    def _generate_learning_points(
        self, effectiveness_data: Dict[str, Any]
    ) -> List[str]:
        """Generate learning points from campaign effectiveness."""
        learning_points = []

        effectiveness_score = effectiveness_data.get("effectiveness_score", 0)
        if effectiveness_score > 1.0:
            learning_points.append(
                "Campaign exceeded expectations - replicate successful elements"
            )
        else:
            learning_points.append("Campaign underperformed - analyze failure points")

        return learning_points

    def _suggest_future_optimizations(
        self, effectiveness_data: Dict[str, Any]
    ) -> List[str]:
        """Suggest optimizations for future campaigns."""
        optimizations = []

        performance_metrics = effectiveness_data.get("performance_metrics", {})
        ctr = performance_metrics.get("ctr", 0)
        conversion_rate = performance_metrics.get("conversion_rate", 0)

        if ctr < 2.0:
            optimizations.append("Improve ad creative and targeting")
        if conversion_rate < 3.0:
            optimizations.append("Optimize landing page experience")

        return optimizations or ["Continue current strategy"]

    def _generate_coordination_next_steps(
        self, coordination_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate next steps for agent coordination."""
        next_steps = []

        failed_coordinations = [
            r for r in coordination_results if r["coordination_status"] != "approved"
        ]
        if failed_coordinations:
            next_steps.append("Resolve pricing conflicts with failed coordinations")

        next_steps.append("Monitor coordinated pricing during campaign execution")
        return next_steps

    def _generate_inventory_campaign_recommendations(
        self, validation_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate campaign recommendations based on inventory validation."""
        recommendations = []

        insufficient_products = [
            r for r in validation_results if r["availability_status"] == "insufficient"
        ]
        if insufficient_products:
            recommendations.append("Delay campaign launch until inventory is restocked")

        limited_products = [
            r for r in validation_results if r["availability_status"] == "limited"
        ]
        if limited_products:
            recommendations.append("Implement inventory monitoring during campaign")

        return recommendations or ["Proceed with campaign as planned"]

    def _generate_inventory_monitoring_requirements(
        self, validation_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate inventory monitoring requirements."""
        requirements = []

        for result in validation_results:
            if result["risk_level"] in ["high", "medium"]:
                requirements.append(
                    {
                        "product_id": result["product_id"],
                        "monitoring_frequency": "hourly"
                        if result["risk_level"] == "high"
                        else "daily",
                        "alert_threshold": result["available_for_campaign"] * 0.2,
                    }
                )

        return requirements

    def _generate_promotion_learning_recommendations(
        self, success_rate: float, avg_roi: float, avg_discount: float
    ) -> List[str]:
        """Generate learning recommendations from promotion history."""
        recommendations = []

        if success_rate < 0.6:
            recommendations.append("Review targeting and timing strategies")
        if avg_roi < 1.5:
            recommendations.append(
                "Focus on cost optimization and conversion improvement"
            )
        if avg_discount > 35:
            recommendations.append("Test lower discount rates to preserve margins")

        return recommendations or ["Continue current promotional strategy"]

    def _extract_best_practices(
        self, successful_campaigns: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract best practices from successful campaigns."""
        practices = []

        if successful_campaigns:
            avg_discount = sum(
                c["decision"]["parameters"]["discount_percentage"]
                for c in successful_campaigns
            ) / len(successful_campaigns)
            practices.append(
                f"Optimal discount range appears to be around {avg_discount:.0f}%"
            )

            common_types = {}
            for campaign in successful_campaigns:
                campaign_type = campaign["decision"]["campaign_type"]
                common_types[campaign_type] = common_types.get(campaign_type, 0) + 1

            if common_types:
                best_type = max(common_types, key=lambda x: common_types[x])
                practices.append(f"{best_type} campaigns show highest success rate")

        return practices or ["Insufficient data for best practice extraction"]

    def make_promotional_decision(
        self, campaign_request: Dict[str, Any], market_context: Dict[str, Any]
    ) -> AgentDecision:
        """
        Make a comprehensive promotional decision using all available tools and analysis.

        Args:
            campaign_request: Campaign requirements and parameters
            market_context: Market conditions and external factors

        Returns:
            AgentDecision object with the promotional recommendation
        """
        # Start Langfuse span for promotional decision making
        span_id = self.langfuse_integration.start_agent_span(
            agent_id=self.agent_id,
            operation="make_promotional_decision",
            input_data={
                "campaign_type": campaign_request.get("type", "unknown"),
                "product_ids": campaign_request.get("product_ids", []),
                "budget": campaign_request.get("budget", 0),
            },
        )

        try:
            logger.info(
                "campaign_type=<%s> | making promotional decision",
                campaign_request.get("type", "unknown"),
            )

            # Construct healthcare/wellness-specific prompt for the agent
            product_categories = campaign_request.get("product_categories", ["general"])
            health_focus = campaign_request.get("health_focus", "general_wellness")
            current_month = datetime.now().month

            prompt = f"""
            Analyze the following healthcare/wellness campaign request and market data to make an optimal promotional decision:
            
            Campaign Request:
            - Type: {campaign_request.get("type", "general")}
            - Products: {campaign_request.get("product_ids", [])}
            - Product Categories: {product_categories} (Healthcare/Wellness)
            - Health Focus: {health_focus}
            - Budget: ${campaign_request.get("budget", 0):,.2f}
            - Duration: {campaign_request.get("duration_days", 7)} days
            - Target Audience: {campaign_request.get("target_audience", "general")}
            - Current Month: {current_month}
            
            Market Context:
            - Social Sentiment: {market_context.get("social_sentiment", "neutral")}
            - Health Trends: {market_context.get("health_trends", "stable")}
            - Competitor Activity: {market_context.get("competitor_activity", "normal")}
            - Seasonal Factor: {market_context.get("seasonal_factor", 1.0)}
            - Inventory Levels: {market_context.get("inventory_status", "normal")}
            - Season: {market_context.get("current_season", "unknown")}
            
            Healthcare/Wellness Promotional Considerations:
            - Winter months (Oct-Mar) are peak for immune support campaigns (vitamins, supplements)
            - New Year period (Dec-Feb) is optimal for fitness and wellness campaigns
            - Health awareness months: Heart Health (Feb), Mental Health (May), Wellness (Sept)
            - Healthcare products require conservative discounts (15-25%) due to regulatory considerations
            - Wellness/fitness products can have more aggressive promotions (20-35%)
            - Bundle opportunities: Vitamin C+Zinc+D, Yoga mat+bands+blocks, Essential oils+diffuser
            - Regulatory compliance: Avoid medical claims, focus on wellness and lifestyle benefits
            - Health influencer partnerships: Fitness trainers, wellness coaches, nutrition experts
            
            Please use the available tools to:
            1. Analyze social sentiment for health and wellness promotional opportunities
            2. Create appropriate healthcare/wellness campaigns (immune support, fitness, stress relief)
            3. Coordinate with pricing agent on compliance-aware promotional pricing
            4. Schedule campaigns aligned with seasonal health patterns and awareness months
            5. Set up effectiveness tracking with healthcare/wellness-specific KPIs
            6. Review historical performance for similar healthcare/wellness campaigns
            
            Provide a comprehensive healthcare/wellness promotional strategy with regulatory compliance and clear rationale.
            """

            # Use the Strands agent to process the request
            response = self.agent(prompt)

            # Extract decision parameters from the agent's response
            decision_params = {
                "campaign_type": campaign_request.get("type"),
                "product_ids": campaign_request.get("product_ids", []),
                "budget": campaign_request.get("budget", 0),
                "analysis_performed": True,
                "tools_used": [
                    "sentiment_analysis",
                    "campaign_creation",
                    "coordination",
                    "scheduling",
                ],
                "recommendation_source": "strands_agent_analysis",
            }

            # Create agent decision record
            decision = AgentDecision(
                agent_id=self.agent_id,
                action_type=ActionType.PROMOTION_CREATION,
                parameters=decision_params,
                rationale=f"Comprehensive promotional analysis using Strands agent with multiple tools: {response[:200]}...",
                confidence_score=0.88,
                expected_outcome={
                    "analysis_type": "comprehensive",
                    "tools_utilized": len(decision_params.get("tools_used", [])),
                    "agent_response_length": len(response),
                    "campaign_type": campaign_request.get("type"),
                },
                context={
                    "campaign_request": campaign_request,
                    "market_context": market_context,
                    "agent_response": response,
                },
            )

            # Store decision in memory
            memory_id = agent_memory.store_decision(
                agent_id=self.agent_id,
                decision=decision,
                context={
                    "campaign_request": campaign_request,
                    "market_context": market_context,
                },
            )

            logger.info(
                "campaign_type=<%s>, decision_id=<%s>, memory_id=<%s> | promotional decision completed",
                campaign_request.get("type", "unknown"),
                str(decision.id),
                memory_id,
            )

            # End Langfuse span with success outcome
            self.langfuse_integration.end_agent_span(
                span_id=span_id,
                outcome={
                    "decision_id": str(decision.id),
                    "action_type": decision.action_type.value,
                    "confidence_score": decision.confidence_score,
                    "success": True,
                },
            )

            return decision

        except Exception as e:
            logger.error(
                "campaign_type=<%s>, error=<%s> | failed to make promotional decision",
                campaign_request.get("type", "unknown"),
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

            # Return fallback decision
            return AgentDecision(
                agent_id=self.agent_id,
                action_type=ActionType.PROMOTION_CREATION,
                parameters={"error": str(e)},
                rationale=f"Failed to complete promotional analysis: {str(e)}",
                confidence_score=0.1,
                expected_outcome={"status": "error"},
                context={"error": str(e)},
            )

    def update_decision_outcome(self, decision_id: str, outcome_data: Dict[str, Any]):
        """
        Update the outcome of a previous promotional decision for learning.

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


# Global promotion agent instance
promotion_agent = PromotionAgent()
