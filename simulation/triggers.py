"""
Trigger Engine for Healthcare and Wellness Simulation Scenarios

This module manages trigger scenarios that activate agent functions
during simulation demonstrations.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import random
import asyncio

from .products import ProductCatalog
from config.simulation_event_capture import get_simulation_event_capture

logger = logging.getLogger(__name__)


@dataclass
class TriggerScenario:
    """Represents a trigger scenario with conditions and effects"""

    id: str
    name: str
    description: str
    agent_type: (
        str  # 'pricing', 'inventory', 'promotion', 'orchestrator', 'collaboration'
    )
    trigger_type: str  # 'demand_spike', 'price_change', 'stock_alert', etc.
    intensity: str  # 'low', 'medium', 'high'
    conditions: Dict[str, Any] = field(default_factory=dict)
    effects: Dict[str, Any] = field(default_factory=dict)
    cooldown_minutes: int = 30
    last_triggered: Optional[datetime] = None


@dataclass
class ActiveTrigger:
    """Represents an active trigger instance"""

    scenario_id: str
    triggered_at: datetime
    expires_at: datetime
    intensity: str
    effects: Dict[str, Any]


class TriggerEngine:
    """
    Manages trigger scenarios for simulation demonstrations,
    including scheduling, activation, and effect application.
    """

    def __init__(self):
        self.product_catalog: Optional[ProductCatalog] = None
        self.scenarios: Dict[str, TriggerScenario] = {}
        self.active_triggers: List[ActiveTrigger] = []
        self.trigger_history: List[Dict[str, Any]] = []
        self._initialized = False
        
        # Initialize event capture for Langfuse tracing
        self.event_capture = get_simulation_event_capture()

    async def initialize(self, product_catalog: ProductCatalog) -> None:
        """Initialize trigger scenarios"""
        if self._initialized:
            return

        self.product_catalog = product_catalog
        logger.info("Initializing trigger engine for healthcare/wellness scenarios...")

        # Define pricing agent triggers
        pricing_triggers = [
            TriggerScenario(
                id="pricing_immune_demand_spike",
                name="Immune Support Supplements Demand +250%",
                description="Immune support supplements demand surge due to seasonal trends",
                agent_type="pricing",
                trigger_type="demand_spike",
                intensity="high",
                conditions={
                    "seasonal_category": "immune_support",
                    "demand_multiplier": 2.5,
                    "affected_products": [
                        "vit_c_500mg",
                        "vit_d_1000iu",
                        "probiotics_50b",
                    ],
                },
                effects={
                    "function_calls": [
                        "analyze_demand_elasticity",
                        "calculate_optimal_price",
                        "retrieve_pricing_history",
                    ],
                    "expected_actions": ["Price increase for immune supplements"],
                },
            ),
            TriggerScenario(
                id="pricing_competitor_vitamin_drop",
                name="Competitor Vitamin Brand Price Drop -25%",
                description="Major competitor reduces vitamin prices significantly",
                agent_type="pricing",
                trigger_type="competitor_price_change",
                intensity="medium",
                conditions={
                    "competitor": "nature_made",
                    "price_change_percent": -25,
                    "affected_products": ["vit_d_1000iu", "vit_c_500mg"],
                },
                effects={
                    "function_calls": [
                        "get_competitor_prices",
                        "calculate_optimal_price",
                        "make_pricing_decision",
                    ],
                    "expected_actions": ["Competitive price matching or undercutting"],
                },
            ),
            TriggerScenario(
                id="pricing_fitness_slow_moving",
                name="Slow-Moving Fitness Accessories: 45 Days No Sales",
                description="Fitness products haven't sold in 45+ days",
                agent_type="pricing",
                trigger_type="slow_moving_inventory",
                intensity="medium",
                conditions={
                    "days_without_sales": 45,
                    "category": "fitness_accessories",
                    "affected_products": ["yoga_mat", "resistance_bands"],
                },
                effects={
                    "function_calls": [
                        "apply_markdown_strategy",
                        "evaluate_price_impact",
                        "update_decision_outcome",
                    ],
                    "expected_actions": ["Apply clearance pricing to move inventory"],
                },
            ),
            TriggerScenario(
                id="pricing_seasonal_demand_analysis",
                name="Seasonal Demand Elasticity Analysis",
                description="Analyze demand elasticity for seasonal products",
                agent_type="pricing",
                trigger_type="seasonal_analysis",
                intensity="medium",
                conditions={
                    "season": "winter",
                    "analysis_type": "elasticity",
                    "affected_products": ["vit_d_1000iu", "vit_c_500mg"],
                },
                effects={
                    "function_calls": [
                        "analyze_demand_elasticity",
                        "retrieve_pricing_history",
                        "make_pricing_decision",
                    ],
                    "expected_actions": [
                        "Complete elasticity analysis and pricing decision"
                    ],
                },
            ),
            TriggerScenario(
                id="pricing_market_research_update",
                name="Market Research Update: Competitor Analysis",
                description="Update competitor pricing intelligence",
                agent_type="pricing",
                trigger_type="market_research",
                intensity="low",
                conditions={
                    "research_type": "competitor_analysis",
                    "update_frequency": "weekly",
                    "affected_products": ["vit_d_1000iu"],
                },
                effects={
                    "function_calls": [
                        "get_competitor_prices",
                        "evaluate_price_impact",
                        "update_decision_outcome",
                    ],
                    "expected_actions": ["Update competitor pricing database"],
                },
            ),
        ]

        # Define inventory agent triggers
        inventory_triggers = [
            TriggerScenario(
                id="inventory_vitamin_winter_stock",
                name="Vitamin D Stock Critical Before Winter",
                description="Vitamin D inventory low as winter immune season approaches",
                agent_type="inventory",
                trigger_type="stock_alert",
                intensity="high",
                conditions={
                    "product_id": "vit_d_1000iu",
                    "current_stock": 25,
                    "season": "pre_winter",
                    "service_level_threshold": 0.95,
                },
                effects={
                    "function_calls": [
                        "forecast_demand_probabilistic",
                        "generate_restock_alert",
                        "analyze_demand_patterns",
                    ],
                    "expected_actions": ["Emergency restock order for Vitamin D"],
                },
            ),
            TriggerScenario(
                id="inventory_probiotics_forecast_spike",
                name="Probiotic Forecast +180% (Post-Holiday Detox)",
                description="Probiotic demand forecast shows 180% increase after holidays",
                agent_type="inventory",
                trigger_type="demand_forecast_update",
                intensity="high",
                conditions={
                    "product_id": "probiotics_50b",
                    "forecast_increase_percent": 180,
                    "trigger_reason": "post_holiday_detox",
                },
                effects={
                    "function_calls": [
                        "forecast_demand_probabilistic",
                        "calculate_safety_buffer",
                        "make_inventory_decision",
                    ],
                    "expected_actions": ["Increase safety stock for probiotics"],
                },
            ),
            TriggerScenario(
                id="inventory_oils_buffer_breach",
                name="Essential Oils Safety Buffer Breached",
                description="Essential oils inventory below safety buffer threshold",
                agent_type="inventory",
                trigger_type="safety_buffer_breach",
                intensity="medium",
                conditions={
                    "category": "essential_oils",
                    "buffer_breach_percent": 15,
                    "affected_products": [
                        "lavender_oil",
                        "eucalyptus_oil",
                        "tea_tree_oil",
                    ],
                },
                effects={
                    "function_calls": [
                        "identify_slow_moving_inventory",
                        "generate_restock_alert",
                        "retrieve_inventory_history",
                    ],
                    "expected_actions": ["Restock essential oils to maintain buffer"],
                },
            ),
            TriggerScenario(
                id="inventory_seasonal_pattern_analysis",
                name="Seasonal Demand Pattern Analysis",
                description="Analyze demand patterns for seasonal products",
                agent_type="inventory",
                trigger_type="pattern_analysis",
                intensity="medium",
                conditions={
                    "analysis_period": "quarterly",
                    "category": "supplements",
                    "pattern_type": "seasonal",
                },
                effects={
                    "function_calls": [
                        "analyze_demand_patterns",
                        "forecast_demand_probabilistic",
                        "update_decision_outcome",
                    ],
                    "expected_actions": [
                        "Complete pattern analysis and update forecasts"
                    ],
                },
            ),
            TriggerScenario(
                id="inventory_inventory_optimization",
                name="Inventory Optimization Review",
                description="Review and optimize inventory levels across categories",
                agent_type="inventory",
                trigger_type="optimization_review",
                intensity="low",
                conditions={
                    "review_type": "quarterly_optimization",
                    "optimization_target": "service_level_vs_cost",
                    "affected_categories": ["vitamins", "supplements"],
                },
                effects={
                    "function_calls": [
                        "retrieve_inventory_history",
                        "make_inventory_decision",
                        "calculate_safety_buffer",
                    ],
                    "expected_actions": [
                        "Optimize inventory levels for cost efficiency"
                    ],
                },
            ),
        ]

        # Define promotion agent triggers
        promotion_triggers = [
            TriggerScenario(
                id="promotion_wellness_routine_trending",
                name="Wellness Routine Trending on Social Media",
                description="Wellness morning routines gaining massive social traction",
                agent_type="promotion",
                trigger_type="social_trend",
                intensity="high",
                conditions={
                    "trend_topic": "morning_wellness_routine",
                    "sentiment_score": 0.7,
                    "mentions_threshold": 2000,
                },
                effects={
                    "function_calls": [
                        "analyze_social_sentiment",
                        "create_flash_sale",
                        "coordinate_with_pricing_agent",
                    ],
                    "expected_actions": ["Create flash sale for wellness bundle"],
                },
            ),
            TriggerScenario(
                id="promotion_probiotics_bundle_affinity",
                name="Bundle Opportunity: Probiotics + Vitamin C 0.87 Affinity",
                description="High product affinity detected for probiotic-vitamin C bundle",
                agent_type="promotion",
                trigger_type="bundle_opportunity",
                intensity="medium",
                conditions={
                    "product_pair": ["probiotics_50b", "vit_c_500mg"],
                    "affinity_score": 0.87,
                    "min_affinity_threshold": 0.8,
                },
                effects={
                    "function_calls": [
                        "generate_bundle_recommendation",
                        "schedule_promotional_campaign",
                        "validate_inventory_availability",
                    ],
                    "expected_actions": ["Create immune support bundle promotion"],
                },
            ),
            TriggerScenario(
                id="promotion_meditation_flash_sale",
                name="Wellness Flash Sale: Meditation Essentials",
                description="Meditation products ready for seasonal flash sale",
                agent_type="promotion",
                trigger_type="seasonal_promotion",
                intensity="medium",
                conditions={
                    "category": "meditation_tools",
                    "season": "stress_relief_peak",
                    "inventory_ready": True,
                },
                effects={
                    "function_calls": [
                        "create_flash_sale",
                        "evaluate_campaign_effectiveness",
                        "retrieve_promotion_history",
                    ],
                    "expected_actions": ["Launch meditation essentials flash sale"],
                },
            ),
            TriggerScenario(
                id="promotion_cross_agent_coordination",
                name="Cross-Agent Promotion Coordination",
                description="Coordinate promotion with pricing and inventory agents",
                agent_type="promotion",
                trigger_type="agent_coordination",
                intensity="high",
                conditions={
                    "coordination_type": "multi_agent_campaign",
                    "campaign_type": "seasonal_wellness",
                    "agents_involved": ["pricing", "inventory"],
                },
                effects={
                    "function_calls": [
                        "coordinate_with_pricing_agent",
                        "validate_inventory_availability",
                        "schedule_promotional_campaign",
                    ],
                    "expected_actions": ["Coordinate multi-agent promotional campaign"],
                },
            ),
            TriggerScenario(
                id="promotion_campaign_performance_review",
                name="Campaign Performance Review and Optimization",
                description="Review past campaign performance and optimize future promotions",
                agent_type="promotion",
                trigger_type="performance_review",
                intensity="medium",
                conditions={
                    "review_period": "monthly",
                    "metrics_to_review": ["effectiveness", "roi", "conversion"],
                    "optimization_target": "maximize_roi",
                },
                effects={
                    "function_calls": [
                        "evaluate_campaign_effectiveness",
                        "retrieve_promotion_history",
                        "generate_bundle_recommendation",
                    ],
                    "expected_actions": ["Review and optimize promotional campaigns"],
                },
            ),
        ]

        # Define orchestrator triggers
        orchestrator_triggers = [
            TriggerScenario(
                id="orchestrator_wellness_event",
                name="Multi-Category Wellness Event Coordination",
                description="Coordinated wellness event across multiple product categories",
                agent_type="orchestrator",
                trigger_type="multi_category_event",
                intensity="high",
                conditions={
                    "event_type": "wellness_week",
                    "categories_involved": [
                        "essential_oils",
                        "meditation_tools",
                        "vitamins",
                    ],
                    "demand_multiplier": 2.0,
                },
                effects={
                    "function_calls": [
                        "process_market_event",
                        "coordinate_agents",
                        "trigger_collaboration_workflow",
                    ],
                    "expected_actions": [
                        "Coordinate pricing, inventory, and promotion agents"
                    ],
                },
            ),
            TriggerScenario(
                id="orchestrator_healthcare_compliance",
                name="Healthcare Compliance Alert",
                description="Regulatory compliance check for healthcare products",
                agent_type="orchestrator",
                trigger_type="compliance_check",
                intensity="medium",
                conditions={
                    "regulatory_category": "supplement",
                    "check_frequency": "quarterly",
                    "alert_type": "compliance_review",
                },
                effects={
                    "function_calls": ["get_system_status", "register_agents"],
                    "expected_actions": ["Ensure healthcare product compliance"],
                },
            ),
            TriggerScenario(
                id="orchestrator_system_health_monitoring",
                name="System Health and Performance Monitoring",
                description="Monitor overall system health and agent performance",
                agent_type="orchestrator",
                trigger_type="system_monitoring",
                intensity="low",
                conditions={
                    "monitoring_type": "continuous",
                    "check_frequency": "hourly",
                    "alert_thresholds": {"response_time": 2.0, "error_rate": 0.05},
                },
                effects={
                    "function_calls": ["get_system_status", "coordinate_agents"],
                    "expected_actions": ["Monitor and maintain system health"],
                },
            ),
        ]

        # Define collaboration triggers
        collaboration_triggers = [
            TriggerScenario(
                id="collaboration_supplements_fitness",
                name="Cross-Category Learning: Supplements → Fitness",
                description="Apply successful supplement pricing strategies to fitness products",
                agent_type="collaboration",
                trigger_type="cross_category_learning",
                intensity="medium",
                conditions={
                    "source_category": "supplements",
                    "target_category": "fitness_accessories",
                    "success_metric": "pricing_effectiveness",
                    "threshold": 0.8,
                },
                effects={
                    "function_calls": [
                        "cross_agent_learning_from_outcomes",
                        "inventory_to_pricing_slow_moving_alert",
                        "collaborative_market_event_response",
                    ],
                    "expected_actions": [
                        "Apply learned pricing patterns across categories"
                    ],
                },
            ),
            TriggerScenario(
                id="collaboration_influencer_campaign",
                name="Health Influencer Campaign Coordination",
                description="Coordinate influencer campaign across agents",
                agent_type="collaboration",
                trigger_type="influencer_campaign",
                intensity="high",
                conditions={
                    "influencer_type": "wellness_coach",
                    "reach_threshold": 50000,
                    "campaign_type": "seasonal_wellness",
                },
                effects={
                    "function_calls": [
                        "pricing_to_promotion_discount_coordination",
                        "promotion_to_inventory_stock_validation",
                        "collaborative_market_event_response",
                    ],
                    "expected_actions": ["Coordinated campaign across all agents"],
                },
            ),
            TriggerScenario(
                id="collaboration_market_event_coordination",
                name="Collaborative Market Event Response",
                description="Coordinate agent responses to major market events",
                agent_type="collaboration",
                trigger_type="market_event_coordination",
                intensity="high",
                conditions={
                    "event_type": "major_market_shift",
                    "impact_level": "high",
                    "coordination_required": True,
                },
                effects={
                    "function_calls": [
                        "collaborative_market_event_response",
                        "cross_agent_learning_from_outcomes",
                        "pricing_to_promotion_discount_coordination",
                    ],
                    "expected_actions": ["Execute coordinated market event response"],
                },
            ),
        ]

        # Combine all scenarios
        all_scenarios = (
            pricing_triggers
            + inventory_triggers
            + promotion_triggers
            + orchestrator_triggers
            + collaboration_triggers
        )

        for scenario in all_scenarios:
            self.scenarios[scenario.id] = scenario

        self._initialized = True
        logger.info(f"Initialized {len(self.scenarios)} trigger scenarios")

    async def trigger_scenario(
        self, scenario_name: str, intensity: str = "medium"
    ) -> bool:
        """Trigger a specific scenario by name"""
        if not self._initialized:
            return False

        # Find scenario by name
        scenario = None
        for s in self.scenarios.values():
            if s.name == scenario_name:
                scenario = s
                break

        if not scenario:
            logger.warning(f"Scenario '{scenario_name}' not found")
            return False

        # Check cooldown
        if scenario.last_triggered:
            cooldown_end = scenario.last_triggered + timedelta(
                minutes=scenario.cooldown_minutes
            )
            if datetime.now() < cooldown_end:
                logger.info(f"Scenario '{scenario_name}' still in cooldown")
                return False

        # Create active trigger
        triggered_at = datetime.now()
        expires_at = triggered_at + timedelta(hours=2)  # Triggers last 2 hours

        active_trigger = ActiveTrigger(
            scenario_id=scenario.id,
            triggered_at=triggered_at,
            expires_at=expires_at,
            intensity=intensity,
            effects=scenario.effects.copy(),
        )

        self.active_triggers.append(active_trigger)
        scenario.last_triggered = triggered_at

        # Log trigger event
        trigger_history_entry = {
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "triggered_at": triggered_at.isoformat(),
            "intensity": intensity,
            "agent_type": scenario.agent_type,
            "trigger_type": scenario.trigger_type,
        }
        self.trigger_history.append(trigger_history_entry)

        # Capture scenario trigger event for Langfuse tracing
        scenario_trigger_data = {
            "scenario_id": scenario.id,
            "scenario_name": scenario.name,
            "agent_type": scenario.agent_type,
            "trigger_type": scenario.trigger_type,
            "intensity": intensity,
            "conditions": scenario.conditions,
            "effects": scenario.effects,
            "cooldown_minutes": scenario.cooldown_minutes
        }
        self.event_capture.capture_scenario_trigger(scenario_trigger_data)

        logger.info(f"Triggered scenario: {scenario.name} (intensity: {intensity})")
        return True

    async def get_available_triggers(self) -> List[Dict[str, Any]]:
        """Get list of available trigger scenarios"""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "agent_type": s.agent_type,
                "trigger_type": s.trigger_type,
                "intensity_options": ["low", "medium", "high"],
                "cooldown_minutes": s.cooldown_minutes,
                "last_triggered": s.last_triggered.isoformat()
                if s.last_triggered
                else None,
            }
            for s in self.scenarios.values()
        ]

    async def get_active_triggers(self) -> List[Dict[str, Any]]:
        """Get currently active triggers"""
        current_time = datetime.now()
        active = [
            {
                "scenario_id": t.scenario_id,
                "scenario_name": self.scenarios[t.scenario_id].name,
                "triggered_at": t.triggered_at.isoformat(),
                "expires_at": t.expires_at.isoformat(),
                "intensity": t.intensity,
                "agent_type": self.scenarios[t.scenario_id].agent_type,
                "time_remaining_minutes": int(
                    (t.expires_at - current_time).total_seconds() / 60
                ),
            }
            for t in self.active_triggers
            if t.expires_at > current_time
        ]

        return active

    async def process_scheduled_triggers(self) -> None:
        """Process any scheduled trigger activations"""
        # This would handle time-based trigger scheduling
        # For now, it's a placeholder for future enhancement
        pass

    async def cleanup_expired_triggers(self) -> None:
        """Remove expired triggers"""
        current_time = datetime.now()
        self.active_triggers = [
            t for t in self.active_triggers if t.expires_at > current_time
        ]

    async def get_trigger_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trigger history"""
        return self.trigger_history[-limit:] if self.trigger_history else []

    async def create_custom_trigger(
        self,
        name: str,
        agent_type: str,
        trigger_type: str,
        conditions: Dict[str, Any],
        effects: Dict[str, Any],
        intensity: str = "medium",
    ) -> str:
        """Create a custom trigger scenario"""
        trigger_id = f"custom_{int(datetime.now().timestamp())}"

        scenario = TriggerScenario(
            id=trigger_id,
            name=name,
            description=f"Custom {agent_type} trigger: {name}",
            agent_type=agent_type,
            trigger_type=trigger_type,
            intensity=intensity,
            conditions=conditions,
            effects=effects,
        )

        self.scenarios[trigger_id] = scenario
        logger.info(f"Created custom trigger: {name}")
        return trigger_id

    async def reset(self) -> None:
        """Reset trigger engine to initial state"""
        self.active_triggers = []
        self.trigger_history = []

        # Reset last_triggered timestamps
        for scenario in self.scenarios.values():
            scenario.last_triggered = None

        logger.info("Trigger engine reset to initial state")
