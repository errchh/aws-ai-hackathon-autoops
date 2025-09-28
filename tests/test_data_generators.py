"""
Test data generators for comprehensive Langfuse workflow visualization testing.

This module provides realistic test data generators for creating comprehensive
test scenarios that exercise the entire Langfuse integration system.
"""

import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from uuid import uuid4

from models.core import MarketEvent, EventType, Product, AgentDecision, ActionType


class TestDataGenerator:
    """Comprehensive test data generator for Langfuse workflow testing."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the test data generator with optional seed for reproducibility."""
        if seed is not None:
            random.seed(seed)

        # Product catalog for realistic scenarios
        self.product_catalog = self._generate_product_catalog()

        # Agent configurations
        self.agent_configs = {
            "inventory_agent": {
                "tools": [
                    "demand_forecasting",
                    "safety_buffer_calculation",
                    "slow_moving_detection",
                ],
                "decision_types": [
                    "restock_alert",
                    "slow_moving_identification",
                    "stock_level_adjustment",
                ],
            },
            "pricing_agent": {
                "tools": [
                    "elasticity_analysis",
                    "optimal_price_calculation",
                    "markdown_strategy",
                ],
                "decision_types": [
                    "price_update",
                    "markdown_decision",
                    "competitor_analysis",
                ],
            },
            "promotion_agent": {
                "tools": [
                    "social_sentiment_analysis",
                    "campaign_creation",
                    "bundle_recommendation",
                ],
                "decision_types": [
                    "campaign_creation",
                    "flash_sale",
                    "promotional_campaign",
                ],
            },
        }

    def _generate_product_catalog(self, num_products: int = 50) -> List[Product]:
        """Generate a realistic product catalog."""
        categories = ["Electronics", "Clothing", "Home", "Sports", "Books", "Food"]
        products = []

        for i in range(num_products):
            cost = round(random.uniform(5, 400), 2)
            # Ensure current_price is at least 50% of cost to satisfy validation
            min_current_price = cost * 0.5
            current_price = round(random.uniform(min_current_price, 500), 2)
            base_price = round(random.uniform(current_price, 600), 2)
            inventory_level = random.randint(0, 1000)
            # Ensure reorder_point doesn't exceed twice the current inventory
            max_reorder_point = min(100, inventory_level * 2)
            reorder_point = (
                random.randint(10, max_reorder_point) if max_reorder_point >= 10 else 10
            )

            product = Product(
                id=f"PROD_{i:03d}",
                name=f"Product {i:03d}",
                category=random.choice(categories),
                base_price=base_price,
                current_price=current_price,
                cost=cost,
                inventory_level=inventory_level,
                reorder_point=reorder_point,
                supplier_lead_time=random.randint(1, 14),
            )
            products.append(product)

        return products

    def generate_market_events(self, count: int = 10) -> List[MarketEvent]:
        """Generate realistic market events."""
        events = []
        event_types = list(EventType)

        for i in range(count):
            event_type = random.choice(event_types)

            # Generate affected products
            num_affected = random.randint(1, 5)
            affected_products = random.sample(
                [p.id for p in self.product_catalog], num_affected
            )

            # Generate impact magnitude based on event type (ensure >= 0)
            if event_type == EventType.DEMAND_SPIKE:
                impact_magnitude = round(random.uniform(1.5, 3.0), 2)
            elif event_type == EventType.COMPETITOR_PRICE_CHANGE:
                impact_magnitude = round(
                    random.uniform(0.1, 0.3), 2
                )  # Positive impact for competitor price changes
            elif event_type == EventType.SUPPLY_DISRUPTION:
                impact_magnitude = round(
                    random.uniform(0.2, 0.8), 2
                )  # Positive impact magnitude for disruptions
            elif event_type == EventType.SEASONAL_CHANGE:
                impact_magnitude = round(random.uniform(1.2, 2.5), 2)
            else:
                impact_magnitude = round(random.uniform(0.1, 0.5), 2)

            # Generate metadata based on event type
            metadata = self._generate_event_metadata(event_type)

            event = MarketEvent(
                id=uuid4(),
                event_type=event_type,
                affected_products=affected_products,
                impact_magnitude=impact_magnitude,
                description=f"Generated {event_type.value} event {i}",
                timestamp=datetime.now() + timedelta(minutes=random.randint(-60, 60)),
                metadata=metadata,
            )
            events.append(event)

        return events

    def _generate_event_metadata(self, event_type: EventType) -> Dict[str, Any]:
        """Generate realistic metadata for different event types."""
        base_metadata = {
            "source": random.choice(
                [
                    "market_intelligence",
                    "social_media",
                    "competitor_monitoring",
                    "supplier_alert",
                ]
            ),
            "confidence": round(random.uniform(0.7, 0.95), 2),
            "region": random.choice(["US", "EU", "ASIA", "GLOBAL"]),
        }

        if event_type == EventType.DEMAND_SPIKE:
            base_metadata.update(
                {
                    "trend_source": random.choice(
                        ["viral_challenge", "celebrity_endorsement", "news_event"]
                    ),
                    "expected_duration_days": random.randint(3, 14),
                    "social_engagement_score": round(random.uniform(0.6, 0.9), 2),
                }
            )
        elif event_type == EventType.COMPETITOR_PRICE_CHANGE:
            base_metadata.update(
                {
                    "competitor": random.choice(
                        ["Amazon", "Walmart", "Target", "BestBuy"]
                    ),
                    "price_change_percentage": round(random.uniform(-25, -5), 1),
                    "affected_categories": random.sample(
                        ["Electronics", "Clothing", "Home"], random.randint(1, 2)
                    ),
                }
            )
        elif event_type == EventType.SUPPLY_DISRUPTION:
            base_metadata.update(
                {
                    "disruption_type": random.choice(
                        ["logistics", "manufacturing", "raw_materials", "labor"]
                    ),
                    "supplier": f"Supplier_{random.randint(1, 20)}",
                    "expected_resolution_days": random.randint(7, 30),
                }
            )
        elif event_type == EventType.SEASONAL_CHANGE:
            base_metadata.update(
                {
                    "season": random.choice(
                        ["back_to_school", "holiday", "summer", "winter"]
                    ),
                    "historical_impact": round(random.uniform(1.1, 2.0), 2),
                    "category_boost": random.sample(
                        ["Clothing", "Sports", "Books"], random.randint(1, 2)
                    ),
                }
            )

        return base_metadata

    def generate_agent_decisions(self, count: int = 20) -> List[AgentDecision]:
        """Generate realistic agent decisions."""
        decisions = []
        agents = list(self.agent_configs.keys())

        for i in range(count):
            agent_id = random.choice(agents)
            agent_config = self.agent_configs[agent_id]

            # Generate decision based on agent type
            if agent_id == "inventory_agent":
                decision = self._generate_inventory_decision(i)
            elif agent_id == "pricing_agent":
                decision = self._generate_pricing_decision(i)
            else:  # promotion_agent
                decision = self._generate_promotion_decision(i)

            decisions.append(decision)

        return decisions

    def _generate_inventory_decision(self, index: int) -> AgentDecision:
        """Generate a realistic inventory agent decision."""
        decision_types = self.agent_configs["inventory_agent"]["decision_types"]
        decision_type = random.choice(decision_types)

        if decision_type == "restock_alert":
            parameters = {
                "product_id": random.choice([p.id for p in self.product_catalog]),
                "current_stock": random.randint(5, 50),
                "recommended_order": random.randint(100, 500),
                "urgency": random.choice(["low", "medium", "high"]),
            }
            rationale = (
                f"Stock level {parameters['current_stock']} is below safety threshold"
            )
        elif decision_type == "slow_moving_identification":
            parameters = {
                "product_id": random.choice([p.id for p in self.product_catalog]),
                "days_without_sale": random.randint(30, 90),
                "current_stock": random.randint(50, 200),
                "suggested_action": random.choice(
                    ["markdown", "promotion", "clearance"]
                ),
            }
            rationale = (
                f"Product has not sold for {parameters['days_without_sale']} days"
            )
        else:  # stock_level_adjustment
            parameters = {
                "product_id": random.choice([p.id for p in self.product_catalog]),
                "adjustment_type": random.choice(["increase", "decrease"]),
                "adjustment_percentage": round(random.uniform(10, 50), 1),
                "reason": random.choice(
                    ["seasonal_demand", "trend_analysis", "supplier_change"]
                ),
            }
            rationale = f"Adjusting stock levels based on {parameters['reason']}"

        return AgentDecision(
            id=uuid4(),
            agent_id="inventory_agent",
            action_type=ActionType.INVENTORY_RESTOCK,
            parameters=parameters,
            rationale=rationale,
            confidence_score=round(random.uniform(0.7, 0.95), 2),
            expected_outcome={"inventory_turnover_improvement": 0.15},
            timestamp=datetime.now() + timedelta(minutes=random.randint(-30, 30)),
            context={"decision_context": "automated_analysis"},
        )

    def _generate_pricing_decision(self, index: int) -> AgentDecision:
        """Generate a realistic pricing agent decision."""
        decision_types = self.agent_configs["pricing_agent"]["decision_types"]
        decision_type = random.choice(decision_types)

        if decision_type == "price_update":
            parameters = {
                "product_id": random.choice([p.id for p in self.product_catalog]),
                "current_price": round(random.uniform(50, 500), 2),
                "new_price": round(random.uniform(45, 550), 2),
                "price_change_percentage": round(random.uniform(-15, 25), 1),
                "reason": random.choice(
                    ["demand_analysis", "competitor_pricing", "cost_change"]
                ),
            }
            rationale = f"Price optimization based on {parameters['reason']}"
        elif decision_type == "markdown_decision":
            parameters = {
                "product_id": random.choice([p.id for p in self.product_catalog]),
                "markdown_percentage": round(random.uniform(10, 40), 1),
                "duration_days": random.randint(7, 30),
                "reason": random.choice(["clearance", "seasonal", "slow_moving"]),
            }
            rationale = f"Markdown strategy for {parameters['reason']}"
        else:  # competitor_analysis
            parameters = {
                "competitor": random.choice(["Amazon", "Walmart", "Target"]),
                "price_comparison": {
                    "our_price": round(random.uniform(50, 500), 2),
                    "competitor_price": round(random.uniform(45, 550), 2),
                    "price_gap": round(random.uniform(-20, 20), 1),
                },
                "recommended_action": random.choice(["match", "undercut", "premium"]),
            }
            rationale = f"Competitive analysis against {parameters['competitor']}"

        return AgentDecision(
            id=uuid4(),
            agent_id="pricing_agent",
            action_type=ActionType.PRICE_ADJUSTMENT,
            parameters=parameters,
            rationale=rationale,
            confidence_score=round(random.uniform(0.75, 0.95), 2),
            expected_outcome={"revenue_impact": 0.12},
            timestamp=datetime.now() + timedelta(minutes=random.randint(-30, 30)),
            context={"decision_context": "market_analysis"},
        )

    def _generate_promotion_decision(self, index: int) -> AgentDecision:
        """Generate a realistic promotion agent decision."""
        decision_types = self.agent_configs["promotion_agent"]["decision_types"]
        decision_type = random.choice(decision_types)

        if decision_type == "campaign_creation":
            parameters = {
                "campaign_type": random.choice(
                    ["flash_sale", "seasonal", "clearance", "new_product"]
                ),
                "target_products": random.sample(
                    [p.id for p in self.product_catalog], random.randint(1, 5)
                ),
                "discount_percentage": round(random.uniform(10, 30), 1),
                "duration_days": random.randint(3, 14),
                "target_audience": random.choice(
                    ["all", "new_customers", "loyal_customers", "high_value"]
                ),
            }
            rationale = f"Creating {parameters['campaign_type']} campaign"
        elif decision_type == "flash_sale":
            parameters = {
                "product_id": random.choice([p.id for p in self.product_catalog]),
                "discount_percentage": round(random.uniform(20, 50), 1),
                "duration_hours": random.randint(4, 24),
                "max_participants": random.randint(100, 1000),
                "trigger": random.choice(
                    ["low_stock", "seasonal", "competitor_response"]
                ),
            }
            rationale = f"Flash sale for {parameters['trigger']}"
        else:  # promotional_campaign
            parameters = {
                "campaign_name": f"Campaign_{index:03d}",
                "products": random.sample(
                    [p.id for p in self.product_catalog], random.randint(2, 8)
                ),
                "promotion_type": random.choice(["bundle", "cross_sell", "upsell"]),
                "discount_structure": {
                    "base_discount": round(random.uniform(5, 15), 1),
                    "bundle_bonus": round(random.uniform(5, 10), 1),
                },
            }
            rationale = f"Promotional campaign: {parameters['promotion_type']}"

        return AgentDecision(
            id=uuid4(),
            agent_id="promotion_agent",
            action_type=ActionType.PROMOTION_CREATION,
            parameters=parameters,
            rationale=rationale,
            confidence_score=round(random.uniform(0.7, 0.9), 2),
            expected_outcome={"conversion_rate_increase": 0.18},
            timestamp=datetime.now() + timedelta(minutes=random.randint(-30, 30)),
            context={"decision_context": "marketing_analysis"},
        )

    def generate_collaboration_scenarios(self, count: int = 5) -> List[Dict[str, Any]]:
        """Generate realistic collaboration workflow scenarios."""
        scenarios = []

        for i in range(count):
            scenario_type = random.choice(
                [
                    "inventory_to_pricing_slow_moving",
                    "pricing_to_promotion_discount",
                    "promotion_to_inventory_validation",
                ]
            )

            if scenario_type == "inventory_to_pricing_slow_moving":
                scenario_data = {
                    "workflow_type": scenario_type,
                    "slow_moving_items": [
                        {
                            "product_id": random.choice(
                                [p.id for p in self.product_catalog]
                            ),
                            "days_slow": random.randint(30, 90),
                            "current_stock": random.randint(50, 200),
                            "suggested_action": random.choice(
                                ["markdown", "promotion"]
                            ),
                        }
                        for _ in range(random.randint(1, 3))
                    ],
                    "participating_agents": ["inventory_agent", "pricing_agent"],
                    "priority": random.choice(["low", "medium", "high"]),
                }
            elif scenario_type == "pricing_to_promotion_discount":
                scenario_data = {
                    "workflow_type": scenario_type,
                    "discount_opportunities": [
                        {
                            "product_id": random.choice(
                                [p.id for p in self.product_catalog]
                            ),
                            "suggested_discount": round(random.uniform(10, 30), 1),
                            "reason": random.choice(
                                ["clearance", "seasonal", "competitor_response"]
                            ),
                            "expected_impact": round(random.uniform(1.2, 2.5), 2),
                        }
                        for _ in range(random.randint(1, 4))
                    ],
                    "participating_agents": ["pricing_agent", "promotion_agent"],
                    "budget_allocation": round(random.uniform(1000, 10000), 2),
                }
            else:  # promotion_to_inventory_validation
                scenario_data = {
                    "workflow_type": scenario_type,
                    "campaign_requests": [
                        {
                            "product_id": random.choice(
                                [p.id for p in self.product_catalog]
                            ),
                            "campaign_type": random.choice(
                                ["flash_sale", "seasonal_promotion"]
                            ),
                            "duration_days": random.randint(3, 14),
                            "required_stock": random.randint(50, 500),
                        }
                        for _ in range(random.randint(1, 3))
                    ],
                    "participating_agents": ["promotion_agent", "inventory_agent"],
                    "validation_criteria": [
                        "stock_availability",
                        "demand_forecast",
                        "profit_margin",
                    ],
                }

            scenarios.append(scenario_data)

        return scenarios

    def generate_simulation_scenarios(self, count: int = 3) -> List[Dict[str, Any]]:
        """Generate realistic simulation scenarios."""
        scenarios = []

        for i in range(count):
            scenario = {
                "scenario_id": f"sim_scenario_{i:03d}",
                "scenario_name": f"Simulation Scenario {i:03d}",
                "description": f"Comprehensive simulation scenario {i:03d}",
                "duration_minutes": random.randint(30, 120),
                "event_frequency": random.choice(["low", "medium", "high"]),
                "agent_activation_rate": round(random.uniform(0.5, 0.9), 2),
                "market_conditions": {
                    "demand_level": random.choice(
                        ["low", "normal", "high", "volatile"]
                    ),
                    "competition_intensity": random.choice(["low", "medium", "high"]),
                    "supply_stability": random.choice(
                        ["stable", "moderate", "disrupted"]
                    ),
                },
                "expected_outcomes": {
                    "revenue_impact": round(random.uniform(-0.1, 0.3), 2),
                    "efficiency_improvement": round(random.uniform(0.05, 0.25), 2),
                    "collaboration_score": round(random.uniform(0.7, 0.95), 2),
                },
            }
            scenarios.append(scenario)

        return scenarios

    def generate_performance_test_data(self, event_count: int = 100) -> Dict[str, Any]:
        """Generate data for performance testing."""
        return {
            "events": self.generate_market_events(event_count),
            "decisions": self.generate_agent_decisions(event_count // 2),
            "collaboration_scenarios": self.generate_collaboration_scenarios(
                min(10, event_count // 10)
            ),
            "simulation_scenarios": self.generate_simulation_scenarios(1),
            "performance_targets": {
                "max_processing_time_seconds": 300,  # 5 minutes
                "min_success_rate": 0.95,
                "max_memory_increase_mb": 100,
                "max_cpu_usage_percent": 80,
            },
        }

    def generate_load_test_data(self, concurrent_events: int = 20) -> Dict[str, Any]:
        """Generate data for load testing."""
        return {
            "concurrent_events": concurrent_events,
            "events": self.generate_market_events(concurrent_events),
            "load_profile": {
                "ramp_up_time_seconds": 30,
                "sustained_load_time_seconds": 120,
                "ramp_down_time_seconds": 30,
                "target_throughput_events_per_second": concurrent_events / 60,
            },
            "monitoring_points": [
                "response_times",
                "memory_usage",
                "cpu_usage",
                "trace_creation_rate",
                "error_rates",
            ],
        }

    def generate_stress_test_data(
        self, max_concurrent_events: int = 50
    ) -> Dict[str, Any]:
        """Generate data for stress testing."""
        return {
            "max_concurrent_events": max_concurrent_events,
            "stress_levels": [
                {"events": 10, "duration_seconds": 60},
                {"events": 25, "duration_seconds": 120},
                {"events": max_concurrent_events, "duration_seconds": 180},
                {"events": 5, "duration_seconds": 60},  # Cool down
            ],
            "failure_scenarios": [
                "langfuse_unavailable",
                "high_memory_usage",
                "network_latency",
                "agent_timeouts",
            ],
        }

    def generate_realistic_workflow_trace(self) -> Dict[str, Any]:
        """Generate a realistic complete workflow trace for testing."""
        # Start with a market event
        market_event = random.choice(self.generate_market_events(1))

        # Generate agent responses
        decisions = self.generate_agent_decisions(3)

        # Create collaboration scenario
        collaboration = random.choice(self.generate_collaboration_scenarios(1))

        return {
            "market_event": market_event,
            "agent_decisions": decisions,
            "collaboration_workflow": collaboration,
            "expected_trace_structure": {
                "root_trace": {
                    "type": "market_event",
                    "spans": [
                        {"agent": "orchestrator", "operation": "workflow_coordination"},
                        {"agent": "inventory_agent", "operation": "demand_analysis"},
                        {"agent": "pricing_agent", "operation": "price_optimization"},
                        {"agent": "promotion_agent", "operation": "campaign_creation"},
                    ],
                },
                "collaboration_trace": {
                    "type": "collaboration_workflow",
                    "participating_agents": collaboration["participating_agents"],
                },
            },
        }

    def generate_trace_validation_data(self) -> Dict[str, Any]:
        """Generate data for trace validation testing."""
        return {
            "valid_traces": [
                self.generate_realistic_workflow_trace() for _ in range(5)
            ],
            "invalid_traces": [
                {"missing_event_id": True, "trace_data": {"type": "incomplete"}},
                {
                    "missing_agent_responses": True,
                    "trace_data": {"event_id": "test", "type": "market_event"},
                },
                {
                    "invalid_timestamps": True,
                    "trace_data": {
                        "event_id": "test",
                        "start_time": "invalid",
                        "end_time": "also_invalid",
                    },
                },
            ],
            "edge_cases": [
                {
                    "empty_workflow": {"event_id": "empty", "agent_responses": []},
                    "single_agent": {
                        "event_id": "single",
                        "agent_responses": ["only_one"],
                    },
                    "timeout_scenario": {"event_id": "timeout", "timeout": True},
                }
            ],
        }


class MockLangfuseClient:
    """Mock Langfuse client for testing without actual Langfuse connection."""

    def __init__(self):
        """Initialize mock client."""
        self.traces = {}
        self.spans = {}
        self.is_available = True
        self.call_count = 0
        self.last_call_args = None

    def trace(self, *args, **kwargs):
        """Mock trace creation."""
        self.call_count += 1
        self.last_call_args = {"args": args, "kwargs": kwargs}

        trace_id = f"mock_trace_{self.call_count}"
        trace_data = {
            "trace_id": trace_id,
            "name": kwargs.get("name", "mock_trace"),
            "metadata": kwargs.get("metadata", {}),
            "created_at": datetime.now(),
        }
        self.traces[trace_id] = trace_data
        return MockTrace(trace_id, trace_data)

    def span(self, *args, **kwargs):
        """Mock span creation."""
        self.call_count += 1
        self.last_call_args = {"args": args, "kwargs": kwargs}

        span_id = f"mock_span_{self.call_count}"
        span_data = {
            "span_id": span_id,
            "name": kwargs.get("name", "mock_span"),
            "metadata": kwargs.get("metadata", {}),
            "created_at": datetime.now(),
        }
        self.spans[span_id] = span_data
        return MockSpan(span_id, span_data)

    def flush(self):
        """Mock flush operation."""
        return True

    def health_check(self):
        """Mock health check."""
        return {
            "status": "healthy" if self.is_available else "unavailable",
            "traces_count": len(self.traces),
            "spans_count": len(self.spans),
        }


class MockTrace:
    """Mock trace object for testing."""

    def __init__(self, trace_id: str, trace_data: Dict[str, Any]):
        """Initialize mock trace."""
        self.trace_id = trace_id
        self.trace_data = trace_data
        self.spans = []

    def span(self, *args, **kwargs):
        """Create a mock span."""
        span = MockSpan(f"span_{len(self.spans)}", kwargs)
        self.spans.append(span)
        return span

    def update(self, **kwargs):
        """Mock trace update."""
        self.trace_data.update(kwargs)

    def end(self):
        """Mock trace end."""
        self.trace_data["ended_at"] = datetime.now()


class MockSpan:
    """Mock span object for testing."""

    def __init__(self, span_id: str, span_data: Dict[str, Any]):
        """Initialize mock span."""
        self.span_id = span_id
        self.span_data = span_data

    def update(self, **kwargs):
        """Mock span update."""
        self.span_data.update(kwargs)

    def end(self):
        """Mock span end."""
        self.span_data["ended_at"] = datetime.now()

    def set_tag(self, key: str, value: Any):
        """Mock set tag."""
        self.span_data[f"tag_{key}"] = value

    def set_tags(self, tags: Dict[str, Any]):
        """Mock set tags."""
        for key, value in tags.items():
            self.span_data[f"tag_{key}"] = value


def create_test_scenario_generator(
    scenario_type: str = "comprehensive",
) -> TestDataGenerator:
    """Factory function to create test data generators with specific configurations."""
    if scenario_type == "performance":
        return TestDataGenerator(seed=42)  # Reproducible for performance testing
    elif scenario_type == "load":
        return TestDataGenerator(seed=123)  # Different seed for load testing
    elif scenario_type == "stress":
        return TestDataGenerator(seed=456)  # Different seed for stress testing
    else:
        return TestDataGenerator(seed=789)  # Default comprehensive testing
