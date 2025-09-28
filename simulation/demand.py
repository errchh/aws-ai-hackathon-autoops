"""
Demand Generator for Healthcare and Wellness Products

This module generates realistic demand patterns for healthcare and wellness products,
including seasonal trends, demographic factors, and market influences.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import random

from .products import ProductCatalog

logger = logging.getLogger(__name__)


class DemandGenerator:
    """
    Generates realistic demand patterns for healthcare and wellness products
    based on seasonal trends, demographics, and market conditions.
    """

    def __init__(self):
        self.product_catalog: Optional[ProductCatalog] = None
        self.base_demand_patterns: Dict[str, Dict[str, Any]] = {}
        self.current_demand_multipliers: Dict[str, float] = {}
        self._initialized = False

    async def initialize(self, product_catalog: ProductCatalog) -> None:
        """Initialize demand patterns for all products"""
        if self._initialized:
            return

        self.product_catalog = product_catalog
        logger.info("Initializing demand patterns for healthcare/wellness products...")

        # Define seasonal demand patterns for different categories
        self.base_demand_patterns = {
            "immune_support": {
                "base_daily_demand": 25,
                "seasonal_multipliers": {
                    1: 2.5,  # January - Post-holiday immune boost
                    2: 2.2,  # February - Cold/flu season peak
                    3: 1.8,  # March - Spring transition
                    4: 1.3,  # April - Allergy season
                    5: 1.1,  # May - Spring wellness
                    6: 1.0,  # June - Summer baseline
                    7: 0.9,  # July - Summer low
                    8: 1.0,  # August - Back to school
                    9: 1.4,  # September - Fall preparation
                    10: 1.8,  # October - Early winter prep
                    11: 2.1,  # November - Holiday immune focus
                    12: 2.8,  # December - Winter peak
                },
                "weekend_multiplier": 1.3,
                "trend_sensitivity": 0.8,
            },
            "fitness_gear": {
                "base_daily_demand": 18,
                "seasonal_multipliers": {
                    1: 2.8,  # January - New Year resolutions
                    2: 2.5,  # February - Fitness month
                    3: 2.2,  # March - Spring training
                    4: 1.8,  # April - Outdoor activities
                    5: 1.5,  # May - Summer prep
                    6: 1.3,  # June - Summer fitness
                    7: 1.2,  # July - Summer maintenance
                    8: 1.4,  # August - Back to school
                    9: 1.6,  # September - Fall fitness
                    10: 1.8,  # October - Holiday prep
                    11: 2.1,  # November - Thanksgiving
                    12: 2.4,  # December - Holiday fitness
                },
                "weekend_multiplier": 1.4,
                "trend_sensitivity": 0.9,
            },
            "stress_relief": {
                "base_daily_demand": 22,
                "seasonal_multipliers": {
                    1: 1.8,  # January - Post-holiday stress
                    2: 1.6,  # February - Winter blues
                    3: 1.7,  # March - Spring renewal
                    4: 1.9,  # April - Tax season stress
                    5: 2.1,  # May - Mental health month
                    6: 1.8,  # June - Summer relaxation
                    7: 1.5,  # July - Vacation recovery
                    8: 1.7,  # August - Back to routine
                    9: 1.9,  # September - School year stress
                    10: 2.2,  # October - Halloween/pre-holiday
                    11: 2.5,  # November - Holiday stress
                    12: 2.3,  # December - Holiday overwhelm
                },
                "weekend_multiplier": 1.2,
                "trend_sensitivity": 0.7,
            },
            "digestive_health": {
                "base_daily_demand": 15,
                "seasonal_multipliers": {
                    1: 2.2,  # January - Post-holiday detox
                    2: 1.8,  # February - Winter eating
                    3: 1.6,  # March - Spring cleaning
                    4: 1.4,  # April - Fresh produce season
                    5: 1.3,  # May - Spring renewal
                    6: 1.2,  # June - Summer eating
                    7: 1.1,  # July - Vacation eating
                    8: 1.3,  # August - Back to routine
                    9: 1.4,  # September - Fall harvest
                    10: 1.5,  # October - Comfort foods
                    11: 1.9,  # November - Holiday eating
                    12: 2.1,  # December - Festive indulgence
                },
                "weekend_multiplier": 1.1,
                "trend_sensitivity": 0.6,
            },
            "general": {
                "base_daily_demand": 12,
                "seasonal_multipliers": {month: 1.0 for month in range(1, 13)},
                "weekend_multiplier": 1.0,
                "trend_sensitivity": 0.5,
            },
        }

        # Initialize current demand multipliers
        for product_id in self.product_catalog.products.keys():
            self.current_demand_multipliers[product_id] = 1.0

        self._initialized = True
        logger.info("Demand patterns initialized for all product categories")

    async def update_for_time(self, current_time: datetime) -> None:
        """Update demand patterns based on current time"""
        if not self._initialized:
            return

        # Update seasonal multipliers
        current_month = current_time.month
        current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        is_weekend = current_weekday >= 5

        for product_id, product_data in self.product_catalog.product_data.items():
            seasonal_category = product_data.seasonal_category
            pattern = self.base_demand_patterns.get(
                seasonal_category, self.base_demand_patterns["general"]
            )

            # Calculate seasonal multiplier
            seasonal_mult = pattern["seasonal_multipliers"][current_month]

            # Apply weekend multiplier
            weekend_mult = pattern["weekend_multiplier"] if is_weekend else 1.0

            # Add some random variation (±10%)
            random_variation = random.uniform(0.9, 1.1)

            # Calculate final multiplier
            final_multiplier = seasonal_mult * weekend_mult * random_variation

            self.current_demand_multipliers[product_id] = final_multiplier

    async def get_current_demand(self) -> Dict[str, Any]:
        """Get current demand data for all products"""
        if not self._initialized:
            return {}

        demand_data = {}
        for product_id, multiplier in self.current_demand_multipliers.items():
            product_data = self.product_catalog.product_data[product_id]
            seasonal_category = product_data.seasonal_category
            pattern = self.base_demand_patterns.get(
                seasonal_category, self.base_demand_patterns["general"]
            )

            base_demand = pattern["base_daily_demand"]
            current_demand = base_demand * multiplier

            demand_data[product_id] = {
                "base_demand": base_demand,
                "current_multiplier": multiplier,
                "current_demand": current_demand,
                "seasonal_category": seasonal_category,
                "trend_sensitivity": pattern["trend_sensitivity"],
            }

        return demand_data

    async def get_demand_forecast(
        self, product_id: str, days_ahead: int = 7
    ) -> List[float]:
        """Generate demand forecast for a specific product"""
        if not self._initialized or product_id not in self.product_catalog.products:
            return []

        product_data = self.product_catalog.product_data[product_id]
        seasonal_category = product_data.seasonal_category
        pattern = self.base_demand_patterns.get(
            seasonal_category, self.base_demand_patterns["general"]
        )

        forecast = []
        base_demand = pattern["base_daily_demand"]

        for day in range(days_ahead):
            future_time = datetime.now() + timedelta(days=day)
            future_month = future_time.month
            future_weekday = future_time.weekday()
            is_future_weekend = future_weekday >= 5

            # Seasonal multiplier for future month
            seasonal_mult = pattern["seasonal_multipliers"][future_month]
            weekend_mult = pattern["weekend_multiplier"] if is_future_weekend else 1.0

            # Add forecast uncertainty (±15% for longer forecasts)
            uncertainty = 1.0 + (day * 0.02)  # Increases uncertainty over time
            random_factor = random.uniform(1 / uncertainty, uncertainty)

            daily_demand = base_demand * seasonal_mult * weekend_mult * random_factor
            forecast.append(round(daily_demand, 1))

        return forecast

    async def apply_market_event_impact(
        self, product_ids: List[str], impact_magnitude: float, duration_hours: int = 24
    ) -> None:
        """Apply market event impact to specific products"""
        if not self._initialized:
            return

        # Apply temporary demand boost/reduction
        impact_multiplier = 1.0 + impact_magnitude

        for product_id in product_ids:
            if product_id in self.current_demand_multipliers:
                current_mult = self.current_demand_multipliers[product_id]
                new_mult = current_mult * impact_multiplier
                self.current_demand_multipliers[product_id] = new_mult

                logger.info(
                    f"Applied market event impact to {product_id}: {impact_magnitude:+.1%}"
                )

        # Schedule impact removal after duration
        # In a real implementation, this would use asyncio scheduling

    async def get_seasonal_trends(self) -> Dict[str, Any]:
        """Get seasonal trend analysis for all categories"""
        if not self._initialized:
            return {}

        trends = {}
        current_month = datetime.now().month

        for category, pattern in self.base_demand_patterns.items():
            current_mult = pattern["seasonal_multipliers"][current_month]
            avg_mult = sum(pattern["seasonal_multipliers"].values()) / 12

            trends[category] = {
                "current_seasonal_multiplier": current_mult,
                "average_multiplier": avg_mult,
                "seasonal_intensity": (current_mult - avg_mult) / avg_mult,
                "peak_month": max(
                    pattern["seasonal_multipliers"].items(), key=lambda x: x[1]
                )[0],
            }

        return trends

    async def reset(self) -> None:
        """Reset demand patterns to initial state"""
        for product_id in self.product_catalog.products.keys():
            self.current_demand_multipliers[product_id] = 1.0

        logger.info("Demand patterns reset to initial state")
