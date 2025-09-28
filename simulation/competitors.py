"""
Competitor Landscape for Healthcare and Wellness Market

This module simulates competitor pricing and market positioning for
healthcare and wellness brands in the retail environment.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import random

from .products import ProductCatalog

logger = logging.getLogger(__name__)


@dataclass
class CompetitorBrand:
    """Represents a competitor brand with pricing strategy"""

    name: str
    category: str
    market_share: float
    price_strategy: str  # 'premium', 'value', 'discount'
    reliability_score: float  # 0.0-1.0
    product_lines: List[str]


class CompetitorLandscape:
    """
    Simulates competitor landscape for healthcare and wellness products,
    including pricing strategies, market share, and competitive actions.
    """

    def __init__(self):
        self.product_catalog: Optional[ProductCatalog] = None
        self.competitor_brands: Dict[str, CompetitorBrand] = {}
        self.competitor_prices: Dict[
            str, Dict[str, float]
        ] = {}  # product_id -> {competitor: price}
        self.market_events: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize competitor landscape"""
        if self._initialized:
            return

        logger.info("Initializing healthcare/wellness competitor landscape...")

        # Define major competitor brands
        self.competitor_brands = {
            "johnson_johnson": CompetitorBrand(
                name="Johnson & Johnson",
                category="healthcare_essential",
                market_share=0.15,
                price_strategy="premium",
                reliability_score=0.95,
                product_lines=["first_aid", "personal_care"],
            ),
            "pfizer_consumer": CompetitorBrand(
                name="Pfizer Consumer",
                category="healthcare_essential",
                market_share=0.12,
                price_strategy="value",
                reliability_score=0.92,
                product_lines=["vitamins", "supplements"],
            ),
            "cvs_health": CompetitorBrand(
                name="CVS Health",
                category="healthcare_essential",
                market_share=0.18,
                price_strategy="value",
                reliability_score=0.88,
                product_lines=["first_aid", "personal_care", "health_monitoring"],
            ),
            "nature_made": CompetitorBrand(
                name="Nature Made",
                category="healthcare_essential",
                market_share=0.10,
                price_strategy="value",
                reliability_score=0.90,
                product_lines=["vitamins", "supplements"],
            ),
            "garden_of_life": CompetitorBrand(
                name="Garden of Life",
                category="wellness_premium",
                market_share=0.08,
                price_strategy="premium",
                reliability_score=0.85,
                product_lines=["supplements", "organic_products"],
            ),
            "dotera": CompetitorBrand(
                name="doTERRA",
                category="wellness_premium",
                market_share=0.09,
                price_strategy="premium",
                reliability_score=0.87,
                product_lines=["essential_oils"],
            ),
            "young_living": CompetitorBrand(
                name="Young Living",
                category="wellness_premium",
                market_share=0.07,
                price_strategy="premium",
                reliability_score=0.83,
                product_lines=["essential_oils"],
            ),
            "amazon_basic": CompetitorBrand(
                name="Amazon Basic",
                category="general",
                market_share=0.21,
                price_strategy="discount",
                reliability_score=0.75,
                product_lines=["general"],
            ),
        }

        # Initialize competitor prices for all products
        await self._initialize_competitor_prices()

        self._initialized = True
        logger.info(f"Initialized {len(self.competitor_brands)} competitor brands")

    async def _initialize_competitor_prices(self) -> None:
        """Initialize competitor pricing for all products"""
        # This would be called after product catalog is available
        # For now, we'll set it up when competitors update
        pass

    async def update_competitors(self, current_time: datetime) -> None:
        """Update competitor actions and pricing"""
        if not self._initialized:
            return

        # Random competitor price changes (small adjustments)
        for product_id in self.competitor_prices.keys():
            # 10% chance of price change per update
            if random.random() < 0.1:
                competitor = random.choice(
                    list(self.competitor_prices[product_id].keys())
                )
                current_price = self.competitor_prices[product_id][competitor]

                # Small price adjustment (±5%)
                price_change = random.uniform(-0.05, 0.05)
                new_price = current_price * (1 + price_change)

                self.competitor_prices[product_id][competitor] = round(new_price, 2)

                # Log significant price changes
                if abs(price_change) > 0.03:
                    logger.info(
                        f"Competitor {competitor} changed price for {product_id}: "
                        f"${current_price:.2f} → ${new_price:.2f} ({price_change:+.1%})"
                    )

    async def get_current_prices(self) -> Dict[str, Dict[str, float]]:
        """Get current competitor prices for all products"""
        return self.competitor_prices.copy()

    async def get_competitor_price(
        self, product_id: str, competitor: str
    ) -> Optional[float]:
        """Get specific competitor price for a product"""
        if (
            product_id in self.competitor_prices
            and competitor in self.competitor_prices[product_id]
        ):
            return self.competitor_prices[product_id][competitor]
        return None

    async def get_market_average_price(self, product_id: str) -> Optional[float]:
        """Get market average price for a product across all competitors"""
        if product_id not in self.competitor_prices:
            return None

        prices = list(self.competitor_prices[product_id].values())
        if not prices:
            return None

        return sum(prices) / len(prices)

    async def get_price_range(self, product_id: str) -> Optional[Tuple[float, float]]:
        """Get price range (min, max) for a product across competitors"""
        if product_id not in self.competitor_prices:
            return None

        prices = list(self.competitor_prices[product_id].values())
        if not prices:
            return None

        return (min(prices), max(prices))

    async def simulate_price_war(
        self, product_category: str, duration_hours: int = 24
    ) -> None:
        """Simulate a price war in a specific product category"""
        affected_products = []

        # Find products in the category
        for product_id, product_data in self.product_catalog.product_data.items():
            if (
                product_data.category == product_category
                or product_data.subcategory in ["vitamins", "supplements"]
            ):
                affected_products.append(product_id)

        # Apply aggressive price cuts
        for product_id in affected_products:
            if product_id in self.competitor_prices:
                for competitor in self.competitor_prices[product_id]:
                    current_price = self.competitor_prices[product_id][competitor]
                    # 15-25% price cut
                    discount = random.uniform(0.15, 0.25)
                    new_price = current_price * (1 - discount)
                    self.competitor_prices[product_id][competitor] = round(new_price, 2)

        logger.info(
            f"Price war initiated in {product_category}: {len(affected_products)} products affected"
        )

    async def get_competitor_insights(self, product_id: str) -> Dict[str, Any]:
        """Get competitive intelligence for a specific product"""
        if product_id not in self.competitor_prices:
            return {}

        prices = self.competitor_prices[product_id]
        market_avg = await self.get_market_average_price(product_id)
        price_range = await self.get_price_range(product_id)

        insights = {
            "competitor_count": len(prices),
            "market_average": market_avg,
            "price_range": price_range,
            "price_distribution": {
                "premium_count": len(
                    [p for p in prices.values() if p > market_avg * 1.2]
                ),
                "value_count": len(
                    [p for p in prices.values() if p <= market_avg * 1.1]
                ),
                "discount_count": len(
                    [p for p in prices.values() if p < market_avg * 0.9]
                ),
            },
        }

        return insights

    async def get_brand_performance(self) -> Dict[str, Any]:
        """Get overall brand performance metrics"""
        performance = {}

        for brand_id, brand in self.competitor_brands.items():
            # Calculate average pricing strategy adherence
            brand_products = 0
            premium_pricing = 0

            for product_id, competitors in self.competitor_prices.items():
                if brand_id in competitors:
                    brand_products += 1
                    market_avg = await self.get_market_average_price(product_id)
                    if market_avg and competitors[brand_id] > market_avg * 1.1:
                        premium_pricing += 1

            performance[brand.name] = {
                "market_share": brand.market_share,
                "products_offered": brand_products,
                "premium_pricing_ratio": premium_pricing / max(1, brand_products),
                "reliability_score": brand.reliability_score,
            }

        return performance

    async def reset(self) -> None:
        """Reset competitor landscape to initial state"""
        # Reinitialize prices
        await self._initialize_competitor_prices()
        self.market_events = []

        logger.info("Competitor landscape reset to initial state")
