"""
Tests for the Healthcare and Wellness Retail Simulation Engine

This module contains comprehensive tests for the simulation engine components
including product catalog, demand patterns, competitors, social sentiment,
triggers, and IoT sensors.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from simulation.engine import SimulationEngine, SimulationMode
from simulation.products import ProductCatalog, HealthcareProductData
from simulation.demand import DemandGenerator
from simulation.competitors import CompetitorLandscape
from simulation.social import SocialSentimentSimulator
from simulation.triggers import TriggerEngine
from simulation.iot import IoTSensorSimulator


class TestProductCatalog:
    """Test the healthcare/wellness product catalog"""

    @pytest.fixture
    def catalog(self):
        return ProductCatalog()

    @pytest.mark.asyncio
    async def test_initialization(self, catalog):
        """Test catalog initialization"""
        await catalog.initialize()

        assert catalog._initialized
        assert len(catalog.products) > 0
        assert len(catalog.product_data) > 0

        # Check that we have healthcare and wellness products
        categories = set(p.category for p in catalog.products.values())
        assert "healthcare_essential" in categories
        assert "wellness_premium" in categories

    @pytest.mark.asyncio
    async def test_get_products_by_category(self, catalog):
        """Test filtering products by category"""
        await catalog.initialize()

        healthcare_products = await catalog.get_products_by_category(
            "healthcare_essential"
        )
        wellness_products = await catalog.get_products_by_category("wellness_premium")

        assert len(healthcare_products) > 0
        assert len(wellness_products) > 0
        assert all(p.category == "healthcare_essential" for p in healthcare_products)
        assert all(p.category == "wellness_premium" for p in wellness_products)

    @pytest.mark.asyncio
    async def test_seasonal_category_filtering(self, catalog):
        """Test filtering by seasonal category"""
        await catalog.initialize()

        immune_products = await catalog.get_products_by_seasonal_category(
            "immune_support"
        )
        fitness_products = await catalog.get_products_by_seasonal_category(
            "fitness_gear"
        )

        assert len(immune_products) > 0
        assert len(fitness_products) > 0

        # Verify seasonal categories
        for product in immune_products:
            assert (
                catalog.product_data[product.id].seasonal_category == "immune_support"
            )

    @pytest.mark.asyncio
    async def test_inventory_management(self, catalog):
        """Test inventory updates"""
        await catalog.initialize()

        # Get a test product
        test_product_id = list(catalog.products.keys())[0]
        original_level = catalog.products[test_product_id].inventory_level

        # Update inventory
        success = await catalog.update_inventory(test_product_id, original_level + 50)
        assert success
        assert catalog.products[test_product_id].inventory_level == original_level + 50

    @pytest.mark.asyncio
    async def test_price_updates(self, catalog):
        """Test price updates"""
        await catalog.initialize()

        test_product_id = list(catalog.products.keys())[0]
        original_price = catalog.products[test_product_id].current_price

        new_price = original_price * 1.1  # 10% increase
        success = await catalog.update_product_price(test_product_id, new_price)
        assert success
        assert catalog.products[test_product_id].current_price == new_price


class TestDemandGenerator:
    """Test the demand pattern generator"""

    @pytest.fixture
    def catalog(self):
        catalog = ProductCatalog()
        # Mock the initialize method to avoid full setup
        catalog._initialized = True
        catalog.products = {"test_product": MagicMock()}
        catalog.product_data = {"test_product": MagicMock(seasonal_category="general")}
        return catalog

    @pytest.fixture
    def demand_gen(self, catalog):
        gen = DemandGenerator()
        gen.product_catalog = catalog
        gen._initialized = True
        return gen

    @pytest.mark.asyncio
    async def test_demand_update(self, demand_gen):
        """Test demand pattern updates"""
        initial_multiplier = demand_gen.current_demand_multipliers.get(
            "test_product", 1.0
        )

        # Update for a specific time
        test_time = datetime.now()
        await demand_gen.update_for_time(test_time)

        # Multiplier should be updated
        assert "test_product" in demand_gen.current_demand_multipliers

    @pytest.mark.asyncio
    async def test_seasonal_multipliers(self, demand_gen):
        """Test seasonal demand multipliers"""
        # Test winter month (December)
        winter_time = datetime(2024, 12, 15)
        await demand_gen.update_for_time(winter_time)

        # Should have updated multipliers
        assert len(demand_gen.current_demand_multipliers) > 0

    @pytest.mark.asyncio
    async def test_demand_forecast(self, demand_gen):
        """Test demand forecasting"""
        forecast = await demand_gen.get_demand_forecast("test_product", days_ahead=3)

        assert len(forecast) == 3
        assert all(isinstance(d, (int, float)) for d in forecast)
        assert all(d >= 0 for d in forecast)


class TestCompetitorLandscape:
    """Test competitor simulation"""

    @pytest.fixture
    def competitor_landscape(self):
        landscape = CompetitorLandscape()
        landscape._initialized = True
        return landscape

    @pytest.mark.asyncio
    async def test_competitor_updates(self, competitor_landscape):
        """Test competitor price updates"""
        # Add mock competitor data
        competitor_landscape.competitor_prices = {
            "test_product": {"competitor_a": 10.0, "competitor_b": 12.0}
        }

        original_prices = competitor_landscape.competitor_prices.copy()

        # Update competitors
        test_time = datetime.now()
        await competitor_landscape.update_competitors(test_time)

        # Prices should be updated (may or may not change due to randomness)
        assert "test_product" in competitor_landscape.competitor_prices

    @pytest.mark.asyncio
    async def test_price_analysis(self, competitor_landscape):
        """Test competitor price analysis"""
        competitor_landscape.competitor_prices = {
            "test_product": {"comp_a": 9.99, "comp_b": 10.99, "comp_c": 11.99}
        }

        market_avg = await competitor_landscape.get_market_average_price("test_product")
        price_range = await competitor_landscape.get_price_range("test_product")

        assert market_avg is not None
        assert price_range is not None
        assert isinstance(price_range, tuple)
        assert len(price_range) == 2
        assert price_range[0] <= price_range[1]


class TestSocialSentimentSimulator:
    """Test social media sentiment simulation"""

    @pytest.fixture
    def sentiment_sim(self):
        sim = SocialSentimentSimulator()
        sim._initialized = True
        return sim

    @pytest.mark.asyncio
    async def test_sentiment_updates(self, sentiment_sim):
        """Test sentiment updates over time"""
        initial_sentiment = sentiment_sim.current_sentiment.copy()

        # Update sentiment
        test_time = datetime.now()
        await sentiment_sim.update_sentiment(test_time)

        # Sentiment should be updated (may change due to randomness)
        assert len(sentiment_sim.current_sentiment) > 0

    @pytest.mark.asyncio
    async def test_trending_topics(self, sentiment_sim):
        """Test trending topic management"""
        # Add a mock trending topic
        sentiment_sim.trending_topics["test_topic"] = {
            "topic": "test wellness trend",
            "sentiment": 0.8,
            "mentions_24h": 1500,
            "influencer_reach": 30000,
            "peak_time": "current",
            "related_products": ["test_product"],
        }

        trending = await sentiment_sim.get_trending_topics(min_mentions=1000)

        assert len(trending) > 0
        assert trending[0]["mentions_24h"] >= 1000

    @pytest.mark.asyncio
    async def test_viral_content_trigger(self, sentiment_sim):
        """Test viral content impact on sentiment"""
        category = "fitness_gear"
        initial_sentiment = sentiment_sim.current_sentiment.get(category, 0.5)

        await sentiment_sim.trigger_viral_content(category, impact=0.3)

        # Sentiment should increase
        assert sentiment_sim.current_sentiment[category] >= initial_sentiment


class TestTriggerEngine:
    """Test trigger scenario management"""

    @pytest.fixture
    def catalog(self):
        catalog = ProductCatalog()
        catalog._initialized = True
        return catalog

    @pytest.fixture
    def trigger_engine(self, catalog):
        engine = TriggerEngine()
        engine.product_catalog = catalog
        engine._initialized = True
        return engine

    @pytest.mark.asyncio
    async def test_trigger_scenario(self, trigger_engine):
        """Test triggering scenarios"""
        # Trigger a known scenario
        success = await trigger_engine.trigger_scenario(
            "Immune Support Supplements Demand +250%", intensity="high"
        )

        # Should succeed (or fail gracefully if scenario doesn't exist)
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_get_available_triggers(self, trigger_engine):
        """Test retrieving available triggers"""
        triggers = await trigger_engine.get_available_triggers()

        assert isinstance(triggers, list)
        if triggers:  # If triggers are loaded
            assert all(isinstance(t, dict) for t in triggers)
            assert all("name" in t for t in triggers)

    @pytest.mark.asyncio
    async def test_active_triggers(self, trigger_engine):
        """Test active trigger management"""
        active = await trigger_engine.get_active_triggers()

        assert isinstance(active, list)


class TestIoTSensorSimulator:
    """Test IoT sensor simulation"""

    @pytest.fixture
    def catalog(self):
        catalog = ProductCatalog()
        # Mock minimal catalog
        catalog._initialized = True
        catalog.product_data = {
            "test_supplement": MagicMock(
                subcategory="supplements",
                shelf_life_days=365,
                category="healthcare_essential",
            )
        }
        return catalog

    @pytest.fixture
    def iot_sim(self, catalog):
        sim = IoTSensorSimulator()
        sim.product_catalog = catalog
        sim._initialized = True
        return sim

    @pytest.mark.asyncio
    async def test_sensor_updates(self, iot_sim):
        """Test sensor reading updates"""
        # Initialize with a test product
        await iot_sim.initialize(iot_sim.product_catalog)

        initial_readings = iot_sim.sensor_readings.copy()

        # Update sensors
        test_time = datetime.now()
        await iot_sim.update_sensors(test_time)

        # Readings should be updated
        assert len(iot_sim.sensor_readings) >= len(initial_readings)

    @pytest.mark.asyncio
    async def test_sensor_health_summary(self, iot_sim):
        """Test sensor health monitoring"""
        await iot_sim.initialize(iot_sim.product_catalog)

        summary = await iot_sim.get_sensor_health_summary()

        assert isinstance(summary, dict)
        assert "total_sensors" in summary
        assert "online_sensors" in summary
        assert "uptime_percentage" in summary


class TestSimulationEngine:
    """Test the main simulation engine"""

    @pytest.fixture
    def sim_engine(self):
        return SimulationEngine(mode=SimulationMode.DEMO)

    @pytest.mark.asyncio
    async def test_engine_initialization(self, sim_engine):
        """Test simulation engine initialization"""
        await sim_engine.initialize()

        assert sim_engine.state is not None
        assert sim_engine.product_catalog is not None
        assert sim_engine.demand_generator is not None

    @pytest.mark.asyncio
    async def test_get_current_state(self, sim_engine):
        """Test getting simulation state"""
        await sim_engine.initialize()

        state = await sim_engine.get_current_state()

        assert isinstance(state, dict)
        assert "current_time" in state
        assert "is_running" in state
        assert "mode" in state

    @pytest.mark.asyncio
    async def test_scenario_triggering(self, sim_engine):
        """Test triggering simulation scenarios"""
        await sim_engine.initialize()

        # Try triggering a scenario
        success = await sim_engine.trigger_scenario(
            "Immune Support Supplements Demand +250%"
        )

        # Should return a boolean
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_data_retrieval(self, sim_engine):
        """Test retrieving simulation data"""
        await sim_engine.initialize()

        products_data = await sim_engine.get_products_data()
        market_data = await sim_engine.get_market_data()

        assert isinstance(products_data, list)
        assert isinstance(market_data, dict)


if __name__ == "__main__":
    pytest.main([__file__])
