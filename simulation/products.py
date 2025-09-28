"""
Healthcare and Wellness Product Catalog

This module defines the product catalog for healthcare and wellness retail,
including product categories, properties, and initialization data.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

from models.core import Product

logger = logging.getLogger(__name__)


@dataclass
class HealthcareProductData:
    """Data structure for healthcare product definitions"""

    id: str
    name: str
    category: str
    subcategory: str
    base_price: float
    cost: float
    initial_stock: int
    reorder_point: int
    supplier_lead_time_days: int
    shelf_life_days: Optional[int] = None
    health_necessity_score: float = 0.5  # 0.0 = luxury, 1.0 = essential
    regulatory_category: str = "wellness_product"
    seasonal_category: str = "general"


class ProductCatalog:
    """
    Manages the healthcare and wellness product catalog with realistic
    pricing, inventory, and product characteristics.
    """

    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.product_data: Dict[str, HealthcareProductData] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the product catalog with healthcare/wellness products"""
        if self._initialized:
            return

        logger.info("Initializing healthcare/wellness product catalog...")

        # Define healthcare products
        healthcare_products = [
            # Vitamins & Supplements
            HealthcareProductData(
                id="vit_d_1000iu",
                name="Vitamin D 1000IU",
                category="healthcare_essential",
                subcategory="vitamins",
                base_price=12.99,
                cost=8.50,
                initial_stock=150,
                reorder_point=50,
                supplier_lead_time_days=3,
                shelf_life_days=730,
                health_necessity_score=0.9,
                regulatory_category="supplement",
                seasonal_category="immune_support",
            ),
            HealthcareProductData(
                id="vit_c_500mg",
                name="Vitamin C 500mg",
                category="healthcare_essential",
                subcategory="vitamins",
                base_price=9.99,
                cost=6.25,
                initial_stock=200,
                reorder_point=75,
                supplier_lead_time_days=3,
                shelf_life_days=365,
                health_necessity_score=0.8,
                regulatory_category="supplement",
                seasonal_category="immune_support",
            ),
            HealthcareProductData(
                id="vit_b12_1000mcg",
                name="Vitamin B12 1000mcg",
                category="healthcare_essential",
                subcategory="vitamins",
                base_price=14.99,
                cost=9.75,
                initial_stock=120,
                reorder_point=40,
                supplier_lead_time_days=3,
                shelf_life_days=730,
                health_necessity_score=0.7,
                regulatory_category="supplement",
                seasonal_category="energy_focus",
            ),
            HealthcareProductData(
                id="probiotics_50b",
                name="Probiotics 50 Billion CFU",
                category="healthcare_essential",
                subcategory="supplements",
                base_price=24.99,
                cost=16.50,
                initial_stock=80,
                reorder_point=25,
                supplier_lead_time_days=5,
                shelf_life_days=180,
                health_necessity_score=0.6,
                regulatory_category="supplement",
                seasonal_category="digestive_health",
            ),
            # First Aid Supplies
            HealthcareProductData(
                id="bandages_assorted",
                name="Assorted Adhesive Bandages",
                category="healthcare_essential",
                subcategory="first_aid",
                base_price=4.99,
                cost=2.75,
                initial_stock=300,
                reorder_point=100,
                supplier_lead_time_days=2,
                shelf_life_days=1825,
                health_necessity_score=1.0,
                regulatory_category="medical_device",
                seasonal_category="general",
            ),
            HealthcareProductData(
                id="antiseptic_cream",
                name="Antiseptic Cream",
                category="healthcare_essential",
                subcategory="first_aid",
                base_price=6.99,
                cost=4.25,
                initial_stock=150,
                reorder_point=50,
                supplier_lead_time_days=3,
                shelf_life_days=1095,
                health_necessity_score=0.9,
                regulatory_category="medical_device",
                seasonal_category="general",
            ),
            # Personal Care Items
            HealthcareProductData(
                id="hand_sanitizer",
                name="Hand Sanitizer 70%",
                category="healthcare_essential",
                subcategory="personal_care",
                base_price=3.99,
                cost=2.25,
                initial_stock=400,
                reorder_point=150,
                supplier_lead_time_days=2,
                shelf_life_days=730,
                health_necessity_score=0.8,
                regulatory_category="wellness_product",
                seasonal_category="immune_support",
            ),
            HealthcareProductData(
                id="digital_thermometer",
                name="Digital Thermometer",
                category="healthcare_essential",
                subcategory="health_monitoring",
                base_price=12.99,
                cost=8.00,
                initial_stock=100,
                reorder_point=30,
                supplier_lead_time_days=7,
                shelf_life_days=None,
                health_necessity_score=0.8,
                regulatory_category="medical_device",
                seasonal_category="general",
            ),
            # Health Monitoring Devices
            HealthcareProductData(
                id="bp_monitor",
                name="Blood Pressure Monitor",
                category="healthcare_essential",
                subcategory="health_monitoring",
                base_price=49.99,
                cost=32.50,
                initial_stock=50,
                reorder_point=15,
                supplier_lead_time_days=10,
                shelf_life_days=None,
                health_necessity_score=0.7,
                regulatory_category="medical_device",
                seasonal_category="general",
            ),
            HealthcareProductData(
                id="pulse_oximeter",
                name="Pulse Oximeter",
                category="healthcare_essential",
                subcategory="health_monitoring",
                base_price=24.99,
                cost=16.25,
                initial_stock=75,
                reorder_point=20,
                supplier_lead_time_days=7,
                shelf_life_days=None,
                health_necessity_score=0.6,
                regulatory_category="medical_device",
                seasonal_category="general",
            ),
        ]

        # Define wellness products
        wellness_products = [
            # Essential Oils
            HealthcareProductData(
                id="lavender_oil",
                name="Lavender Essential Oil",
                category="wellness_premium",
                subcategory="essential_oils",
                base_price=18.99,
                cost=12.50,
                initial_stock=120,
                reorder_point=40,
                supplier_lead_time_days=5,
                shelf_life_days=1825,
                health_necessity_score=0.2,
                regulatory_category="wellness_product",
                seasonal_category="stress_relief",
            ),
            HealthcareProductData(
                id="eucalyptus_oil",
                name="Eucalyptus Essential Oil",
                category="wellness_premium",
                subcategory="essential_oils",
                base_price=16.99,
                cost=11.25,
                initial_stock=100,
                reorder_point=35,
                supplier_lead_time_days=5,
                shelf_life_days=1825,
                health_necessity_score=0.2,
                regulatory_category="wellness_product",
                seasonal_category="immune_support",
            ),
            HealthcareProductData(
                id="tea_tree_oil",
                name="Tea Tree Essential Oil",
                category="wellness_premium",
                subcategory="essential_oils",
                base_price=15.99,
                cost=10.75,
                initial_stock=110,
                reorder_point=38,
                supplier_lead_time_days=5,
                shelf_life_days=1825,
                health_necessity_score=0.2,
                regulatory_category="wellness_product",
                seasonal_category="general",
            ),
            # Fitness Accessories
            HealthcareProductData(
                id="yoga_mat",
                name="Premium Yoga Mat",
                category="wellness_premium",
                subcategory="fitness_accessories",
                base_price=39.99,
                cost=26.00,
                initial_stock=60,
                reorder_point=20,
                supplier_lead_time_days=7,
                shelf_life_days=None,
                health_necessity_score=0.3,
                regulatory_category="wellness_product",
                seasonal_category="fitness_gear",
            ),
            HealthcareProductData(
                id="resistance_bands",
                name="Resistance Bands Set",
                category="wellness_premium",
                subcategory="fitness_accessories",
                base_price=19.99,
                cost=13.25,
                initial_stock=90,
                reorder_point=30,
                supplier_lead_time_days=5,
                shelf_life_days=None,
                health_necessity_score=0.3,
                regulatory_category="wellness_product",
                seasonal_category="fitness_gear",
            ),
            # Meditation Tools
            HealthcareProductData(
                id="aromatherapy_diffuser",
                name="Aromatherapy Diffuser",
                category="wellness_premium",
                subcategory="meditation_tools",
                base_price=34.99,
                cost=23.00,
                initial_stock=45,
                reorder_point=15,
                supplier_lead_time_days=7,
                shelf_life_days=None,
                health_necessity_score=0.2,
                regulatory_category="wellness_product",
                seasonal_category="stress_relief",
            ),
            HealthcareProductData(
                id="meditation_cushion",
                name="Meditation Cushion",
                category="wellness_premium",
                subcategory="meditation_tools",
                base_price=29.99,
                cost=19.75,
                initial_stock=55,
                reorder_point=18,
                supplier_lead_time_days=5,
                shelf_life_days=None,
                health_necessity_score=0.2,
                regulatory_category="wellness_product",
                seasonal_category="stress_relief",
            ),
            # Organic/Natural Products
            HealthcareProductData(
                id="herbal_tea_blend",
                name="Immune Support Herbal Tea",
                category="wellness_premium",
                subcategory="organic_products",
                base_price=8.99,
                cost=5.75,
                initial_stock=180,
                reorder_point=60,
                supplier_lead_time_days=4,
                shelf_life_days=730,
                health_necessity_score=0.4,
                regulatory_category="wellness_product",
                seasonal_category="immune_support",
            ),
            HealthcareProductData(
                id="protein_powder",
                name="Organic Plant Protein Powder",
                category="wellness_premium",
                subcategory="organic_products",
                base_price=32.99,
                cost=21.50,
                initial_stock=70,
                reorder_point=25,
                supplier_lead_time_days=6,
                shelf_life_days=365,
                health_necessity_score=0.3,
                regulatory_category="supplement",
                seasonal_category="fitness_gear",
            ),
        ]

        # Combine all products
        all_products = healthcare_products + wellness_products

        # Create Product instances
        for product_data in all_products:
            self.product_data[product_data.id] = product_data

            # Calculate expiration date if applicable
            expiration_date = None
            if product_data.shelf_life_days:
                expiration_date = (
                    datetime.now() + timedelta(days=product_data.shelf_life_days)
                ).isoformat()

            product = Product(
                id=product_data.id,
                name=product_data.name,
                category=product_data.category,
                base_price=product_data.base_price,
                current_price=product_data.base_price,  # Start with base price
                cost=product_data.cost,
                inventory_level=product_data.initial_stock,
                reorder_point=product_data.reorder_point,
                supplier_lead_time=product_data.supplier_lead_time_days,
            )

            self.products[product_data.id] = product

        self._initialized = True
        logger.info(f"Initialized {len(self.products)} healthcare/wellness products")

    async def get_product(self, product_id: str) -> Optional[Product]:
        """Get a product by ID"""
        return self.products.get(product_id)

    async def get_products_by_category(self, category: str) -> List[Product]:
        """Get all products in a category"""
        return [p for p in self.products.values() if p.category == category]

    async def get_products_by_seasonal_category(
        self, seasonal_category: str
    ) -> List[Product]:
        """Get products by seasonal category"""
        return [
            p
            for p in self.products.values()
            if self.product_data[p.id].seasonal_category == seasonal_category
        ]

    async def get_all_products_data(self) -> List[Dict[str, Any]]:
        """Get all products data for dashboard/API"""
        return [
            {
                "id": p.id,
                "name": p.name,
                "category": p.category,
                "subcategory": self.product_data[p.id].subcategory,
                "current_price": p.current_price,
                "inventory_level": p.inventory_level,
                "reorder_point": p.reorder_point,
                "health_necessity_score": self.product_data[
                    p.id
                ].health_necessity_score,
                "seasonal_category": self.product_data[p.id].seasonal_category,
            }
            for p in self.products.values()
        ]

    async def update_product_price(self, product_id: str, new_price: float) -> bool:
        """Update product price"""
        if product_id in self.products:
            self.products[product_id].current_price = new_price
            return True
        return False

    async def update_inventory(self, product_id: str, new_level: int) -> bool:
        """Update product inventory level"""
        if product_id in self.products:
            self.products[product_id].inventory_level = new_level
            return True
        return False

    async def get_low_stock_products(self, threshold: int = 50) -> List[Product]:
        """Get products with low inventory"""
        return [p for p in self.products.values() if p.inventory_level <= threshold]

    async def get_expiring_products(self, days_ahead: int = 90) -> List[Product]:
        """Get products expiring within specified days"""
        expiring = []
        cutoff_date = datetime.now() + timedelta(days=days_ahead)

        for product_id, product_data in self.product_data.items():
            if product_data.shelf_life_days:
                # Calculate expiration date from shelf life
                # This is a simplified calculation - in reality would need creation date
                exp_date = datetime.now() + timedelta(days=product_data.shelf_life_days)
                if exp_date <= cutoff_date:
                    expiring.append(self.products[product_id])

        return expiring

    async def reset(self) -> None:
        """Reset catalog to initial state"""
        for product_id, product_data in self.product_data.items():
            product = self.products[product_id]
            product.current_price = product_data.base_price
            product.inventory_level = product_data.initial_stock

        logger.info("Product catalog reset to initial state")
