"""
Shared data API endpoints for the autoops retail optimization system.

This module provides REST API endpoints for shared data access that all agents
can use, including product information and market data.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from models.core import Product, MarketEvent, SystemMetrics, EventType


router = APIRouter()


# Response Models
class ProductListResponse(BaseModel):
    """Response model for product listing."""
    products: List[Product] = Field(..., description="List of products")
    total_count: int = Field(..., description="Total number of products")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")


class MarketDataResponse(BaseModel):
    """Response model for current market data."""
    timestamp: datetime = Field(..., description="Data timestamp")
    active_events: List[MarketEvent] = Field(..., description="Currently active market events")
    market_indicators: Dict[str, float] = Field(..., description="Key market indicators")
    competitor_activity: Dict[str, str] = Field(..., description="Recent competitor activities")
    demand_trends: Dict[str, float] = Field(..., description="Category demand trends")


class ProductSearchResponse(BaseModel):
    """Response model for product search."""
    products: List[Product] = Field(..., description="Matching products")
    search_query: str = Field(..., description="Original search query")
    total_matches: int = Field(..., description="Total number of matches")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")


# Shared Data Endpoints
@router.get("/products", response_model=ProductListResponse)
async def get_products(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price filter"),
    low_stock_only: bool = Query(False, description="Show only low stock items")
) -> ProductListResponse:
    """
    Get a paginated list of products with optional filtering.
    
    This endpoint provides agents with access to the product catalog
    with various filtering and pagination options.
    """
    try:
        # Simulate product data (in real implementation, this would query the database)
        simulated_products = [
            Product(
                id="SKU123456",
                name="Premium Coffee Beans 1kg",
                category="Beverages",
                base_price=24.99,
                current_price=22.99,
                cost=12.50,
                inventory_level=150,
                reorder_point=25,
                supplier_lead_time=7
            ),
            Product(
                id="SKU789012",
                name="Organic Green Tea 500g",
                category="Beverages",
                base_price=18.99,
                current_price=18.99,
                cost=9.25,
                inventory_level=8,  # Low stock
                reorder_point=15,
                supplier_lead_time=5
            ),
            Product(
                id="SKU345678",
                name="Artisan Chocolate Bar 200g",
                category="Confectionery",
                base_price=12.99,
                current_price=10.99,
                cost=6.50,
                inventory_level=75,
                reorder_point=20,
                supplier_lead_time=10
            ),
            Product(
                id="SKU901234",
                name="Gourmet Pasta 500g",
                category="Food",
                base_price=8.99,
                current_price=8.99,
                cost=4.25,
                inventory_level=200,
                reorder_point=50,
                supplier_lead_time=3
            )
        ]
        
        # Apply filters
        filtered_products = simulated_products
        
        if category:
            filtered_products = [p for p in filtered_products if p.category.lower() == category.lower()]
        
        if min_price is not None:
            filtered_products = [p for p in filtered_products if p.current_price >= min_price]
        
        if max_price is not None:
            filtered_products = [p for p in filtered_products if p.current_price <= max_price]
        
        if low_stock_only:
            filtered_products = [p for p in filtered_products if p.inventory_level <= p.reorder_point]
        
        # Apply pagination
        total_count = len(filtered_products)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_products = filtered_products[start_idx:end_idx]
        
        return ProductListResponse(
            products=paginated_products,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve products: {str(e)}"
        )


@router.get("/products/{product_id}", response_model=Product)
async def get_product(product_id: str) -> Product:
    """
    Get detailed information for a specific product.
    
    This endpoint provides agents with complete product information
    including current pricing, inventory levels, and supplier details.
    """
    try:
        # Simulate product lookup (in real implementation, this would query the database)
        if product_id == "SKU123456":
            return Product(
                id="SKU123456",
                name="Premium Coffee Beans 1kg",
                category="Beverages",
                base_price=24.99,
                current_price=22.99,
                cost=12.50,
                inventory_level=150,
                reorder_point=25,
                supplier_lead_time=7
            )
        elif product_id == "SKU789012":
            return Product(
                id="SKU789012",
                name="Organic Green Tea 500g",
                category="Beverages",
                base_price=18.99,
                current_price=18.99,
                cost=9.25,
                inventory_level=8,
                reorder_point=15,
                supplier_lead_time=5
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product {product_id} not found"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve product: {str(e)}"
        )


@router.get("/products/search", response_model=ProductSearchResponse)
async def search_products(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return")
) -> ProductSearchResponse:
    """
    Search products by name, category, or other attributes.
    
    This endpoint allows agents to find products using text search
    across various product attributes.
    """
    try:
        import time
        start_time = time.time()
        
        # Simulate product search (in real implementation, this would use full-text search)
        all_products = [
            Product(
                id="SKU123456",
                name="Premium Coffee Beans 1kg",
                category="Beverages",
                base_price=24.99,
                current_price=22.99,
                cost=12.50,
                inventory_level=150,
                reorder_point=25,
                supplier_lead_time=7
            ),
            Product(
                id="SKU789012",
                name="Organic Green Tea 500g",
                category="Beverages",
                base_price=18.99,
                current_price=18.99,
                cost=9.25,
                inventory_level=8,
                reorder_point=15,
                supplier_lead_time=5
            ),
            Product(
                id="SKU345678",
                name="Artisan Chocolate Bar 200g",
                category="Confectionery",
                base_price=12.99,
                current_price=10.99,
                cost=6.50,
                inventory_level=75,
                reorder_point=20,
                supplier_lead_time=10
            )
        ]
        
        # Simple text search simulation
        query_lower = query.lower()
        matching_products = [
            p for p in all_products
            if query_lower in p.name.lower() or query_lower in p.category.lower()
        ]
        
        # Apply limit
        limited_results = matching_products[:limit]
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return ProductSearchResponse(
            products=limited_results,
            search_query=query,
            total_matches=len(matching_products),
            search_time_ms=round(search_time_ms, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search products: {str(e)}"
        )


@router.get("/market-data/current", response_model=MarketDataResponse)
async def get_current_market_data() -> MarketDataResponse:
    """
    Get current market conditions and events.
    
    This endpoint provides agents with real-time market intelligence
    including active events, competitor activities, and demand trends.
    """
    try:
        current_time = datetime.now(timezone.utc)
        
        # Simulate active market events
        active_events = [
            MarketEvent(
                event_type=EventType.DEMAND_SPIKE,
                affected_products=["SKU123456", "SKU789012"],
                impact_magnitude=0.75,
                metadata={
                    "trigger": "social_media_trend",
                    "expected_duration_hours": 48
                },
                description="Viral social media post increased demand for coffee products"
            ),
            MarketEvent(
                event_type=EventType.COMPETITOR_PRICE_CHANGE,
                affected_products=["SKU345678"],
                impact_magnitude=0.45,
                metadata={
                    "competitor": "Competitor A",
                    "price_change_percentage": -15
                },
                description="Major competitor reduced chocolate prices by 15%"
            )
        ]
        
        # Simulate market indicators
        market_indicators = {
            "consumer_confidence_index": 78.5,
            "retail_sales_growth": 2.3,
            "inflation_rate": 3.1,
            "unemployment_rate": 4.2,
            "seasonal_factor": 1.15  # Holiday season boost
        }
        
        # Simulate competitor activity
        competitor_activity = {
            "Competitor A": "Launched flash sale on confectionery items",
            "Competitor B": "Introduced new premium coffee line",
            "Competitor C": "Expanded organic product selection"
        }
        
        # Simulate demand trends by category
        demand_trends = {
            "Beverages": 12.5,  # % change from last period
            "Confectionery": -5.2,
            "Food": 8.1,
            "Organic": 18.7
        }
        
        return MarketDataResponse(
            timestamp=current_time,
            active_events=active_events,
            market_indicators=market_indicators,
            competitor_activity=competitor_activity,
            demand_trends=demand_trends
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve market data: {str(e)}"
        )


@router.get("/system-metrics", response_model=SystemMetrics)
async def get_system_metrics() -> SystemMetrics:
    """
    Get current system performance metrics.
    
    This endpoint provides agents with system-wide performance indicators
    for monitoring overall optimization effectiveness.
    """
    try:
        # Simulate system metrics calculation
        # In real implementation: metrics = await calculate_system_metrics()
        
        return SystemMetrics(
            total_revenue=125000.50,
            total_profit=45000.25,
            inventory_turnover=8.5,
            stockout_incidents=3,
            waste_reduction_percentage=15.2,
            price_optimization_score=0.87,
            promotion_effectiveness=0.72,
            agent_collaboration_score=0.91,
            decision_count=247,
            response_time_avg=1.35
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )