#!/usr/bin/env python3
"""
Demo script for the FastAPI execution layer.

This script demonstrates the key API endpoints that agents will use
to execute actions in the retail optimization system.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from uuid import uuid4

import httpx


API_BASE_URL = "http://localhost:8000"


async def demo_pricing_endpoints():
    """Demonstrate pricing-related endpoints."""
    print("\n=== PRICING ENDPOINTS DEMO ===")
    
    async with httpx.AsyncClient() as client:
        # 1. Update product price
        print("\n1. Updating product price...")
        price_update = {
            "product_id": "SKU123456",
            "new_price": 21.99,
            "reason": "Competitive pricing adjustment based on market analysis",
            "agent_id": "pricing_agent"
        }
        
        response = await client.post(f"{API_BASE_URL}/api/pricing/update-price", json=price_update)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 2. Apply markdown
        print("\n2. Applying markdown to slow-moving inventory...")
        markdown_request = {
            "product_id": "SKU789012",
            "discount_percentage": 20.0,
            "duration_hours": 72,
            "reason": "Clearing slow-moving inventory to improve turnover",
            "agent_id": "pricing_agent"
        }
        
        response = await client.post(f"{API_BASE_URL}/api/pricing/apply-markdown", json=markdown_request)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 3. Get competitor analysis
        print("\n3. Getting competitor analysis...")
        response = await client.get(f"{API_BASE_URL}/api/pricing/competitor-analysis/SKU123456")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")


async def demo_inventory_endpoints():
    """Demonstrate inventory-related endpoints."""
    print("\n=== INVENTORY ENDPOINTS DEMO ===")
    
    async with httpx.AsyncClient() as client:
        # 1. Update stock level
        print("\n1. Updating stock level...")
        stock_update = {
            "product_id": "SKU123456",
            "new_stock_level": 180,
            "reason": "New delivery received from supplier",
            "agent_id": "inventory_agent",
            "source": "delivery"
        }
        
        response = await client.post(f"{API_BASE_URL}/api/inventory/update-stock", json=stock_update)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 2. Create restock alert
        print("\n2. Creating restock alert...")
        restock_alert = {
            "product_id": "SKU789012",
            "current_stock": 8,
            "recommended_quantity": 150,
            "urgency": "high",
            "reason": "Stock level below safety threshold, high demand forecast",
            "agent_id": "inventory_agent"
        }
        
        response = await client.post(f"{API_BASE_URL}/api/inventory/create-restock-alert", json=restock_alert)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 3. Generate demand forecast
        print("\n3. Generating demand forecast...")
        forecast_request = {
            "product_id": "SKU123456",
            "forecast_days": 14,
            "include_seasonality": True,
            "include_trends": True
        }
        
        response = await client.post(f"{API_BASE_URL}/api/inventory/demand-forecast", json=forecast_request)
        print(f"Status: {response.status_code}")
        forecast_data = response.json()
        print(f"Total expected demand: {forecast_data['total_expected_demand']}")
        print(f"Average daily demand: {forecast_data['average_daily_demand']}")
        print(f"Trend direction: {forecast_data['trend_direction']}")


async def demo_promotion_endpoints():
    """Demonstrate promotion-related endpoints."""
    print("\n=== PROMOTION ENDPOINTS DEMO ===")
    
    async with httpx.AsyncClient() as client:
        # 1. Create promotional campaign
        print("\n1. Creating promotional campaign...")
        start_time = datetime.now(timezone.utc) + timedelta(hours=1)
        end_time = start_time + timedelta(days=3)
        
        campaign_request = {
            "campaign_name": "Weekend Coffee Special",
            "campaign_type": "flash_sale",
            "product_ids": ["SKU123456", "SKU789012"],
            "discount_percentage": 15.0,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "target_audience": "coffee_enthusiasts",
            "budget_limit": 2000.0,
            "agent_id": "promotion_agent"
        }
        
        response = await client.post(f"{API_BASE_URL}/api/promotions/create-campaign", json=campaign_request)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 2. Create product bundle
        print("\n2. Creating product bundle...")
        bundle_request = {
            "bundle_name": "Coffee Lover's Bundle",
            "anchor_product_id": "SKU123456",
            "complementary_product_ids": ["SKU789012", "SKU345678"],
            "bundle_discount_percentage": 12.0,
            "minimum_quantity": 1,
            "agent_id": "promotion_agent"
        }
        
        response = await client.post(f"{API_BASE_URL}/api/promotions/create-bundle", json=bundle_request)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 3. Analyze sentiment
        print("\n3. Analyzing social media sentiment...")
        sentiment_request = {
            "product_category": "Beverages",
            "keywords": ["coffee", "premium", "organic"],
            "time_period_hours": 24,
            "platforms": ["twitter", "instagram", "facebook"]
        }
        
        response = await client.post(f"{API_BASE_URL}/api/promotions/sentiment-analysis", json=sentiment_request)
        print(f"Status: {response.status_code}")
        sentiment_data = response.json()
        print(f"Overall sentiment: {sentiment_data['overall_sentiment']}")
        print(f"Sentiment trend: {sentiment_data['sentiment_trend']}")
        print(f"Total mentions: {sentiment_data['total_mentions']}")


async def demo_shared_endpoints():
    """Demonstrate shared data endpoints."""
    print("\n=== SHARED DATA ENDPOINTS DEMO ===")
    
    async with httpx.AsyncClient() as client:
        # 1. Get products
        print("\n1. Getting product list...")
        response = await client.get(f"{API_BASE_URL}/api/products", params={"limit": 5})
        print(f"Status: {response.status_code}")
        products_data = response.json()
        print(f"Total products: {products_data['total_count']}")
        print(f"Products returned: {len(products_data['products'])}")
        
        # 2. Get specific product
        print("\n2. Getting specific product...")
        response = await client.get(f"{API_BASE_URL}/api/products/SKU123456")
        print(f"Status: {response.status_code}")
        product_data = response.json()
        print(f"Product: {product_data['name']}")
        print(f"Current price: ${product_data['current_price']}")
        print(f"Inventory level: {product_data['inventory_level']}")
        
        # 3. Get market data
        print("\n3. Getting current market data...")
        response = await client.get(f"{API_BASE_URL}/api/market-data/current")
        print(f"Status: {response.status_code}")
        market_data = response.json()
        print(f"Active events: {len(market_data['active_events'])}")
        print(f"Market indicators: {list(market_data['market_indicators'].keys())}")


async def demo_decision_endpoints():
    """Demonstrate decision logging endpoints."""
    print("\n=== DECISION LOGGING ENDPOINTS DEMO ===")
    
    async with httpx.AsyncClient() as client:
        # 1. Log a decision
        print("\n1. Logging agent decision...")
        decision_request = {
            "agent_id": "pricing_agent",
            "action_type": "price_adjustment",
            "parameters": {
                "product_id": "SKU123456",
                "new_price": 21.99,
                "previous_price": 24.99,
                "reason": "competitive_adjustment"
            },
            "rationale": "Adjusting price to match competitor average while maintaining profit margins",
            "confidence_score": 0.87,
            "expected_outcome": {
                "demand_increase_percentage": 18,
                "revenue_impact": 750.00,
                "margin_impact": -2.5
            },
            "context": {
                "competitor_average": 22.50,
                "inventory_level": 180,
                "demand_trend": "stable"
            }
        }
        
        response = await client.post(f"{API_BASE_URL}/api/decisions/log", json=decision_request)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # 2. Query decisions
        print("\n2. Querying historical decisions...")
        response = await client.get(
            f"{API_BASE_URL}/api/decisions/query",
            params={"agent_id": "pricing_agent", "limit": 3}
        )
        print(f"Status: {response.status_code}")
        decisions_data = response.json()
        print(f"Decisions found: {decisions_data['total_count']}")
        
        # 3. Get decision summary
        print("\n3. Getting decision summary...")
        response = await client.get(f"{API_BASE_URL}/api/decisions/summary/pricing_agent")
        print(f"Status: {response.status_code}")
        summary_data = response.json()
        print(f"Total decisions: {summary_data['total_decisions']}")
        print(f"Average confidence: {summary_data['average_confidence']}")
        print(f"Success rate: {summary_data['success_rate']}")


async def main():
    """Run all API demonstrations."""
    print("🚀 AutoOps Retail Optimization API Demo")
    print("=" * 50)
    
    try:
        # Check if API is running
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/health")
            if response.status_code != 200:
                print("❌ API is not running. Please start the API server first:")
                print("   uvicorn api.main:app --reload")
                return
            
            print("✅ API is running and healthy")
        
        # Run all demos
        await demo_pricing_endpoints()
        await demo_inventory_endpoints()
        await demo_promotion_endpoints()
        await demo_shared_endpoints()
        await demo_decision_endpoints()
        
        print("\n" + "=" * 50)
        print("✅ All API endpoints demonstrated successfully!")
        print("\nThe FastAPI execution layer is ready for agent integration.")
        
    except httpx.ConnectError:
        print("❌ Could not connect to API server.")
        print("Please start the API server first:")
        print("   uvicorn api.main:app --reload")
    except Exception as e:
        print(f"❌ Error during demo: {e}")


if __name__ == "__main__":
    asyncio.run(main())