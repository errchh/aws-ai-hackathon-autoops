"""
Comprehensive tests for FastAPI execution layer endpoints.

This module tests all API endpoints with various input scenarios to ensure
proper functionality and error handling.
"""

import pytest
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app
from models.core import ActionType, EventType


# Test client setup
client = TestClient(app)


class TestRootEndpoints:
    """Test root and health endpoints."""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AutoOps Retail Optimization API"
        assert "version" in data
        assert data["status"] == "operational"
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "autoops-retail-api"


class TestPricingEndpoints:
    """Test pricing-related API endpoints."""
    
    def test_update_price_success(self):
        """Test successful price update."""
        request_data = {
            "product_id": "SKU123456",
            "new_price": 21.99,
            "reason": "Market competition adjustment",
            "agent_id": "pricing_agent"
        }
        
        response = client.post("/api/pricing/update-price", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["product_id"] == "SKU123456"
        assert data["new_price"] == 21.99
        assert "decision_id" in data
        assert "timestamp" in data
    
    def test_update_price_invalid_data(self):
        """Test price update with invalid data."""
        request_data = {
            "product_id": "SKU123456",
            "new_price": -5.00,  # Invalid negative price
            "reason": "Test",
            "agent_id": "pricing_agent"
        }
        
        response = client.post("/api/pricing/update-price", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_update_price_large_change(self):
        """Test price update with unreasonably large change."""
        request_data = {
            "product_id": "SKU123456",
            "new_price": 100.00,  # Very large price change
            "reason": "Testing large change",
            "agent_id": "pricing_agent"
        }
        
        response = client.post("/api/pricing/update-price", json=request_data)
        assert response.status_code == 400  # Should reject large changes
    
    def test_apply_markdown_success(self):
        """Test successful markdown application."""
        request_data = {
            "product_id": "SKU123456",
            "discount_percentage": 15.0,
            "duration_hours": 48,
            "reason": "Slow-moving inventory clearance",
            "agent_id": "pricing_agent"
        }
        
        response = client.post("/api/pricing/apply-markdown", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["product_id"] == "SKU123456"
        assert data["discount_percentage"] == 15.0
        assert data["markdown_price"] < data["original_price"]
        assert "expires_at" in data
    
    def test_apply_markdown_excessive_discount(self):
        """Test markdown with excessive discount percentage."""
        request_data = {
            "product_id": "SKU123456",
            "discount_percentage": 80.0,  # Excessive discount
            "reason": "Testing excessive discount",
            "agent_id": "pricing_agent"
        }
        
        response = client.post("/api/pricing/apply-markdown", json=request_data)
        assert response.status_code == 400  # Should reject excessive discounts
    
    def test_competitor_analysis(self):
        """Test competitor pricing analysis."""
        response = client.get("/api/pricing/competitor-analysis/SKU123456")
        assert response.status_code == 200
        
        data = response.json()
        assert data["product_id"] == "SKU123456"
        assert "current_price" in data
        assert "competitor_prices" in data
        assert "market_position" in data
        assert data["market_position"] in ["lowest", "average", "highest"]
        assert len(data["competitor_prices"]) > 0
    
    def test_price_impact_analysis(self):
        """Test price impact analysis."""
        response = client.get(
            "/api/pricing/price-impact/SKU123456",
            params={"proposed_price": 20.99, "current_price": 24.99}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["product_id"] == "SKU123456"
        assert "price_change_percentage" in data
        assert "expected_demand_change" in data
        assert "expected_revenue_impact" in data
        assert "elasticity_coefficient" in data


class TestInventoryEndpoints:
    """Test inventory-related API endpoints."""
    
    def test_update_stock_success(self):
        """Test successful stock update."""
        request_data = {
            "product_id": "SKU123456",
            "new_stock_level": 200,
            "reason": "New delivery received",
            "agent_id": "inventory_agent",
            "source": "delivery"
        }
        
        response = client.post("/api/inventory/update-stock", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["product_id"] == "SKU123456"
        assert data["new_stock"] == 200
        assert "decision_id" in data
    
    def test_update_stock_large_change(self):
        """Test stock update with unreasonably large change."""
        request_data = {
            "product_id": "SKU123456",
            "new_stock_level": 5000,  # Very large stock change
            "reason": "Testing large change",
            "agent_id": "inventory_agent"
        }
        
        response = client.post("/api/inventory/update-stock", json=request_data)
        assert response.status_code == 400  # Should reject large changes
    
    def test_create_restock_alert_success(self):
        """Test successful restock alert creation."""
        request_data = {
            "product_id": "SKU789012",
            "current_stock": 8,
            "recommended_quantity": 100,
            "urgency": "high",
            "reason": "Stock below safety threshold",
            "agent_id": "inventory_agent"
        }
        
        response = client.post("/api/inventory/create-restock-alert", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["product_id"] == "SKU789012"
        assert data["recommended_quantity"] == 100
        assert "estimated_cost" in data
        assert "expected_delivery" in data
    
    def test_create_restock_alert_excessive_quantity(self):
        """Test restock alert with excessive quantity."""
        request_data = {
            "product_id": "SKU789012",
            "current_stock": 8,
            "recommended_quantity": 50000,  # Excessive quantity
            "urgency": "high",
            "reason": "Testing excessive quantity",
            "agent_id": "inventory_agent"
        }
        
        response = client.post("/api/inventory/create-restock-alert", json=request_data)
        assert response.status_code == 400  # Should reject excessive quantities
    
    def test_demand_forecast(self):
        """Test demand forecasting."""
        request_data = {
            "product_id": "SKU123456",
            "forecast_days": 30,
            "include_seasonality": True,
            "include_trends": True
        }
        
        response = client.post("/api/inventory/demand-forecast", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["product_id"] == "SKU123456"
        assert data["forecast_period_days"] == 30
        assert "total_expected_demand" in data
        assert "forecast_points" in data
        assert len(data["forecast_points"]) == 30
        assert data["trend_direction"] in ["increasing", "stable", "decreasing"]
    
    def test_inventory_analysis(self):
        """Test inventory analysis."""
        response = client.get("/api/inventory/analysis/SKU123456")
        assert response.status_code == 200
        
        data = response.json()
        assert data["product_id"] == "SKU123456"
        assert "current_stock" in data
        assert "reorder_point" in data
        assert "safety_stock" in data
        assert "stockout_risk" in data
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0


class TestPromotionEndpoints:
    """Test promotion-related API endpoints."""
    
    def test_create_campaign_success(self):
        """Test successful campaign creation."""
        start_time = datetime.now(timezone.utc) + timedelta(hours=1)
        end_time = start_time + timedelta(days=7)
        
        request_data = {
            "campaign_name": "Flash Sale Coffee",
            "campaign_type": "flash_sale",
            "product_ids": ["SKU123456", "SKU789012"],
            "discount_percentage": 20.0,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "target_audience": "coffee_lovers",
            "budget_limit": 5000.0,
            "agent_id": "promotion_agent"
        }
        
        response = client.post("/api/promotions/create-campaign", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["campaign_name"] == "Flash Sale Coffee"
        assert len(data["affected_products"]) == 2
        assert "estimated_impact" in data
        assert "campaign_id" in data
    
    def test_create_campaign_invalid_timing(self):
        """Test campaign creation with invalid timing."""
        start_time = datetime.now(timezone.utc) + timedelta(hours=1)
        end_time = start_time - timedelta(hours=2)  # End before start
        
        request_data = {
            "campaign_name": "Invalid Campaign",
            "campaign_type": "flash_sale",
            "product_ids": ["SKU123456"],
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "agent_id": "promotion_agent"
        }
        
        response = client.post("/api/promotions/create-campaign", json=request_data)
        assert response.status_code == 400  # Should reject invalid timing
    
    def test_create_bundle_success(self):
        """Test successful bundle creation."""
        request_data = {
            "bundle_name": "Coffee & Tea Bundle",
            "anchor_product_id": "SKU123456",
            "complementary_product_ids": ["SKU789012", "SKU345678"],
            "bundle_discount_percentage": 15.0,
            "minimum_quantity": 1,
            "agent_id": "promotion_agent"
        }
        
        response = client.post("/api/promotions/create-bundle", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert data["bundle_name"] == "Coffee & Tea Bundle"
        assert data["total_products"] == 3
        assert data["bundle_price"] < data["original_price"]
        assert data["savings_amount"] > 0
    
    def test_create_bundle_duplicate_products(self):
        """Test bundle creation with duplicate products."""
        request_data = {
            "bundle_name": "Invalid Bundle",
            "anchor_product_id": "SKU123456",
            "complementary_product_ids": ["SKU123456", "SKU789012"],  # Duplicate anchor
            "bundle_discount_percentage": 15.0,
            "agent_id": "promotion_agent"
        }
        
        response = client.post("/api/promotions/create-bundle", json=request_data)
        assert response.status_code == 400  # Should reject duplicate products
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis."""
        request_data = {
            "product_category": "Beverages",
            "keywords": ["coffee", "premium"],
            "time_period_hours": 24,
            "platforms": ["twitter", "instagram"]
        }
        
        response = client.post("/api/promotions/sentiment-analysis", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "overall_sentiment" in data
        assert -1 <= data["overall_sentiment"] <= 1
        assert "sentiment_trend" in data
        assert data["sentiment_trend"] in ["improving", "stable", "declining"]
        assert "platform_data" in data
        assert len(data["platform_data"]) == 2  # twitter and instagram
    
    def test_campaign_performance(self):
        """Test campaign performance retrieval."""
        campaign_id = str(uuid4())
        response = client.get(f"/api/promotions/campaign-performance/{campaign_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["campaign_id"] == campaign_id
        assert "impressions" in data
        assert "clicks" in data
        assert "conversions" in data
        assert "roi" in data
        assert data["impressions"] >= data["clicks"] >= data["conversions"]


class TestSharedEndpoints:
    """Test shared data API endpoints."""
    
    def test_get_products_default(self):
        """Test getting products with default parameters."""
        response = client.get("/api/products")
        assert response.status_code == 200
        
        data = response.json()
        assert "products" in data
        assert "total_count" in data
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert len(data["products"]) <= 50
    
    def test_get_products_with_filters(self):
        """Test getting products with filters."""
        response = client.get(
            "/api/products",
            params={
                "category": "Beverages",
                "min_price": 15.0,
                "max_price": 30.0,
                "low_stock_only": True
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        # Verify filters are applied (in real implementation)
        assert "products" in data
    
    def test_get_product_by_id(self):
        """Test getting a specific product by ID."""
        response = client.get("/api/products/SKU123456")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == "SKU123456"
        assert "name" in data
        assert "current_price" in data
        assert "inventory_level" in data
    
    def test_get_product_not_found(self):
        """Test getting a non-existent product."""
        response = client.get("/api/products/NONEXISTENT")
        assert response.status_code == 404
    
    def test_search_products(self):
        """Test product search functionality."""
        response = client.get(
            "/api/products/search",
            params={"query": "coffee", "limit": 10}
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["search_query"] == "coffee"
        assert "products" in data
        assert "search_time_ms" in data
        assert len(data["products"]) <= 10
    
    def test_get_market_data(self):
        """Test getting current market data."""
        response = client.get("/api/market-data/current")
        assert response.status_code == 200
        
        data = response.json()
        assert "timestamp" in data
        assert "active_events" in data
        assert "market_indicators" in data
        assert "competitor_activity" in data
        assert "demand_trends" in data
    
    def test_get_system_metrics(self):
        """Test getting system metrics."""
        response = client.get("/api/system-metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_revenue" in data
        assert "total_profit" in data
        assert "inventory_turnover" in data
        assert "agent_collaboration_score" in data
        assert 0 <= data["agent_collaboration_score"] <= 1


class TestDecisionEndpoints:
    """Test decision logging API endpoints."""
    
    def test_log_decision_success(self):
        """Test successful decision logging."""
        request_data = {
            "agent_id": "pricing_agent",
            "action_type": "price_adjustment",
            "parameters": {
                "product_id": "SKU123456",
                "new_price": 21.99,
                "previous_price": 24.99
            },
            "rationale": "Adjusting price to match competitor pricing",
            "confidence_score": 0.85,
            "expected_outcome": {
                "demand_increase": 15,
                "revenue_impact": 500
            },
            "context": {
                "competitor_average": 22.50,
                "inventory_level": 150
            }
        }
        
        response = client.post("/api/decisions/log", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "decision_id" in data
        assert "timestamp" in data
        assert "storage_location" in data
    
    def test_log_decision_invalid_data(self):
        """Test decision logging with invalid data."""
        request_data = {
            "agent_id": "",  # Empty agent ID
            "action_type": "price_adjustment",
            "parameters": {},  # Empty parameters
            "rationale": "Short",  # Too short rationale
            "confidence_score": 1.5,  # Invalid confidence score
            "expected_outcome": {}  # Empty expected outcome
        }
        
        response = client.post("/api/decisions/log", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_query_decisions(self):
        """Test querying historical decisions."""
        response = client.get(
            "/api/decisions/query",
            params={
                "agent_id": "pricing_agent",
                "action_type": "price_adjustment",
                "limit": 10
            }
        )
        assert response.status_code == 200
        
        data = response.json()
        assert "decisions" in data
        assert "total_count" in data
        assert "query_time_ms" in data
        assert len(data["decisions"]) <= 10
    
    def test_find_similar_decisions(self):
        """Test finding similar decisions."""
        request_data = {
            "current_context": {
                "product_category": "Beverages",
                "inventory_level": 150,
                "competitor_prices": [22.99, 24.49]
            },
            "agent_id": "pricing_agent",
            "similarity_threshold": 0.7,
            "max_results": 5
        }
        
        response = client.post("/api/decisions/similar", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "similar_decisions" in data
        assert "search_context" in data
        assert len(data["similar_decisions"]) <= 5
        
        # Verify similarity scores are above threshold
        for result in data["similar_decisions"]:
            assert result["similarity_score"] >= 0.7
    
    def test_get_decision_summary(self):
        """Test getting decision summary for an agent."""
        response = client.get("/api/decisions/summary/pricing_agent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["agent_id"] == "pricing_agent"
        assert "total_decisions" in data
        assert "average_confidence" in data
        assert "action_type_distribution" in data
        assert "recent_activity" in data
        assert 0 <= data["average_confidence"] <= 1
    
    def test_delete_decision(self):
        """Test deleting a decision."""
        decision_id = str(uuid4())
        response = client.delete(f"/api/decisions/{decision_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert decision_id in data["message"]


class TestErrorHandling:
    """Test error handling across all endpoints."""
    
    def test_invalid_json(self):
        """Test handling of invalid JSON data."""
        response = client.post(
            "/api/pricing/update-price",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        response = client.post("/api/pricing/update-price", json={})
        assert response.status_code == 422
    
    def test_invalid_uuid_format(self):
        """Test handling of invalid UUID format."""
        response = client.get("/api/promotions/campaign-performance/invalid-uuid")
        assert response.status_code == 422
    
    def test_nonexistent_endpoints(self):
        """Test handling of non-existent endpoints."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])