"""
Unit tests for core data models and validation.

This module tests all data models (Product, MarketEvent, AgentDecision, 
CollaborationRequest, SystemMetrics) and their validation functions.
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import UUID
from pydantic import ValidationError

from models import (
    Product,
    MarketEvent,
    AgentDecision,
    CollaborationRequest,
    SystemMetrics,
    EventType,
    ActionType,
    RequestType,
    UrgencyLevel,
    ValidationResult,
    validate_product_data,
    validate_market_event_data,
    validate_agent_decision_data,
    validate_collaboration_request_data,
    validate_system_metrics_data,
    validate_batch_data,
)


class TestProduct:
    """Test cases for Product model."""
    
    def test_valid_product_creation(self):
        """Test creating a valid product."""
        product_data = {
            "id": "SKU123456",
            "name": "Premium Coffee Beans 1kg",
            "category": "Beverages",
            "base_price": 24.99,
            "current_price": 22.99,
            "cost": 12.50,
            "inventory_level": 150,
            "reorder_point": 25,
            "supplier_lead_time": 7
        }
        
        product = Product(**product_data)
        
        assert product.id == "SKU123456"
        assert product.name == "Premium Coffee Beans 1kg"
        assert product.category == "Beverages"
        assert product.base_price == 24.99
        assert product.current_price == 22.99
        assert product.cost == 12.50
        assert product.inventory_level == 150
        assert product.reorder_point == 25
        assert product.supplier_lead_time == 7
    
    def test_product_price_validation(self):
        """Test product price validation rules."""
        # Test current price too low compared to cost
        with pytest.raises(ValidationError) as exc_info:
            Product(
                id="SKU123",
                name="Test Product",
                category="Test",
                base_price=20.0,
                current_price=5.0,  # Less than 50% of cost
                cost=12.0,
                inventory_level=100,
                reorder_point=10,
                supplier_lead_time=5
            )
        
        assert "Current price cannot be less than 50% of cost" in str(exc_info.value)
    
    def test_product_reorder_point_validation(self):
        """Test reorder point validation."""
        # Test reorder point too high
        with pytest.raises(ValidationError) as exc_info:
            Product(
                id="SKU123",
                name="Test Product", 
                category="Test",
                base_price=20.0,
                current_price=18.0,
                cost=10.0,
                inventory_level=50,
                reorder_point=150,  # More than twice inventory
                supplier_lead_time=5
            )
        
        assert "Reorder point cannot exceed twice the current inventory" in str(exc_info.value)
    
    def test_product_negative_values(self):
        """Test that negative values are rejected."""
        with pytest.raises(ValidationError):
            Product(
                id="SKU123",
                name="Test Product",
                category="Test", 
                base_price=-10.0,  # Negative price
                current_price=15.0,
                cost=8.0,
                inventory_level=50,
                reorder_point=10,
                supplier_lead_time=5
            )
    
    def test_product_serialization(self):
        """Test product JSON serialization."""
        product = Product(
            id="SKU123",
            name="Test Product",
            category="Test",
            base_price=20.0,
            current_price=18.0,
            cost=10.0,
            inventory_level=50,
            reorder_point=10,
            supplier_lead_time=5
        )
        
        json_data = product.model_dump()
        assert json_data["id"] == "SKU123"
        assert json_data["current_price"] == 18.0
        
        # Test round-trip serialization
        new_product = Product(**json_data)
        assert new_product.id == product.id
        assert new_product.current_price == product.current_price


class TestMarketEvent:
    """Test cases for MarketEvent model."""
    
    def test_valid_market_event_creation(self):
        """Test creating a valid market event."""
        event_data = {
            "event_type": EventType.DEMAND_SPIKE,
            "affected_products": ["SKU123", "SKU456"],
            "impact_magnitude": 0.75,
            "metadata": {"trigger": "social_media_trend"},
            "description": "Viral post increased demand"
        }
        
        event = MarketEvent(**event_data)
        
        assert event.event_type == EventType.DEMAND_SPIKE
        assert event.affected_products == ["SKU123", "SKU456"]
        assert event.impact_magnitude == 0.75
        assert event.metadata["trigger"] == "social_media_trend"
        assert isinstance(event.id, UUID)
        assert isinstance(event.timestamp, datetime)
    
    def test_market_event_empty_products(self):
        """Test that empty affected products list is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            MarketEvent(
                event_type=EventType.DEMAND_SPIKE,
                affected_products=[],  # Empty list
                impact_magnitude=0.5
            )
        
        assert "At least one product must be affected by the event" in str(exc_info.value)
    
    def test_market_event_impact_magnitude_bounds(self):
        """Test impact magnitude bounds validation."""
        # Test negative impact
        with pytest.raises(ValidationError):
            MarketEvent(
                event_type=EventType.DEMAND_SPIKE,
                affected_products=["SKU123"],
                impact_magnitude=-0.1  # Negative
            )
        
        # Test impact > 1.0
        with pytest.raises(ValidationError):
            MarketEvent(
                event_type=EventType.DEMAND_SPIKE,
                affected_products=["SKU123"],
                impact_magnitude=1.5  # > 1.0
            )
    
    def test_market_event_enum_values(self):
        """Test that only valid enum values are accepted."""
        with pytest.raises(ValidationError):
            MarketEvent(
                event_type="invalid_event_type",  # Invalid enum
                affected_products=["SKU123"],
                impact_magnitude=0.5
            )


class TestAgentDecision:
    """Test cases for AgentDecision model."""
    
    def test_valid_agent_decision_creation(self):
        """Test creating a valid agent decision."""
        decision_data = {
            "agent_id": "pricing_agent",
            "action_type": ActionType.PRICE_ADJUSTMENT,
            "parameters": {
                "product_id": "SKU123",
                "new_price": 19.99,
                "previous_price": 22.99
            },
            "rationale": "Reducing price to increase demand and clear inventory",
            "confidence_score": 0.85,
            "expected_outcome": {
                "demand_increase_percentage": 25,
                "inventory_turnover_days": 14
            }
        }
        
        decision = AgentDecision(**decision_data)
        
        assert decision.agent_id == "pricing_agent"
        assert decision.action_type == ActionType.PRICE_ADJUSTMENT
        assert decision.parameters["product_id"] == "SKU123"
        assert decision.confidence_score == 0.85
        assert isinstance(decision.id, UUID)
        assert isinstance(decision.timestamp, datetime)
    
    def test_agent_decision_empty_parameters(self):
        """Test that empty parameters are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            AgentDecision(
                agent_id="pricing_agent",
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={},  # Empty parameters
                rationale="Test rationale",
                confidence_score=0.8,
                expected_outcome={"test": "value"}
            )
        
        assert "Parameters cannot be empty" in str(exc_info.value)
    
    def test_agent_decision_confidence_bounds(self):
        """Test confidence score bounds validation."""
        # Test negative confidence
        with pytest.raises(ValidationError):
            AgentDecision(
                agent_id="pricing_agent",
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={"test": "value"},
                rationale="Test rationale",
                confidence_score=-0.1,  # Negative
                expected_outcome={"test": "value"}
            )
        
        # Test confidence > 1.0
        with pytest.raises(ValidationError):
            AgentDecision(
                agent_id="pricing_agent",
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={"test": "value"},
                rationale="Test rationale",
                confidence_score=1.5,  # > 1.0
                expected_outcome={"test": "value"}
            )
    
    def test_agent_decision_short_rationale(self):
        """Test that short rationale is rejected."""
        with pytest.raises(ValidationError):
            AgentDecision(
                agent_id="pricing_agent",
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={"test": "value"},
                rationale="Short",  # Too short
                confidence_score=0.8,
                expected_outcome={"test": "value"}
            )


class TestCollaborationRequest:
    """Test cases for CollaborationRequest model."""
    
    def test_valid_collaboration_request_creation(self):
        """Test creating a valid collaboration request."""
        request_data = {
            "requesting_agent": "inventory_agent",
            "target_agent": "pricing_agent",
            "request_type": RequestType.COORDINATION,
            "context": {
                "product_id": "SKU123",
                "current_inventory": 150,
                "recommended_action": "markdown"
            },
            "urgency": UrgencyLevel.MEDIUM,
            "message": "Product has been slow-moving. Recommend price reduction."
        }
        
        request = CollaborationRequest(**request_data)
        
        assert request.requesting_agent == "inventory_agent"
        assert request.target_agent == "pricing_agent"
        assert request.request_type == RequestType.COORDINATION
        assert request.urgency == UrgencyLevel.MEDIUM
        assert isinstance(request.id, UUID)
        assert isinstance(request.timestamp, datetime)
    
    def test_collaboration_request_same_agents(self):
        """Test that same requesting and target agents are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            CollaborationRequest(
                requesting_agent="pricing_agent",
                target_agent="pricing_agent",  # Same as requesting
                request_type=RequestType.CONSULTATION,
                context={"test": "value"},
                urgency=UrgencyLevel.LOW,
                message="Test message for collaboration"
            )
        
        assert "Requesting and target agents must be different" in str(exc_info.value)
    
    def test_collaboration_request_past_deadline(self):
        """Test that past deadline is rejected."""
        past_time = datetime.now(timezone.utc) - timedelta(hours=1)
        
        with pytest.raises(ValidationError) as exc_info:
            CollaborationRequest(
                requesting_agent="inventory_agent",
                target_agent="pricing_agent",
                request_type=RequestType.COORDINATION,
                context={"test": "value"},
                urgency=UrgencyLevel.MEDIUM,
                message="Test message for collaboration",
                response_deadline=past_time  # Past deadline
            )
        
        assert "Response deadline must be in the future" in str(exc_info.value)


class TestSystemMetrics:
    """Test cases for SystemMetrics model."""
    
    def test_valid_system_metrics_creation(self):
        """Test creating valid system metrics."""
        metrics_data = {
            "total_revenue": 125000.50,
            "total_profit": 45000.25,
            "inventory_turnover": 8.5,
            "stockout_incidents": 3,
            "waste_reduction_percentage": 15.2,
            "price_optimization_score": 0.87,
            "promotion_effectiveness": 0.72,
            "agent_collaboration_score": 0.91,
            "decision_count": 247,
            "response_time_avg": 1.35
        }
        
        metrics = SystemMetrics(**metrics_data)
        
        assert metrics.total_revenue == 125000.50
        assert metrics.total_profit == 45000.25
        assert metrics.inventory_turnover == 8.5
        assert metrics.price_optimization_score == 0.87
        assert isinstance(metrics.id, UUID)
        assert isinstance(metrics.timestamp, datetime)
    
    def test_system_metrics_profit_exceeds_revenue(self):
        """Test that profit exceeding revenue is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SystemMetrics(
                total_revenue=100000.0,
                total_profit=150000.0,  # Exceeds revenue
                inventory_turnover=5.0,
                stockout_incidents=2,
                waste_reduction_percentage=10.0,
                price_optimization_score=0.8,
                promotion_effectiveness=0.7,
                agent_collaboration_score=0.9,
                decision_count=100,
                response_time_avg=2.0
            )
        
        assert "Profit cannot exceed revenue" in str(exc_info.value)
    
    def test_system_metrics_negative_values(self):
        """Test that negative values are handled correctly."""
        # Negative profit should be allowed (losses)
        metrics = SystemMetrics(
            total_revenue=100000.0,
            total_profit=-10000.0,  # Negative profit (loss)
            inventory_turnover=5.0,
            stockout_incidents=2,
            waste_reduction_percentage=10.0,
            price_optimization_score=0.8,
            promotion_effectiveness=0.7,
            agent_collaboration_score=0.9,
            decision_count=100,
            response_time_avg=2.0
        )
        
        assert metrics.total_profit == -10000.0
        
        # But negative revenue should be rejected
        with pytest.raises(ValidationError):
            SystemMetrics(
                total_revenue=-100000.0,  # Negative revenue
                total_profit=10000.0,
                inventory_turnover=5.0,
                stockout_incidents=2,
                waste_reduction_percentage=10.0,
                price_optimization_score=0.8,
                promotion_effectiveness=0.7,
                agent_collaboration_score=0.9,
                decision_count=100,
                response_time_avg=2.0
            )


class TestValidationFunctions:
    """Test cases for validation utility functions."""
    
    def test_validate_product_data_success(self):
        """Test successful product validation."""
        product_data = {
            "id": "SKU123",
            "name": "Test Product",
            "category": "Test",
            "base_price": 20.0,
            "current_price": 18.0,
            "cost": 10.0,
            "inventory_level": 50,
            "reorder_point": 10,
            "supplier_lead_time": 5
        }
        
        result = validate_product_data(product_data)
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert bool(result) is True
    
    def test_validate_product_data_failure(self):
        """Test product validation failure."""
        product_data = {
            "id": "SKU123",
            "name": "Test Product",
            "category": "Test",
            "base_price": 20.0,
            "current_price": 8.0,  # Less than cost
            "cost": 10.0,
            "inventory_level": 50,
            "reorder_point": 10,
            "supplier_lead_time": 5
        }
        
        result = validate_product_data(product_data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "Current price must be greater than cost" in result.errors
    
    def test_validate_agent_decision_data_agent_action_mismatch(self):
        """Test agent decision validation with mismatched agent and action."""
        decision_data = {
            "agent_id": "pricing_agent",
            "action_type": ActionType.INVENTORY_RESTOCK,  # Wrong action for pricing agent
            "parameters": {"product_id": "SKU123", "quantity": 100},
            "rationale": "Test rationale for validation",
            "confidence_score": 0.8,
            "expected_outcome": {"test": "value"}
        }
        
        result = validate_agent_decision_data(decision_data)
        
        assert not result.is_valid
        assert any("Pricing agent cannot perform action" in error for error in result.errors)
    
    def test_validate_batch_data(self):
        """Test batch validation functionality."""
        product_data_list = [
            {
                "id": "SKU123",
                "name": "Product 1",
                "category": "Test",
                "base_price": 20.0,
                "current_price": 18.0,
                "cost": 10.0,
                "inventory_level": 50,
                "reorder_point": 10,
                "supplier_lead_time": 5
            },
            {
                "id": "SKU456",
                "name": "Product 2",
                "category": "Test",
                "base_price": 15.0,
                "current_price": 5.0,  # Invalid - too low
                "cost": 8.0,
                "inventory_level": 30,
                "reorder_point": 5,
                "supplier_lead_time": 3
            }
        ]
        
        results = validate_batch_data(product_data_list, "product")
        
        assert len(results) == 2
        assert results[0].is_valid  # First product is valid
        assert not results[1].is_valid  # Second product is invalid
    
    def test_validation_result_string_representation(self):
        """Test ValidationResult string representation."""
        # Test successful validation
        success_result = ValidationResult(True)
        assert str(success_result) == "Validation passed"
        
        # Test failed validation
        error_result = ValidationResult(False, ["Error 1", "Error 2"])
        assert "Validation failed: Error 1; Error 2" in str(error_result)
    
    def test_validate_batch_data_invalid_model_type(self):
        """Test batch validation with invalid model type."""
        with pytest.raises(ValueError) as exc_info:
            validate_batch_data([{}], "invalid_model_type")
        
        assert "Unknown model type: invalid_model_type" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])