"""
End-to-end tests for agent collaboration scenarios.

This module tests the collaboration workflows between Pricing, Inventory, 
and Promotion agents to ensure coordinated decision-making works correctly.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from uuid import uuid4

from agents.collaboration import CollaborationWorkflow, CollaborationType
from models.core import AgentDecision, ActionType


class TestAgentCollaboration:
    """Test suite for agent collaboration workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collaboration_workflow = CollaborationWorkflow()
        
        # Sample slow-moving items for testing
        self.sample_slow_moving_items = [
            {
                "product_id": "PROD001",
                "days_without_sale": 45,
                "inventory_value": 2500.0,
                "current_stock": 75,
                "urgency_score": 85
            },
            {
                "product_id": "PROD002", 
                "days_without_sale": 30,
                "inventory_value": 1200.0,
                "current_stock": 40,
                "urgency_score": 70
            }
        ]
        
        # Sample discount opportunities for testing
        self.sample_discount_opportunities = [
            {
                "product_id": "PROD003",
                "current_price": 50.0,
                "discount_percentage": 20,
                "reason": "slow_moving_inventory"
            },
            {
                "product_id": "PROD004",
                "current_price": 75.0,
                "discount_percentage": 15,
                "reason": "competitive_pressure"
            }
        ]
        
        # Sample campaign requests for testing
        self.sample_campaign_requests = [
            {
                "campaign_id": "CAMP001",
                "product_ids": ["PROD005", "PROD006"],
                "expected_demand_increase": 150,
                "duration_days": 7,
                "current_inventory": {"PROD005": 100, "PROD006": 80},
                "daily_demand": {"PROD005": 8, "PROD006": 6}
            }
        ]
        
        # Sample market event for testing
        self.sample_market_event = {
            "event_type": "demand_spike",
            "severity": "high",
            "affected_products": ["PROD007", "PROD008"],
            "expected_impact": "300% demand increase",
            "duration_estimate": "24-48 hours"
        }
    
    @pytest.mark.asyncio
    async def test_inventory_to_pricing_slow_moving_alert(self):
        """Test inventory-to-pricing collaboration for slow-moving items."""
        # Mock agent memory
        with patch('agents.collaboration.agent_memory') as mock_memory:
            mock_memory.store_decision.return_value = "memory_123"
            
            # Execute collaboration
            result = await self.collaboration_workflow.inventory_to_pricing_slow_moving_alert(
                slow_moving_items=self.sample_slow_moving_items
            )
            
            # Verify collaboration was initiated
            assert result["status"] == "initiated"
            assert result["type"] == CollaborationType.INVENTORY_TO_PRICING.value
            assert result["items_processed"] == 2
            assert result["pricing_recommendations"] == 2
            assert "collaboration_id" in result
            
            # Verify collaboration message structure
            message = result["collaboration_message"]
            assert message["message_type"] == "slow_moving_inventory_alert"
            assert message["from_agent"] == "inventory_agent"
            assert message["to_agent"] == "pricing_agent"
            assert len(message["pricing_recommendations"]) == 2
            
            # Verify pricing recommendations
            recommendations = message["pricing_recommendations"]
            assert all("recommended_markdown_percentage" in rec for rec in recommendations)
            assert all("rationale" in rec for rec in recommendations)
            
            # Verify memory storage was called
            mock_memory.store_decision.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pricing_to_promotion_discount_coordination(self):
        """Test pricing-to-promotion coordination for discount strategies."""
        with patch('agents.collaboration.agent_memory') as mock_memory:
            mock_memory.store_decision.return_value = "memory_456"
            
            # Execute collaboration
            result = await self.collaboration_workflow.pricing_to_promotion_discount_coordination(
                discount_opportunities=self.sample_discount_opportunities
            )
            
            # Verify collaboration was initiated
            assert result["status"] == "initiated"
            assert result["type"] == CollaborationType.PRICING_TO_PROMOTION.value
            assert result["opportunities_processed"] == 2
            assert result["campaign_recommendations"] == 2
            
            # Verify collaboration message structure
            message = result["collaboration_message"]
            assert message["message_type"] == "discount_coordination_request"
            assert message["from_agent"] == "pricing_agent"
            assert message["to_agent"] == "promotion_agent"
            
            # Verify campaign recommendations
            recommendations = message["campaign_recommendations"]
            assert len(recommendations) == 2
            assert all("recommended_campaign" in rec for rec in recommendations)
            assert all("expected_impact" in rec for rec in recommendations)
            
            # Verify different campaign types based on discount reasons
            campaign_types = [rec["recommended_campaign"]["type"] for rec in recommendations]
            assert "clearance_sale" in campaign_types  # slow_moving_inventory reason
            assert "price_match_promotion" in campaign_types  # competitive_pressure reason
    
    @pytest.mark.asyncio
    async def test_promotion_to_inventory_stock_validation(self):
        """Test promotion-to-inventory stock validation."""
        with patch('agents.collaboration.agent_memory') as mock_memory:
            mock_memory.store_decision.return_value = "memory_789"
            
            # Execute collaboration
            result = await self.collaboration_workflow.promotion_to_inventory_stock_validation(
                campaign_requests=self.sample_campaign_requests
            )
            
            # Verify validation was completed
            assert result["status"] == "completed"
            assert result["type"] == CollaborationType.PROMOTION_TO_INVENTORY.value
            assert result["campaigns_validated"] == 1
            
            # Verify validation results
            validation_results = result["validation_results"]
            assert len(validation_results) == 1
            
            campaign_validation = validation_results[0]
            assert "campaign_feasible" in campaign_validation
            assert "stock_requirements" in campaign_validation
            assert len(campaign_validation["stock_requirements"]) == 2  # Two products
            
            # Verify stock requirement details
            stock_reqs = campaign_validation["stock_requirements"]
            for req in stock_reqs:
                assert "product_id" in req
                assert "current_stock" in req
                assert "expected_campaign_demand" in req
                assert "stock_sufficient" in req
                assert "recommended_actions" in req
    
    @pytest.mark.asyncio
    async def test_cross_agent_learning_from_outcomes(self):
        """Test cross-agent learning from shared decision outcomes."""
        # Sample decision outcomes
        decision_outcomes = [
            {
                "agent_id": "inventory_agent",
                "decision_type": "restock_alert",
                "success": True,
                "collaboration_id": "collab_001",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "agent_id": "pricing_agent",
                "decision_type": "price_adjustment",
                "success": True,
                "collaboration_id": "collab_001",
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            {
                "agent_id": "promotion_agent",
                "decision_type": "campaign_creation",
                "success": False,
                "collaboration_id": None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        with patch('agents.collaboration.agent_memory') as mock_memory:
            mock_memory.store_decision.return_value = "memory_learning"
            
            # Execute cross-agent learning
            result = await self.collaboration_workflow.cross_agent_learning_from_outcomes(
                decision_outcomes=decision_outcomes
            )
            
            # Verify learning was completed
            assert result["status"] == "completed"
            assert result["type"] == CollaborationType.CROSS_AGENT_LEARNING.value
            assert result["outcomes_analyzed"] == 3
            assert len(result["agents_involved"]) == 3
            
            # Verify shared insights were generated
            assert "shared_insights" in result
            assert len(result["shared_insights"]) > 0
            
            # Verify agent performance summary
            performance_summary = result["agent_performance_summary"]
            assert "inventory_agent" in performance_summary
            assert "pricing_agent" in performance_summary
            assert "promotion_agent" in performance_summary
            
            # Verify learning recommendations
            assert "learning_recommendations" in result
            assert len(result["learning_recommendations"]) > 0
            
            # Verify memory storage for all agents
            assert mock_memory.store_decision.call_count == 3  # One for each agent
    
    @pytest.mark.asyncio
    async def test_collaborative_market_event_response(self):
        """Test collaborative response to market events."""
        with patch('agents.collaboration.agent_memory') as mock_memory:
            mock_memory.store_decision.return_value = "memory_event"
            
            # Execute collaborative market event response
            result = await self.collaboration_workflow.collaborative_market_event_response(
                market_event=self.sample_market_event
            )
            
            # Verify collaboration was initiated
            assert result["status"] == "initiated"
            assert result["type"] == CollaborationType.MARKET_EVENT_RESPONSE.value
            assert result["event_type"] == "demand_spike"
            assert result["event_severity"] == "high"
            
            # Verify all agents are participating
            assert len(result["participating_agents"]) == 3
            assert "inventory_agent" in result["participating_agents"]
            assert "pricing_agent" in result["participating_agents"]
            assert "promotion_agent" in result["participating_agents"]
            
            # Verify response strategy was generated
            strategy = result["response_strategy"]
            assert strategy["strategy_type"] == "demand_surge_response"
            assert strategy["primary_objective"] == "maximize_revenue_capture"
            assert strategy["coordination_priority"] == "high"
            
            # Verify agent action plans
            action_plans = result["agent_action_plans"]
            assert len(action_plans) == 3
            
            # Verify inventory agent actions for demand spike
            inventory_actions = action_plans["inventory_agent"]["actions"]
            assert "validate_current_stock_levels" in inventory_actions
            assert "calculate_surge_demand_forecast" in inventory_actions
            
            # Verify pricing agent actions for demand spike
            pricing_actions = action_plans["pricing_agent"]["actions"]
            assert "analyze_demand_elasticity" in pricing_actions
            assert "calculate_optimal_surge_pricing" in pricing_actions
            
            # Verify promotion agent actions for demand spike
            promotion_actions = action_plans["promotion_agent"]["actions"]
            assert "create_urgency_campaigns" in promotion_actions
            assert "amplify_social_media_presence" in promotion_actions
            
            # Verify coordination timeline
            timeline = result["coordination_timeline"]
            assert len(timeline) > 0
            assert all("phase" in phase for phase in timeline)
            assert all("actions" in phase for phase in timeline)
    
    @pytest.mark.asyncio
    async def test_collaboration_workflow_error_handling(self):
        """Test error handling in collaboration workflows."""
        # Test with invalid slow-moving items data
        with patch('agents.collaboration.agent_memory') as mock_memory:
            mock_memory.store_decision.side_effect = Exception("Memory storage failed")
            
            result = await self.collaboration_workflow.inventory_to_pricing_slow_moving_alert(
                slow_moving_items=[]  # Empty list
            )
            
            # Should handle gracefully and return error status
            assert result["status"] == "failed"
            assert "Error initiating collaboration" in result["analysis"]
    
    def test_helper_methods(self):
        """Test collaboration workflow helper methods."""
        # Test urgency level calculation
        high_urgency_items = [
            {"days_without_sale": 50, "inventory_value": 5000},
            {"days_without_sale": 60, "inventory_value": 3000}
        ]
        urgency = self.collaboration_workflow._calculate_urgency_level(high_urgency_items)
        assert urgency == "critical"
        
        low_urgency_items = [
            {"days_without_sale": 10, "inventory_value": 500}
        ]
        urgency = self.collaboration_workflow._calculate_urgency_level(low_urgency_items)
        assert urgency == "low"
        
        # Test optimal channels selection
        channels = self.collaboration_workflow._get_optimal_channels("flash_sale")
        assert "push_notifications" in channels
        assert "social_media" in channels
        
        # Test messaging strategy
        messaging = self.collaboration_workflow._get_messaging_strategy("clearance_sale", 25.0)
        assert "Final Clearance" in messaging["primary_message"]
        assert "25% Off" in messaging["primary_message"]
    
    @pytest.mark.asyncio
    async def test_end_to_end_collaboration_scenario(self):
        """Test complete end-to-end collaboration scenario."""
        with patch('agents.collaboration.agent_memory') as mock_memory:
            mock_memory.store_decision.return_value = "memory_e2e"
            
            # Step 1: Inventory identifies slow-moving items and alerts pricing
            inventory_result = await self.collaboration_workflow.inventory_to_pricing_slow_moving_alert(
                slow_moving_items=self.sample_slow_moving_items
            )
            assert inventory_result["status"] == "initiated"
            
            # Step 2: Pricing creates discount opportunities and coordinates with promotion
            pricing_result = await self.collaboration_workflow.pricing_to_promotion_discount_coordination(
                discount_opportunities=self.sample_discount_opportunities
            )
            assert pricing_result["status"] == "initiated"
            
            # Step 3: Promotion validates stock availability with inventory
            promotion_result = await self.collaboration_workflow.promotion_to_inventory_stock_validation(
                campaign_requests=self.sample_campaign_requests
            )
            assert promotion_result["status"] == "completed"
            
            # Step 4: Market event triggers collaborative response
            event_result = await self.collaboration_workflow.collaborative_market_event_response(
                market_event=self.sample_market_event
            )
            assert event_result["status"] == "initiated"
            
            # Step 5: Cross-agent learning from all outcomes
            learning_outcomes = [
                {
                    "agent_id": "inventory_agent",
                    "decision_type": "collaboration",
                    "success": True,
                    "collaboration_id": inventory_result["collaboration_id"]
                },
                {
                    "agent_id": "pricing_agent", 
                    "decision_type": "collaboration",
                    "success": True,
                    "collaboration_id": pricing_result["collaboration_id"]
                },
                {
                    "agent_id": "promotion_agent",
                    "decision_type": "collaboration", 
                    "success": True,
                    "collaboration_id": promotion_result["collaboration_id"]
                }
            ]
            
            learning_result = await self.collaboration_workflow.cross_agent_learning_from_outcomes(
                decision_outcomes=learning_outcomes
            )
            assert learning_result["status"] == "completed"
            
            # Verify end-to-end workflow completed successfully
            assert len(self.collaboration_workflow.active_collaborations) >= 4
            assert all(result["status"] in ["initiated", "completed"] for result in [
                inventory_result, pricing_result, promotion_result, event_result, learning_result
            ])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])