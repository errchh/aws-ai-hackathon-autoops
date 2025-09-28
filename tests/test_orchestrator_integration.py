"""
Integration tests for the AWS Strands Agents Orchestrator.

This module contains comprehensive integration tests for the RetailOptimizationOrchestrator
to verify multi-agent coordination scenarios and system-wide functionality.
"""

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from agents.orchestrator import RetailOptimizationOrchestrator, SystemStatus
from models.core import SystemMetrics


class TestRetailOptimizationOrchestrator:
    """Integration test suite for the Retail Optimization Orchestrator."""
    
    def setup_method(self):
        """Set up test orchestrator and mock agents."""
        self.orchestrator = RetailOptimizationOrchestrator()
        
        # Create mock agents
        self.mock_agents = []
        for agent_id in ['pricing_agent', 'inventory_agent', 'promotion_agent']:
            mock_agent = Mock()
            mock_agent.agent_id = agent_id
            self.mock_agents.append(mock_agent)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization and basic properties."""
        assert self.orchestrator.orchestrator_id == "retail_optimization_orchestrator"
        assert self.orchestrator.system_status == SystemStatus.OFFLINE
        assert len(self.orchestrator.agents) == 0
        assert len(self.orchestrator.agent_status) == 0
        assert isinstance(self.orchestrator.system_metrics, SystemMetrics)
    
    def test_agent_registration(self):
        """Test agent registration functionality."""
        # Test successful registration
        success = self.orchestrator.register_agents(self.mock_agents)
        assert success is True
        
        # Verify agents are registered
        assert len(self.orchestrator.agents) == 3
        assert 'pricing_agent' in self.orchestrator.agents
        assert 'inventory_agent' in self.orchestrator.agents
        assert 'promotion_agent' in self.orchestrator.agents
        
        # Verify system status updated
        assert self.orchestrator.system_status == SystemStatus.HEALTHY
        
        # Verify agent status initialized
        for agent_id in ['pricing_agent', 'inventory_agent', 'promotion_agent']:
            assert agent_id in self.orchestrator.agent_status
            assert self.orchestrator.agent_status[agent_id]["status"] == "active"
            assert self.orchestrator.agent_status[agent_id]["message_count"] == 0
            assert self.orchestrator.agent_status[agent_id]["error_count"] == 0
            assert self.orchestrator.agent_status[agent_id]["success_rate"] == 1.0
    
    def test_partial_agent_registration(self):
        """Test system behavior with partial agent registration."""
        # Register only 2 agents
        partial_agents = self.mock_agents[:2]
        success = self.orchestrator.register_agents(partial_agents)
        assert success is True
        
        # System should be degraded with only 2 agents
        assert self.orchestrator.system_status == SystemStatus.DEGRADED
        assert len(self.orchestrator.agents) == 2
    
    def test_single_agent_registration(self):
        """Test system behavior with single agent registration."""
        # Register only 1 agent
        single_agent = [self.mock_agents[0]]
        success = self.orchestrator.register_agents(single_agent)
        assert success is True
        
        # System should be critical with only 1 agent
        assert self.orchestrator.system_status == SystemStatus.CRITICAL
        assert len(self.orchestrator.agents) == 1
    
    def test_get_system_status(self):
        """Test system status reporting."""
        # Test initial status
        status = self.orchestrator.get_system_status()
        assert status["orchestrator_id"] == "retail_optimization_orchestrator"
        assert status["system_status"] == "offline"
        assert status["registered_agents"] == 0
        assert status["active_agents"] == 0
        
        # Register agents and test updated status
        self.orchestrator.register_agents(self.mock_agents)
        status = self.orchestrator.get_system_status()
        assert status["system_status"] == "healthy"
        assert status["registered_agents"] == 3
        assert status["active_agents"] == 3
        assert "agent_health" in status
        assert len(status["agent_health"]) == 3
    
    @pytest.mark.asyncio
    async def test_market_event_processing(self):
        """Test market event processing and agent coordination."""
        # Register agents first
        self.orchestrator.register_agents(self.mock_agents)
        
        # Test demand spike event
        event_data = {
            'event_type': 'demand_spike',
            'affected_products': ['SKU123', 'SKU456'],
            'impact_magnitude': 0.75,
            'metadata': {'trigger': 'social_media_trend'}
        }
        
        result = await self.orchestrator.process_market_event(event_data)
        
        # Verify event processing
        assert result["status"] == "completed"
        assert "event_id" in result
        assert "workflow_id" in result
        assert result["agent_responses"] >= 1  # At least one agent should respond
        assert "processing_time" in result
        assert result["responding_agents"] is not None
    
    @pytest.mark.asyncio
    async def test_competitor_price_change_event(self):
        """Test competitor price change event processing."""
        self.orchestrator.register_agents(self.mock_agents)
        
        event_data = {
            'event_type': 'competitor_price_change',
            'affected_products': ['SKU789'],
            'impact_magnitude': 0.45,
            'metadata': {'competitor': 'Competitor A', 'price_change': -15}
        }
        
        result = await self.orchestrator.process_market_event(event_data)
        
        assert result["status"] == "completed"
        assert result["agent_responses"] >= 1
        assert 'pricing_agent' in result["responding_agents"]
    
    @pytest.mark.asyncio
    async def test_social_trend_event(self):
        """Test social trend event processing."""
        self.orchestrator.register_agents(self.mock_agents)
        
        event_data = {
            'event_type': 'social_trend',
            'affected_products': ['SKU101', 'SKU102'],
            'impact_magnitude': 0.60,
            'metadata': {'platform': 'instagram', 'trend_type': 'viral_post'}
        }
        
        result = await self.orchestrator.process_market_event(event_data)
        
        assert result["status"] == "completed"
        assert 'promotion_agent' in result["responding_agents"]
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self):
        """Test direct agent coordination functionality."""
        self.orchestrator.register_agents(self.mock_agents)
        
        coordination_request = {
            'requesting_agent': 'inventory_agent',
            'target_agents': ['pricing_agent', 'promotion_agent'],
            'coordination_type': 'consultation',
            'content': {
                'slow_moving_items': ['SKU123', 'SKU456'],
                'urgency': 'medium'
            }
        }
        
        result = await self.orchestrator.coordinate_agents(coordination_request)
        
        assert result["status"] == "completed"
        assert result["requesting_agent"] == "inventory_agent"
        assert len(result["target_agents"]) == 2
        assert result["successful_responses"] >= 0
        assert "coordination_results" in result
    
    @pytest.mark.asyncio
    async def test_coordination_with_unavailable_agents(self):
        """Test coordination when some target agents are unavailable."""
        # Register only 2 agents
        self.orchestrator.register_agents(self.mock_agents[:2])
        
        coordination_request = {
            'requesting_agent': 'inventory_agent',
            'target_agents': ['pricing_agent', 'promotion_agent'],  # promotion_agent not registered
            'coordination_type': 'consultation',
            'content': {'message': 'test coordination'}
        }
        
        result = await self.orchestrator.coordinate_agents(coordination_request)
        
        # Should still succeed with available agents
        assert result["status"] == "completed"
        assert len(result["target_agents"]) == 1  # Only pricing_agent available
        assert 'pricing_agent' in result["target_agents"]
    
    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self):
        """Test conflict detection and resolution between agents."""
        self.orchestrator.register_agents(self.mock_agents)
        
        # Create event that might cause conflicts
        event_data = {
            'event_type': 'demand_spike',
            'affected_products': ['SKU123'],
            'impact_magnitude': 0.80
        }
        
        result = await self.orchestrator.process_market_event(event_data)
        
        # Check if conflicts were detected and handled
        assert result["status"] == "completed"
        assert "conflicts_detected" in result
        
        # If conflicts were detected, resolution should be present
        if result["conflicts_detected"] > 0:
            assert "resolution" in result
            assert result["resolution"] is not None
    
    @pytest.mark.asyncio
    async def test_collaboration_workflow_inventory_to_pricing(self):
        """Test inventory-to-pricing collaboration workflow."""
        self.orchestrator.register_agents(self.mock_agents)
        
        workflow_data = {
            'slow_moving_items': [
                {
                    'product_id': 'SKU123',
                    'days_without_sale': 30,
                    'inventory_value': 500.0,
                    'current_stock': 25
                },
                {
                    'product_id': 'SKU456',
                    'days_without_sale': 45,
                    'inventory_value': 750.0,
                    'current_stock': 40
                }
            ]
        }
        
        # Mock the collaboration workflow to avoid AWS dependency
        with patch('agents.orchestrator.collaboration_workflow') as mock_workflow:
            mock_workflow.inventory_to_pricing_slow_moving_alert.return_value = {
                "status": "initiated",
                "collaboration_id": "test-123",
                "items_processed": 2
            }
            
            result = await self.orchestrator.trigger_collaboration_workflow(
                'inventory_to_pricing_slow_moving',
                workflow_data
            )
            
            assert result["status"] == "initiated"
            assert "collaboration_id" in result
    
    @pytest.mark.asyncio
    async def test_collaboration_workflow_pricing_to_promotion(self):
        """Test pricing-to-promotion collaboration workflow."""
        self.orchestrator.register_agents(self.mock_agents)
        
        workflow_data = {
            'discount_opportunities': [
                {
                    'product_id': 'SKU789',
                    'current_price': 29.99,
                    'discount_percentage': 15,
                    'reason': 'competitive_pressure'
                }
            ]
        }
        
        with patch('agents.orchestrator.collaboration_workflow') as mock_workflow:
            mock_workflow.pricing_to_promotion_discount_coordination.return_value = {
                "status": "initiated",
                "collaboration_id": "test-456",
                "opportunities_processed": 1
            }
            
            result = await self.orchestrator.trigger_collaboration_workflow(
                'pricing_to_promotion_discount',
                workflow_data
            )
            
            assert result["status"] == "initiated"
    
    @pytest.mark.asyncio
    async def test_collaboration_workflow_promotion_to_inventory(self):
        """Test promotion-to-inventory collaboration workflow."""
        self.orchestrator.register_agents(self.mock_agents)
        
        workflow_data = {
            'campaign_requests': [
                {
                    'campaign_id': 'CAMP001',
                    'product_ids': ['SKU123', 'SKU456'],
                    'expected_demand_increase': 150,
                    'duration_days': 7
                }
            ]
        }
        
        with patch('agents.orchestrator.collaboration_workflow') as mock_workflow:
            mock_workflow.promotion_to_inventory_stock_validation.return_value = {
                "status": "completed",
                "collaboration_id": "test-789",
                "campaigns_validated": 1
            }
            
            result = await self.orchestrator.trigger_collaboration_workflow(
                'promotion_to_inventory_validation',
                workflow_data
            )
            
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_cross_agent_learning_workflow(self):
        """Test cross-agent learning workflow."""
        self.orchestrator.register_agents(self.mock_agents)
        
        workflow_data = {
            'decision_outcomes': [
                {
                    'agent_id': 'pricing_agent',
                    'decision_type': 'price_adjustment',
                    'success': True,
                    'outcome_metrics': {'revenue_increase': 0.15}
                },
                {
                    'agent_id': 'inventory_agent',
                    'decision_type': 'restock_alert',
                    'success': True,
                    'outcome_metrics': {'stockout_prevention': True}
                }
            ],
            'learning_context': {
                'market_conditions': 'high_demand',
                'time_period': '2024-01-15'
            }
        }
        
        with patch('agents.orchestrator.collaboration_workflow') as mock_workflow:
            mock_workflow.cross_agent_learning_from_outcomes.return_value = {
                "status": "completed",
                "learning_id": "learn-123",
                "insights_generated": 3
            }
            
            result = await self.orchestrator.trigger_collaboration_workflow(
                'cross_agent_learning',
                workflow_data
            )
            
            assert result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_market_event_response_workflow(self):
        """Test collaborative market event response workflow."""
        self.orchestrator.register_agents(self.mock_agents)
        
        workflow_data = {
            'market_event': {
                'event_type': 'supply_disruption',
                'affected_products': ['SKU123', 'SKU456'],
                'severity': 'high'
            },
            'participating_agents': ['inventory_agent', 'pricing_agent']
        }
        
        with patch('agents.orchestrator.collaboration_workflow') as mock_workflow:
            mock_workflow.collaborative_market_event_response.return_value = {
                "status": "completed",
                "response_id": "resp-456",
                "participating_agents": 2
            }
            
            result = await self.orchestrator.trigger_collaboration_workflow(
                'market_event_response',
                workflow_data
            )
            
            assert result["status"] == "completed"
    
    def test_unknown_collaboration_workflow(self):
        """Test handling of unknown collaboration workflow types."""
        self.orchestrator.register_agents(self.mock_agents)
        
        async def test_unknown():
            result = await self.orchestrator.trigger_collaboration_workflow(
                'unknown_workflow_type',
                {}
            )
            assert result["status"] == "error"
            assert "Unknown collaboration workflow type" in result["analysis"]
        
        asyncio.run(test_unknown())
    
    def test_system_metrics_updates(self):
        """Test system metrics updates during operations."""
        self.orchestrator.register_agents(self.mock_agents)
        
        # Initial metrics
        initial_decision_count = self.orchestrator.system_metrics.decision_count
        initial_collaboration_score = self.orchestrator.system_metrics.agent_collaboration_score
        
        # Simulate workflow completion
        mock_responses = [
            ('pricing_agent', {'status': 'success'}),
            ('inventory_agent', {'status': 'success'})
        ]
        
        self.orchestrator._update_system_metrics('test-workflow', mock_responses)
        
        # Verify metrics updated
        assert self.orchestrator.system_metrics.decision_count > initial_decision_count
        assert self.orchestrator.system_metrics.agent_collaboration_score >= initial_collaboration_score
    
    def test_agent_status_tracking(self):
        """Test agent status tracking and updates."""
        self.orchestrator.register_agents(self.mock_agents)
        
        # Test status update
        agent_id = 'pricing_agent'
        initial_message_count = self.orchestrator.agent_status[agent_id]["message_count"]
        
        self.orchestrator._update_agent_status(agent_id, "message_processed")
        
        # Verify status updated
        assert self.orchestrator.agent_status[agent_id]["message_count"] > initial_message_count
        assert self.orchestrator.agent_status[agent_id]["last_heartbeat"] is not None
    
    def test_agent_error_tracking(self):
        """Test agent error tracking and success rate calculation."""
        self.orchestrator.register_agents(self.mock_agents)
        
        agent_id = 'pricing_agent'
        
        # Simulate some successful messages
        for _ in range(5):
            self.orchestrator._update_agent_status(agent_id, "message_processed")
        
        initial_success_rate = self.orchestrator.agent_status[agent_id]["success_rate"]
        
        # Simulate an error
        self.orchestrator._update_agent_error_count(agent_id)
        
        # Verify error tracking
        assert self.orchestrator.agent_status[agent_id]["error_count"] == 1
        assert self.orchestrator.agent_status[agent_id]["success_rate"] < initial_success_rate
    
    @pytest.mark.asyncio
    async def test_event_processing_timeout_handling(self):
        """Test handling of event processing timeouts."""
        self.orchestrator.register_agents(self.mock_agents)
        
        # Mock a timeout scenario by patching asyncio.wait_for
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            event_data = {
                'event_type': 'demand_spike',
                'affected_products': ['SKU123'],
                'impact_magnitude': 0.75
            }
            
            result = await self.orchestrator.process_market_event(event_data)
            
            assert result["status"] == "timeout"
            assert "timed out" in result["analysis"]
    
    @pytest.mark.asyncio
    async def test_coordination_timeout_handling(self):
        """Test handling of coordination timeouts."""
        self.orchestrator.register_agents(self.mock_agents)
        
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            coordination_request = {
                'requesting_agent': 'inventory_agent',
                'target_agents': ['pricing_agent'],
                'coordination_type': 'consultation',
                'content': {'test': 'data'}
            }
            
            result = await self.orchestrator.coordinate_agents(coordination_request)
            
            assert result["status"] == "timeout"
            assert "timed out" in result["analysis"]
    
    def test_orchestrator_health_monitoring(self):
        """Test orchestrator health monitoring capabilities."""
        self.orchestrator.register_agents(self.mock_agents)
        
        # Get system status
        status = self.orchestrator.get_system_status()
        
        # Verify health monitoring data
        assert "agent_health" in status
        assert len(status["agent_health"]) == 3
        
        for agent_id in ['pricing_agent', 'inventory_agent', 'promotion_agent']:
            agent_health = status["agent_health"][agent_id]
            assert "status" in agent_health
            assert "success_rate" in agent_health
            assert "last_heartbeat" in agent_health
            assert agent_health["status"] == "active"
            assert agent_health["success_rate"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])