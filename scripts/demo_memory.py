#!/usr/bin/env python3
"""Demo script showing memory system functionality."""

import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.memory import AgentMemory
from models.core import AgentDecision, ActionType


def demo_memory_operations():
    """Demonstrate memory system operations."""
    print("=== Agent Memory System Demo ===\n")
    
    # Mock embedding service for demo
    with patch('agents.memory.EmbeddingService') as mock_service_class:
        mock_service = Mock()
        mock_service.generate_embedding.return_value = [0.1] * 384
        mock_service_class.return_value = mock_service
        
        memory = AgentMemory()
        
        print("1. Storing agent decisions...")
        
        # Create sample decisions
        pricing_decision = AgentDecision(
            agent_id="pricing_agent",
            timestamp=datetime.now(),
            action_type=ActionType.PRICE_ADJUSTMENT,
            parameters={
                "product_id": "SKU123",
                "old_price": 29.99,
                "new_price": 24.99,
                "discount_percentage": 16.7
            },
            rationale="Reducing price to clear slow-moving inventory and improve turnover",
            confidence_score=0.87,
            expected_outcome={
                "demand_increase_percentage": 25,
                "inventory_turnover_days": 14,
                "revenue_impact": -750.0
            }
        )
        
        inventory_decision = AgentDecision(
            agent_id="inventory_agent",
            timestamp=datetime.now(),
            action_type=ActionType.STOCK_ALERT,
            parameters={
                "product_id": "SKU456",
                "current_stock": 15,
                "reorder_quantity": 200,
                "urgency": "high"
            },
            rationale="Stock level below safety threshold with increasing demand forecast",
            confidence_score=0.92,
            expected_outcome={
                "stockout_prevention": True,
                "service_level": 0.98,
                "cost_impact": 2400.0
            }
        )
        
        # Store decisions with context
        pricing_context = {
            "product_category": "electronics",
            "inventory_level": 85,
            "days_in_stock": 45,
            "competitor_avg_price": 26.50,
            "demand_trend": "declining"
        }
        
        inventory_context = {
            "product_category": "home_goods",
            "lead_time_days": 7,
            "demand_forecast": 180,
            "seasonal_factor": 1.2,
            "supplier_reliability": 0.95
        }
        
        pricing_memory_id = memory.store_decision(
            agent_id="pricing_agent",
            decision=pricing_decision,
            context=pricing_context
        )
        print(f"   ✓ Stored pricing decision: {pricing_memory_id}")
        
        inventory_memory_id = memory.store_decision(
            agent_id="inventory_agent", 
            decision=inventory_decision,
            context=inventory_context
        )
        print(f"   ✓ Stored inventory decision: {inventory_memory_id}")
        
        print("\n2. Adding outcomes to decisions...")
        
        # Add outcomes after some time
        pricing_outcome = {
            "actual_demand_increase": 22.5,
            "inventory_sold": 45,
            "actual_revenue_impact": -680.0,
            "effectiveness_score": 0.84,
            "customer_satisfaction": 0.91
        }
        
        inventory_outcome = {
            "restock_completed": True,
            "actual_demand": 175,
            "service_level_achieved": 0.97,
            "stockout_prevented": True,
            "effectiveness_score": 0.89
        }
        
        memory.update_outcome(pricing_memory_id, pricing_outcome)
        memory.update_outcome(inventory_memory_id, inventory_outcome)
        print("   ✓ Updated pricing decision outcome")
        print("   ✓ Updated inventory decision outcome")
        
        print("\n3. Retrieving similar decisions...")
        
        # Search for similar pricing decisions
        similar_pricing_context = {
            "product_category": "electronics",
            "inventory_level": 90,
            "days_in_stock": 40,
            "demand_trend": "declining"
        }
        
        similar_decisions = memory.retrieve_similar_decisions(
            agent_id="pricing_agent",
            current_context=similar_pricing_context,
            action_type=ActionType.PRICE_ADJUSTMENT.value,
            limit=3,
            similarity_threshold=0.5
        )
        
        print(f"   ✓ Found {len(similar_decisions)} similar pricing decisions")
        for i, (decision_data, similarity) in enumerate(similar_decisions):
            print(f"     - Decision {i+1}: {similarity:.2f} similarity")
            print(f"       Action: {decision_data['decision']['action_type']}")
            print(f"       Confidence: {decision_data['decision']['confidence_score']}")
            if decision_data.get('outcome'):
                print(f"       Effectiveness: {decision_data['outcome'].get('effectiveness_score', 'N/A')}")
        
        print("\n4. Getting agent decision history...")
        
        pricing_history = memory.get_agent_decision_history(
            agent_id="pricing_agent",
            include_outcomes=True,
            limit=10
        )
        
        inventory_history = memory.get_agent_decision_history(
            agent_id="inventory_agent",
            include_outcomes=True,
            limit=10
        )
        
        print(f"   ✓ Pricing agent has {len(pricing_history)} decisions with outcomes")
        print(f"   ✓ Inventory agent has {len(inventory_history)} decisions with outcomes")
        
        print("\n5. System metrics...")
        
        metrics = memory.get_system_metrics()
        print(f"   ✓ Total decisions stored: {metrics['total_decisions']}")
        print(f"   ✓ Decisions with outcomes: {metrics['decisions_with_outcomes']}")
        print(f"   ✓ Agent breakdown:")
        for agent, count in metrics['agent_decision_counts'].items():
            print(f"     - {agent}: {count} decisions")
        
        print("\n6. Demonstrating learning from past decisions...")
        
        # Show how an agent could use past decisions for current situation
        current_situation = {
            "product_category": "electronics",
            "inventory_level": 75,
            "days_in_stock": 50,
            "demand_trend": "declining",
            "competitor_avg_price": 25.00
        }
        
        print("   Current situation:")
        print(f"     - Product category: {current_situation['product_category']}")
        print(f"     - Inventory level: {current_situation['inventory_level']}")
        print(f"     - Days in stock: {current_situation['days_in_stock']}")
        print(f"     - Demand trend: {current_situation['demand_trend']}")
        
        relevant_decisions = memory.retrieve_similar_decisions(
            agent_id="pricing_agent",
            current_context=current_situation,
            limit=3
        )
        
        if relevant_decisions:
            print(f"\n   ✓ Found {len(relevant_decisions)} relevant past decisions:")
            for i, (past_decision, similarity) in enumerate(relevant_decisions):
                decision = past_decision['decision']
                outcome = past_decision.get('outcome', {})
                
                print(f"\n     Past Decision {i+1} (similarity: {similarity:.2f}):")
                print(f"       - Action: {decision['action_type']}")
                print(f"       - Rationale: {decision['rationale'][:60]}...")
                print(f"       - Confidence: {decision['confidence_score']}")
                
                if outcome:
                    print(f"       - Actual effectiveness: {outcome.get('effectiveness_score', 'N/A')}")
                    print(f"       - Revenue impact: ${outcome.get('actual_revenue_impact', 'N/A')}")
                
                print("       → This suggests similar actions might be effective")
        else:
            print("   ✓ No similar past decisions found (agent is learning from scratch)")
        
        print("\n=== Demo Complete ===")
        print("The memory system successfully:")
        print("  ✓ Stored agent decisions with full context")
        print("  ✓ Updated decisions with actual outcomes")
        print("  ✓ Retrieved similar past decisions for learning")
        print("  ✓ Provided decision history for analysis")
        print("  ✓ Enabled agents to learn from past experiences")


if __name__ == "__main__":
    demo_memory_operations()