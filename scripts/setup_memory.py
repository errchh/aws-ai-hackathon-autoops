#!/usr/bin/env python3
"""Script to initialize and test the ChromaDB memory system."""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.memory import AgentMemory, EmbeddingService
from models.core import AgentDecision, ActionType
from config.settings import settings


def test_embedding_service():
    """Test the embedding service functionality."""
    print("Testing Embedding Service...")
    
    try:
        # Note: This will fail without proper AWS credentials
        # In a real environment, you would have AWS credentials configured
        service = EmbeddingService()
        
        # Test with a simple text
        test_text = "This is a test for embedding generation"
        print(f"Generating embedding for: '{test_text}'")
        
        # This would normally call AWS Bedrock
        # For testing without AWS, we'll catch the exception
        try:
            embedding = service.generate_embedding(test_text)
            print(f"✓ Embedding generated successfully (dimension: {len(embedding)})")
            return True
        except Exception as e:
            print(f"⚠ Embedding service requires AWS credentials: {e}")
            print("  This is expected in development without AWS setup")
            return False
            
    except Exception as e:
        print(f"✗ Embedding service initialization failed: {e}")
        return False


def test_chromadb_setup():
    """Test ChromaDB initialization and basic operations."""
    print("\nTesting ChromaDB Setup...")
    
    try:
        # Create memory instance (this will initialize ChromaDB)
        print(f"Initializing ChromaDB at: {settings.chromadb.persist_directory}")
        
        # Mock the embedding service for testing
        from unittest.mock import Mock, patch
        
        with patch('agents.memory.EmbeddingService') as mock_service_class:
            mock_service = Mock()
            mock_service.generate_embedding.return_value = [0.1] * 384
            mock_service_class.return_value = mock_service
            
            memory = AgentMemory()
            print("✓ ChromaDB initialized successfully")
            
            # Test basic operations
            print("Testing basic memory operations...")
            
            # Create test decision
            from models.core import ActionType
            test_decision = AgentDecision(
                agent_id="test_agent",
                timestamp=datetime.now(),
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={"test_param": "test_value"},
                rationale="Testing memory system setup",
                confidence_score=0.95,
                expected_outcome={"test_metric": 100.0}
            )
            
            test_context = {
                "test_context": "setup_test",
                "environment": "development",
                "timestamp": datetime.now().isoformat()
            }
            
            # Store decision
            memory_id = memory.store_decision(
                agent_id="test_agent",
                decision=test_decision,
                context=test_context
            )
            print(f"✓ Decision stored with ID: {memory_id}")
            
            # Update outcome
            test_outcome = {
                "success": True,
                "test_score": 0.98,
                "setup_time": "2024-01-15T10:30:00Z"
            }
            memory.update_outcome(memory_id, test_outcome)
            print("✓ Outcome updated successfully")
            
            # Retrieve similar decisions
            similar = memory.retrieve_similar_decisions(
                agent_id="test_agent",
                current_context=test_context,
                limit=5
            )
            print(f"✓ Retrieved {len(similar)} similar decisions")
            
            # Get decision history
            history = memory.get_agent_decision_history(
                agent_id="test_agent",
                limit=10
            )
            print(f"✓ Retrieved {len(history)} historical decisions")
            
            # Get system metrics
            metrics = memory.get_system_metrics()
            print(f"✓ System metrics: {metrics['total_decisions']} total decisions")
            
            print("✓ All ChromaDB operations completed successfully")
            return True
            
    except Exception as e:
        print(f"✗ ChromaDB setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_sample_data():
    """Create sample data for demonstration."""
    print("\nCreating sample data...")
    
    try:
        from unittest.mock import Mock, patch
        
        with patch('agents.memory.EmbeddingService') as mock_service_class:
            mock_service = Mock()
            mock_service.generate_embedding.return_value = [0.1] * 384
            mock_service_class.return_value = mock_service
            
            memory = AgentMemory()
            
            # Sample decisions for different agents
            sample_decisions = [
                {
                    "agent_id": "pricing_agent",
                    "decision": AgentDecision(
                        agent_id="pricing_agent",
                        timestamp=datetime.now(),
                        action_type=ActionType.PRICE_ADJUSTMENT,
                        parameters={"product_id": "SKU123", "new_price": 29.99, "old_price": 34.99},
                        rationale="Reducing price due to high inventory and declining demand",
                        confidence_score=0.87,
                        expected_outcome={"revenue_impact": -500.0, "inventory_turnover": 1.2}
                    ),
                    "context": {
                        "product_id": "SKU123",
                        "current_inventory": 150,
                        "demand_trend": "declining",
                        "competitor_avg_price": 31.50,
                        "days_in_stock": 45
                    },
                    "outcome": {
                        "actual_revenue_impact": -450.0,
                        "inventory_reduction": 35,
                        "effectiveness_score": 0.82
                    }
                },
                {
                    "agent_id": "inventory_agent", 
                    "decision": AgentDecision(
                        agent_id="inventory_agent",
                        timestamp=datetime.now(),
                        action_type=ActionType.STOCK_ALERT,
                        parameters={"product_id": "SKU456", "recommended_quantity": 200},
                        rationale="Inventory below safety threshold with increasing demand",
                        confidence_score=0.92,
                        expected_outcome={"stockout_prevention": True, "service_level": 0.98}
                    ),
                    "context": {
                        "product_id": "SKU456",
                        "current_inventory": 25,
                        "safety_stock": 50,
                        "demand_forecast": 180,
                        "lead_time_days": 7
                    },
                    "outcome": {
                        "restock_completed": True,
                        "actual_demand": 175,
                        "service_level_achieved": 0.97,
                        "effectiveness_score": 0.89
                    }
                },
                {
                    "agent_id": "promotion_agent",
                    "decision": AgentDecision(
                        agent_id="promotion_agent", 
                        timestamp=datetime.now(),
                        action_type=ActionType.PROMOTION_CREATION,
                        parameters={
                            "product_ids": ["SKU789", "SKU790"],
                            "discount_percent": 20,
                            "duration_hours": 24
                        },
                        rationale="Social sentiment trending positive for product category",
                        confidence_score=0.78,
                        expected_outcome={"sales_lift": 2.5, "revenue_increase": 3000.0}
                    ),
                    "context": {
                        "product_category": "electronics",
                        "social_sentiment": 0.75,
                        "inventory_levels": {"SKU789": 80, "SKU790": 120},
                        "seasonal_factor": 1.1
                    },
                    "outcome": {
                        "actual_sales_lift": 2.3,
                        "revenue_increase": 2850.0,
                        "inventory_sold": {"SKU789": 45, "SKU790": 67},
                        "effectiveness_score": 0.85
                    }
                }
            ]
            
            # Store sample decisions
            stored_ids = []
            for sample in sample_decisions:
                memory_id = memory.store_decision(
                    agent_id=sample["agent_id"],
                    decision=sample["decision"],
                    context=sample["context"],
                    outcome=sample["outcome"]
                )
                stored_ids.append(memory_id)
                print(f"✓ Stored {sample['agent_id']} decision: {sample['decision'].action_type}")
            
            print(f"✓ Created {len(stored_ids)} sample decisions")
            
            # Display final metrics
            metrics = memory.get_system_metrics()
            print(f"\nFinal system state:")
            print(f"  Total decisions: {metrics['total_decisions']}")
            print(f"  Decisions with outcomes: {metrics['decisions_with_outcomes']}")
            print(f"  Agent breakdown: {metrics['agent_decision_counts']}")
            
            return True
            
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup and test function."""
    print("=== ChromaDB Memory System Setup ===")
    print(f"Project root: {project_root}")
    print(f"ChromaDB persist directory: {settings.chromadb.persist_directory}")
    print(f"Collection name: {settings.chromadb.collection_name}")
    
    # Ensure data directory exists
    os.makedirs(settings.chromadb.persist_directory, exist_ok=True)
    print(f"✓ Data directory created/verified")
    
    # Test components
    embedding_ok = test_embedding_service()
    chromadb_ok = test_chromadb_setup()
    
    if chromadb_ok:
        sample_data_ok = create_sample_data()
    else:
        sample_data_ok = False
    
    # Summary
    print("\n=== Setup Summary ===")
    print(f"Embedding Service: {'✓' if embedding_ok else '⚠'} {'OK' if embedding_ok else 'Needs AWS credentials'}")
    print(f"ChromaDB Setup: {'✓' if chromadb_ok else '✗'} {'OK' if chromadb_ok else 'Failed'}")
    print(f"Sample Data: {'✓' if sample_data_ok else '✗'} {'OK' if sample_data_ok else 'Failed'}")
    
    if chromadb_ok:
        print("\n✓ Memory system is ready for use!")
        print("  - ChromaDB is initialized and operational")
        print("  - Sample data has been created for testing")
        print("  - Configure AWS credentials for embedding service")
    else:
        print("\n✗ Memory system setup incomplete")
        print("  - Check ChromaDB installation and permissions")
        
    return chromadb_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)