#!/usr/bin/env python3
"""
Demonstration of Langfuse integration foundation.

This script shows how to use the Langfuse integration foundation
that was set up as part of task 1.
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.langfuse_config import get_langfuse_client, LangfuseConfig
from config.langfuse_integration import get_langfuse_integration


def demonstrate_configuration():
    """Demonstrate Langfuse configuration management."""
    print("=== Langfuse Configuration Demo ===")
    
    # Get the global client
    client = get_langfuse_client()
    
    print(f"Client available: {client.is_available}")
    print(f"Configuration: {client.config}")
    
    # Perform health check
    health = client.health_check()
    print(f"Health check: {health}")
    
    print()


def demonstrate_integration_service():
    """Demonstrate Langfuse integration service."""
    print("=== Langfuse Integration Service Demo ===")
    
    # Get the integration service
    service = get_langfuse_integration()
    
    print(f"Service available: {service.is_available}")
    
    # Demonstrate graceful degradation
    print("\n--- Graceful Degradation Demo ---")
    
    # These operations will work gracefully even without credentials
    trace_id = service.create_simulation_trace({
        "type": "demand_spike",
        "source": "iot_sensor",
        "product_id": "PROD_123",
        "location": "store_001",
        "magnitude": 2.5
    })
    print(f"Simulation trace ID: {trace_id}")
    
    span_id = service.start_agent_span(
        "inventory_agent",
        "forecast_demand",
        parent_trace_id=trace_id,
        input_data={
            "product_id": "PROD_123",
            "historical_data": "last_30_days",
            "external_factors": ["weather", "promotions"]
        }
    )
    print(f"Agent span ID: {span_id}")
    
    # Log a decision
    service.log_agent_decision(
        "inventory_agent",
        {
            "type": "restock_recommendation",
            "inputs": {"current_stock": 50, "forecasted_demand": 200},
            "outputs": {"recommended_quantity": 150, "urgency": "high"},
            "confidence": 0.85,
            "reasoning": "Demand spike detected, current stock insufficient"
        }
    )
    print("Logged agent decision")
    
    # End the span
    service.end_agent_span(span_id, {
        "forecast_result": "high_demand",
        "recommended_action": "immediate_restock",
        "confidence_score": 0.85
    })
    print("Ended agent span")
    
    # Track collaboration
    collab_trace_id = service.track_collaboration(
        "inventory_pricing_sync",
        ["inventory_agent", "pricing_agent"],
        {
            "trigger": "stock_level_change",
            "coordination_type": "price_adjustment"
        }
    )
    print(f"Collaboration trace ID: {collab_trace_id}")
    
    # Finalize traces
    service.finalize_trace(trace_id, {
        "final_outcome": "restock_initiated",
        "total_agents_involved": 2,
        "processing_time_ms": 150
    })
    
    if collab_trace_id:
        service.finalize_trace(collab_trace_id, {
            "collaboration_result": "price_adjusted",
            "agents_coordinated": 2
        })
    
    print("Finalized traces")
    
    # Demonstrate context manager
    print("\n--- Context Manager Demo ---")
    with service.trace_operation(
        "market_analysis",
        input_data={"market_segment": "electronics", "time_period": "last_week"},
        metadata={"analysis_type": "competitive_pricing"}
    ) as trace:
        print(f"Operation trace: {trace}")
        # Simulate some work
        import time
        time.sleep(0.1)
    
    # Health check
    health = service.health_check()
    print(f"\nService health: {health}")
    
    # Flush any pending data
    service.flush()
    print("Flushed pending traces")
    
    print()


def demonstrate_with_mock_credentials():
    """Demonstrate with mock credentials (for testing purposes)."""
    print("=== Mock Credentials Demo ===")
    
    # Create a configuration with mock credentials
    mock_config = LangfuseConfig(
        public_key="pk-lf-mock-key-for-demo",
        secret_key="sk-lf-mock-secret-for-demo",
        host="https://mock.langfuse.com",
        enabled=True,
        environment="demo",
        release="0.1.0"
    )
    
    print(f"Mock configuration: {mock_config}")
    
    # Note: This would fail to connect, but shows the configuration structure
    print("Note: This configuration would be used in a real environment")
    print("with actual Langfuse credentials and host.")
    
    print()


def main():
    """Main demonstration function."""
    print("Langfuse Integration Foundation Demo")
    print("=" * 50)
    print()
    
    demonstrate_configuration()
    demonstrate_integration_service()
    demonstrate_with_mock_credentials()
    
    print("Demo completed successfully!")
    print()
    print("Next steps:")
    print("1. Set up actual Langfuse credentials in .env file")
    print("2. Configure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
    print("3. Run this demo again to see actual tracing in action")
    print("4. Proceed to implement agent instrumentation (Task 2)")


if __name__ == "__main__":
    main()