#!/usr/bin/env python3
"""
Demonstration script for Langfuse integration foundation.

This script shows how the Langfuse integration works and can be used
to test the setup without requiring actual Langfuse credentials.
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, '.')

from config.langfuse_config import LangfuseConfig, LangfuseClient, get_langfuse_client
from config.langfuse_integration import LangfuseIntegrationService, get_langfuse_integration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_langfuse_config():
    """Demonstrate Langfuse configuration management."""
    print("\n" + "="*60)
    print("LANGFUSE CONFIGURATION DEMO")
    print("="*60)
    
    # Create a test configuration
    config = LangfuseConfig(
        public_key="pk-test-demo",
        secret_key="sk-test-demo",
        host="https://demo.langfuse.com",
        enabled=True,
        sample_rate=0.5,
        debug=True
    )
    
    print(f"✓ Created Langfuse configuration:")
    print(f"  - Host: {config.host}")
    print(f"  - Enabled: {config.enabled}")
    print(f"  - Sample Rate: {config.sample_rate}")
    print(f"  - Debug Mode: {config.debug}")
    print(f"  - Max Latency: {config.max_latency_ms}ms")
    print(f"  - Buffer Size: {config.buffer_size}")


def demo_langfuse_client():
    """Demonstrate Langfuse client initialization and error handling."""
    print("\n" + "="*60)
    print("LANGFUSE CLIENT DEMO")
    print("="*60)
    
    # Test with invalid configuration (should enable fallback mode)
    print("\n1. Testing client with invalid configuration...")
    invalid_config = LangfuseConfig(
        public_key="invalid-key",
        secret_key="invalid-secret",
        enabled=True
    )
    
    client = LangfuseClient(invalid_config)
    print(f"   Client available: {client.is_available}")
    print(f"   Fallback mode: {client._fallback_mode}")
    
    # Test health check
    print("\n2. Testing health check...")
    health_status = client.health_check()
    for key, value in health_status.items():
        print(f"   {key}: {value}")
    
    # Test with disabled configuration
    print("\n3. Testing client with disabled configuration...")
    disabled_config = LangfuseConfig(
        public_key="pk-test",
        secret_key="sk-test",
        enabled=False
    )
    
    disabled_client = LangfuseClient(disabled_config)
    print(f"   Client available: {disabled_client.is_available}")
    print(f"   Configuration enabled: {disabled_client.config.enabled}")


def demo_integration_service():
    """Demonstrate Langfuse integration service functionality."""
    print("\n" + "="*60)
    print("LANGFUSE INTEGRATION SERVICE DEMO")
    print("="*60)
    
    # Get the global integration service
    service = get_langfuse_integration()
    
    print(f"✓ Integration service initialized")
    print(f"  - Service enabled: {service.is_enabled()}")
    
    # Test health status
    print("\n1. Health Status:")
    health_status = service.get_health_status()
    for key, value in health_status.items():
        print(f"   {key}: {value}")
    
    # Simulate creating traces and spans (will work in fallback mode)
    print("\n2. Simulating workflow tracing...")
    
    # Create a simulation trace
    event_data = {
        "event_type": "demand_spike",
        "trigger_source": "iot_sensor",
        "product_id": "PROD-123",
        "location": "store_001",
        "timestamp": datetime.now().isoformat()
    }
    
    trace_id = service.create_simulation_trace(event_data)
    print(f"   Created simulation trace: {trace_id}")
    
    # Start agent spans
    inventory_span_id = service.start_agent_span(
        agent_id="inventory_agent",
        operation="forecast_demand",
        parent_trace_id=trace_id,
        inputs={"product_id": "PROD-123", "days_ahead": 7}
    )
    print(f"   Started inventory agent span: {inventory_span_id}")
    
    pricing_span_id = service.start_agent_span(
        agent_id="pricing_agent", 
        operation="calculate_optimal_price",
        parent_trace_id=trace_id,
        inputs={"product_id": "PROD-123", "demand_forecast": 150}
    )
    print(f"   Started pricing agent span: {pricing_span_id}")
    
    # Log agent decisions
    inventory_decision = {
        "decision_type": "restock_recommendation",
        "inputs": {"current_stock": 50, "forecast": 150},
        "outputs": {"recommendation": "restock", "quantity": 100},
        "confidence": 0.85,
        "reasoning": "Demand forecast exceeds current stock by 100 units"
    }
    
    service.log_agent_decision("inventory_agent", inventory_decision)
    print(f"   Logged inventory agent decision")
    
    # End spans
    service.end_agent_span(
        inventory_span_id,
        outcome={"recommendation": "restock", "quantity": 100},
        status="success"
    )
    print(f"   Ended inventory agent span")
    
    service.end_agent_span(
        pricing_span_id,
        outcome={"optimal_price": 29.99, "markdown": 0.15},
        status="success"
    )
    print(f"   Ended pricing agent span")
    
    # Track collaboration
    collaboration_trace_id = service.track_collaboration(
        workflow_id="inventory_pricing_sync",
        participating_agents=["inventory_agent", "pricing_agent"],
        workflow_data={"trigger": "demand_spike", "priority": "high"}
    )
    print(f"   Started collaboration tracking: {collaboration_trace_id}")
    
    # Finalize traces
    service.finalize_trace(trace_id, {"status": "completed", "agents_involved": 2})
    service.finalize_trace(collaboration_trace_id, {"resolution": "successful"})
    print(f"   Finalized traces")


def demo_context_manager():
    """Demonstrate the trace operation context manager."""
    print("\n" + "="*60)
    print("CONTEXT MANAGER DEMO")
    print("="*60)
    
    service = get_langfuse_integration()
    
    # Use context manager for automatic trace management
    print("\n1. Using trace_operation context manager...")
    
    with service.trace_operation(
        "demand_forecasting",
        agent_id="inventory_agent",
        inputs={"product_id": "PROD-456", "historical_days": 30}
    ) as trace_info:
        print(f"   Trace enabled: {trace_info['enabled']}")
        print(f"   Span ID: {trace_info['span_id']}")
        
        # Simulate some work
        import time
        time.sleep(0.1)
        
        print("   Performed demand forecasting operation")
    
    print("   Context manager automatically cleaned up trace")


def demo_global_instances():
    """Demonstrate global instance management."""
    print("\n" + "="*60)
    print("GLOBAL INSTANCES DEMO")
    print("="*60)
    
    # Test singleton behavior
    client1 = get_langfuse_client()
    client2 = get_langfuse_client()
    
    service1 = get_langfuse_integration()
    service2 = get_langfuse_integration()
    
    print(f"✓ Langfuse client singleton: {client1 is client2}")
    print(f"✓ Integration service singleton: {service1 is service2}")
    
    print(f"\nClient instances:")
    print(f"  - Client 1 ID: {id(client1)}")
    print(f"  - Client 2 ID: {id(client2)}")
    
    print(f"\nService instances:")
    print(f"  - Service 1 ID: {id(service1)}")
    print(f"  - Service 2 ID: {id(service2)}")


def main():
    """Run all demonstrations."""
    print("LANGFUSE INTEGRATION FOUNDATION DEMO")
    print("This demo shows the Langfuse integration working in fallback mode")
    print("(no actual Langfuse connection required)")
    
    try:
        demo_langfuse_config()
        demo_langfuse_client()
        demo_integration_service()
        demo_context_manager()
        demo_global_instances()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nThe Langfuse integration foundation is working correctly.")
        print("To enable actual tracing, set the following environment variables:")
        print("  - LANGFUSE_PUBLIC_KEY=your_public_key")
        print("  - LANGFUSE_SECRET_KEY=your_secret_key")
        print("  - LANGFUSE_HOST=https://cloud.langfuse.com (or your host)")
        print("  - LANGFUSE_ENABLED=true")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())