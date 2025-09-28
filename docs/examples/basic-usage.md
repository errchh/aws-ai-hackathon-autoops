# Basic Usage Examples

This section demonstrates the fundamental usage patterns for the Langfuse integration.

## Simple Tracing Example

```python
#!/usr/bin/env python3
"""
Basic tracing example demonstrating core Langfuse integration concepts.

This example shows how to:
- Initialize the integration service
- Create traces and spans
- Log agent decisions
- Handle errors gracefully
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.langfuse_integration import get_langfuse_integration
from datetime import datetime

def basic_tracing_example():
    """Demonstrate basic tracing functionality."""
    print("=== Basic Tracing Example ===")

    # Get the integration service
    service = get_langfuse_integration()

    # Check if service is available
    print(f"Service available: {service.is_available}")

    if not service.is_available:
        print("Langfuse not configured, running in fallback mode")
        return

    # Create a simulation trace
    print("\n1. Creating simulation trace...")
    event_data = {
        "type": "demand_spike",
        "source": "iot_sensor",
        "product_id": "PROD-123",
        "location": "store_001",
        "magnitude": 2.5
    }

    trace_id = service.create_simulation_trace(event_data)
    print(f"Created trace: {trace_id}")

    # Start agent spans
    print("\n2. Starting agent operations...")

    # Inventory agent span
    inventory_span = service.start_agent_span(
        agent_id="inventory_agent",
        operation="forecast_demand",
        parent_trace_id=trace_id,
        input_data={
            "product_id": "PROD-123",
            "historical_days": 30,
            "external_factors": ["weather", "promotions"]
        }
    )
    print(f"Started inventory span: {inventory_span}")

    # Pricing agent span
    pricing_span = service.start_agent_span(
        agent_id="pricing_agent",
        operation="calculate_optimal_price",
        parent_trace_id=trace_id,
        input_data={
            "product_id": "PROD-123",
            "demand_forecast": 150
        }
    )
    print(f"Started pricing span: {pricing_span}")

    # Log agent decisions
    print("\n3. Logging agent decisions...")

    inventory_decision = {
        "decision_type": "restock_recommendation",
        "inputs": {"current_stock": 50, "forecasted_demand": 200},
        "outputs": {"recommended_quantity": 150, "urgency": "high"},
        "confidence": 0.85,
        "reasoning": "Demand spike detected, current stock insufficient"
    }

    service.log_agent_decision("inventory_agent", inventory_decision)
    print("Logged inventory decision")

    # End spans with outcomes
    print("\n4. Completing operations...")

    service.end_agent_span(
        inventory_span,
        outcome={"forecast_result": "high_demand", "recommended_action": "immediate_restock"}
    )
    print("Completed inventory operation")

    service.end_agent_span(
        pricing_span,
        outcome={"optimal_price": 29.99, "markdown": 0.15}
    )
    print("Completed pricing operation")

    # Track collaboration
    print("\n5. Tracking collaboration...")

    collab_trace_id = service.track_collaboration(
        workflow_id="inventory_pricing_sync",
        participating_agents=["inventory_agent", "pricing_agent"],
        workflow_data={"trigger": "demand_spike", "priority": "high"}
    )
    print(f"Created collaboration trace: {collab_trace_id}")

    # Finalize traces
    print("\n6. Finalizing traces...")

    service.finalize_trace(trace_id, {
        "status": "completed",
        "total_agents_involved": 2,
        "processing_time_ms": 250
    })

    if collab_trace_id:
        service.finalize_trace(collab_trace_id, {
            "collaboration_result": "price_adjusted",
            "agents_coordinated": 2
        })

    print("All traces finalized")

    # Show service health
    print("\n7. Service health check...")
    health = service.health_check()
    print(f"Active traces: {health.get('active_traces', 0)}")
    print(f"Active spans: {health.get('active_spans', 0)}")

def context_manager_example():
    """Demonstrate context manager usage."""
    print("\n=== Context Manager Example ===")

    service = get_langfuse_integration()

    # Using context manager for automatic trace management
    with service.trace_operation(
        operation_name="demand_forecasting",
        input_data={"product_id": "PROD-456", "historical_days": 30},
        metadata={"analysis_type": "competitive_pricing"}
    ) as trace:
        print(f"Trace enabled: {trace is not None}")

        # Simulate some work
        import time
        time.sleep(0.1)

        print("Performed demand forecasting operation")

        # Add custom data to trace if available
        if trace:
            # This would add data to the trace in a real implementation
            print("Would add output data to trace")

    print("Context manager automatically cleaned up trace")

def error_handling_example():
    """Demonstrate error handling with tracing."""
    print("\n=== Error Handling Example ===")

    service = get_langfuse_integration()

    try:
        with service.trace_operation(
            operation_name="risky_operation",
            input_data={"risk_level": "high"},
            metadata={"agent_id": "test_agent"}
        ) as trace:
            print("Starting risky operation...")

            # Simulate an error
            raise ValueError("Something went wrong!")

    except ValueError as e:
        print(f"Caught error: {e}")

        # Log error in trace if available
        if service.is_available:
            try:
                service.log_error(
                    trace_id=None,  # Would be the trace ID in real implementation
                    error=e,
                    context={"operation": "risky_operation"}
                )
            except:
                pass  # Ignore tracing errors

        print("Error handled gracefully")

def main():
    """Run all examples."""
    print("Langfuse Integration Basic Usage Examples")
    print("=" * 50)

    basic_tracing_example()
    context_manager_example()
    error_handling_example()

    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Configure actual Langfuse credentials")
    print("2. Run these examples with tracing enabled")
    print("3. Check the Langfuse dashboard to see traces")
    print("4. Explore advanced examples")

if __name__ == "__main__":
    main()
```

## Key Concepts Demonstrated

### 1. Service Initialization
```python
service = get_langfuse_integration()
```

### 2. Trace Creation
```python
trace_id = service.create_simulation_trace(event_data)
```

### 3. Span Management
```python
span_id = service.start_agent_span(agent_id, operation, parent_trace_id, input_data)
service.end_agent_span(span_id, outcome)
```

### 4. Decision Logging
```python
service.log_agent_decision(agent_id, decision_data)
```

### 5. Collaboration Tracking
```python
collab_trace_id = service.track_collaboration(workflow_id, agents, workflow_data)
```

### 6. Context Managers
```python
with service.trace_operation("operation_name") as trace:
    # Your code here
```

### 7. Error Handling
```python
try:
    # Traced operation
except Exception as e:
    # Handle error and log to trace
```

## Expected Output

When run with Langfuse properly configured:

```
=== Basic Tracing Example ===
Service available: True
Created trace: sim_20240115_103000_123456
Started inventory span: inventory_agent_forecast_demand_20240115_103000_123457
Started pricing span: pricing_agent_calculate_optimal_price_20240115_103000_123458
Logged inventory decision
Completed inventory operation
Completed pricing operation
Created collaboration trace: collab_inventory_pricing_sync_20240115_103000_123459
All traces finalized
Active traces: 0
Active spans: 0
```

When run without Langfuse configuration:

```
=== Basic Tracing Example ===
Service available: False
Langfuse not configured, running in fallback mode
```

## Common Patterns

### Pattern 1: Operation with Automatic Cleanup
```python
with service.trace_operation("my_operation") as trace:
    result = perform_operation()
    # Trace automatically finalized when exiting context
```

### Pattern 2: Manual Span Management
```python
span_id = service.start_agent_span("agent", "operation", trace_id)
try:
    result = perform_operation()
finally:
    service.end_agent_span(span_id, {"result": result})
```

### Pattern 3: Error-Aware Tracing
```python
with service.trace_operation("risky_operation") as trace:
    try:
        result = perform_operation()
    except Exception as e:
        if trace:
            trace.add_error_data({"error": str(e)})
        raise
```

These basic examples provide the foundation for understanding how to integrate Langfuse tracing into your applications. The patterns shown here can be extended and customized for more complex use cases.