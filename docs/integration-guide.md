# Langfuse Integration Guide

This guide provides comprehensive instructions for setting up and configuring the Langfuse workflow visualization integration in the AutoOps retail optimization system.

## Prerequisites

Before integrating Langfuse, ensure you have:

1. **Python 3.9 or higher**
2. **Langfuse account** with API credentials
3. **Network connectivity** to Langfuse services
4. **AWS Bedrock access** for agent operations

## Installation and Setup

### 1. Install Dependencies

The Langfuse integration uses the official Langfuse Python SDK v3:

```bash
# Using uv (recommended)
uv add langfuse

# Or using pip
pip install langfuse>=3.0.0
```

### 2. Configure Environment Variables

Add the following variables to your `.env` file:

```bash
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxx
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=true

# Optional Configuration
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_DEBUG=false
LANGFUSE_FLUSH_INTERVAL=5
LANGFUSE_MAX_RETRIES=3
LANGFUSE_MAX_LATENCY_MS=100
LANGFUSE_ENABLE_SAMPLING=true
LANGFUSE_BUFFER_SIZE=1000
```

### 3. Verify Installation

Test your setup using the provided demo script:

```bash
cd /home/er/Documents/project/aws-ai-hackathon-autoops
python examples/langfuse_foundation_demo.py
```

## Configuration Management

### Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `LANGFUSE_PUBLIC_KEY` | Your Langfuse public key | - | Yes |
| `LANGFUSE_SECRET_KEY` | Your Langfuse secret key | - | Yes |
| `LANGFUSE_HOST` | Langfuse service URL | `https://cloud.langfuse.com` | No |
| `LANGFUSE_ENABLED` | Enable/disable tracing | `true` | No |
| `LANGFUSE_SAMPLE_RATE` | Sampling rate (0.0-1.0) | `1.0` | No |
| `LANGFUSE_DEBUG` | Enable debug logging | `false` | No |
| `LANGFUSE_FLUSH_INTERVAL` | Flush interval in seconds | `5` | No |
| `LANGFUSE_MAX_RETRIES` | Maximum retry attempts | `3` | No |
| `LANGFUSE_MAX_LATENCY_MS` | Max acceptable latency | `100` | No |
| `LANGFUSE_ENABLE_SAMPLING` | Enable intelligent sampling | `true` | No |
| `LANGFUSE_BUFFER_SIZE` | Local buffer size | `1000` | No |

### Configuration Classes

The integration provides several configuration classes:

#### LangfuseConfig

Main configuration class for Langfuse settings:

```python
from config.langfuse_config import LangfuseConfig

config = LangfuseConfig(
    public_key="pk-lf-xxxxxxxxxxxx",
    secret_key="sk-lf-xxxxxxxxxxxx",
    host="https://cloud.langfuse.com",
    enabled=True,
    sample_rate=0.8,
    debug=False
)
```

#### Integration Service Configuration

Configure the integration service behavior:

```python
from config.langfuse_integration import get_langfuse_integration

service = get_langfuse_integration()
service.configure(
    enable_sampling=True,
    max_latency_ms=150,
    buffer_size=2000
)
```

## Basic Usage

### Initializing the Integration

```python
from config.langfuse_integration import get_langfuse_integration

# Get the global integration service
service = get_langfuse_integration()

# Check if service is available
if service.is_available:
    print("Langfuse integration is ready")
else:
    print("Langfuse integration is in fallback mode")
```

### Creating Traces

#### Simulation Event Traces

```python
# Create a trace for a simulation event
event_data = {
    "type": "demand_spike",
    "source": "iot_sensor",
    "product_id": "PROD-123",
    "location": "store_001",
    "magnitude": 2.5
}

trace_id = service.create_simulation_trace(event_data)
print(f"Created trace: {trace_id}")
```

#### Agent Operation Traces

```python
# Start an agent operation span
span_id = service.start_agent_span(
    agent_id="inventory_agent",
    operation="forecast_demand",
    parent_trace_id=trace_id,
    input_data={
        "product_id": "PROD-123",
        "historical_days": 30,
        "external_factors": ["weather", "promotions"]
    }
)
```

#### Context Manager Approach

For automatic trace lifecycle management:

```python
# Automatic trace management with context manager
with service.trace_operation(
    operation_name="demand_forecasting",
    agent_id="inventory_agent",
    input_data={"product_id": "PROD-123", "days_ahead": 7},
    metadata={"priority": "high", "model": "claude-3"}
) as trace_info:
    # Your agent logic here
    forecast_result = perform_demand_forecast()

    # Add output data to trace
    trace_info.add_output_data({
        "forecast": forecast_result,
        "confidence": 0.85
    })
```

### Logging Agent Decisions

```python
# Log an agent decision
decision_data = {
    "decision_type": "restock_recommendation",
    "inputs": {
        "current_stock": 50,
        "forecasted_demand": 200
    },
    "outputs": {
        "recommended_quantity": 150,
        "urgency": "high",
        "reasoning": "Demand spike detected, current stock insufficient"
    },
    "confidence": 0.85,
    "processing_time_ms": 125
}

service.log_agent_decision("inventory_agent", decision_data)
```

### Tracking Collaboration

```python
# Track multi-agent collaboration
collaboration_data = {
    "workflow_type": "inventory_pricing_sync",
    "participating_agents": ["inventory_agent", "pricing_agent"],
    "coordination_messages": [
        {
            "from": "inventory_agent",
            "to": "pricing_agent",
            "message": "Stock levels updated",
            "timestamp": "2024-01-15T10:30:15Z"
        }
    ],
    "conflicts_detected": [],
    "resolution_outcome": "successful"
}

collab_trace_id = service.track_collaboration(collaboration_data)
```

### Finalizing Traces

```python
# Finalize a trace with outcome
final_outcome = {
    "status": "completed",
    "total_agents_involved": 2,
    "processing_time_ms": 250,
    "success": True,
    "business_impact": "optimized_inventory_levels"
}

service.finalize_trace(trace_id, final_outcome)
```

## Advanced Usage

### Custom Trace Attributes

Add custom metadata to traces:

```python
# Add custom attributes to traces
service.add_trace_attribute(trace_id, "business_unit", "retail")
service.add_trace_attribute(trace_id, "priority", "high")
service.add_trace_attribute(trace_id, "model_version", "v2.1")
```

### Error Handling

```python
try:
    # Your agent operation
    result = perform_risky_operation()
except Exception as e:
    # Log error with context
    service.log_error(
        trace_id=trace_id,
        error=e,
        context={
            "operation": "demand_forecast",
            "input_data": {"product_id": "PROD-123"},
            "error_type": "data_processing"
        }
    )
    raise
```

### Performance Monitoring

```python
# Monitor operation performance
with service.trace_operation("complex_calculation") as trace:
    start_time = time.time()

    # Your operation
    result = complex_calculation()

    processing_time = time.time() - start_time
    trace.add_performance_metric("processing_time_ms", processing_time * 1000)
    trace.add_performance_metric("memory_usage_mb", get_memory_usage())
```

### Batch Operations

For high-throughput scenarios:

```python
# Batch multiple operations
operations = [
    {"agent": "inventory", "operation": "forecast", "data": {...}},
    {"agent": "pricing", "operation": "optimize", "data": {...}},
    {"agent": "promotion", "operation": "plan", "data": {...}}
]

batch_trace_id = service.start_batch_trace("weekly_optimization", operations)

for op in operations:
    with service.trace_operation(
        op["operation"],
        agent_id=op["agent"],
        parent_trace_id=batch_trace_id
    ) as trace:
        # Process each operation
        process_operation(op)
```

## Integration with Existing Agents

### Inventory Agent Integration

```python
class InventoryAgent:
    def __init__(self):
        self.service = get_langfuse_integration()

    def forecast_demand(self, product_id: str, days_ahead: int):
        with self.service.trace_operation(
            "demand_forecast",
            agent_id="inventory_agent",
            input_data={"product_id": product_id, "days_ahead": days_ahead}
        ) as trace:
            # Your forecasting logic
            forecast = self._calculate_forecast(product_id, days_ahead)

            trace.add_output_data({
                "forecast_result": forecast,
                "confidence": 0.85
            })

            return forecast
```

### Pricing Agent Integration

```python
class PricingAgent:
    def calculate_optimal_price(self, product_id: str, market_conditions: dict):
        with self.service.trace_operation(
            "price_optimization",
            agent_id="pricing_agent",
            input_data={
                "product_id": product_id,
                "market_conditions": market_conditions
            }
        ) as trace:
            # Your pricing logic
            optimal_price = self._optimize_price(product_id, market_conditions)

            trace.add_output_data({
                "optimal_price": optimal_price,
                "profit_margin": 0.25,
                "competitive_position": "premium"
            })

            return optimal_price
```

### Promotion Agent Integration

```python
class PromotionAgent:
    def create_campaign(self, target_products: list, campaign_type: str):
        with self.service.trace_operation(
            "campaign_creation",
            agent_id="promotion_agent",
            input_data={
                "target_products": target_products,
                "campaign_type": campaign_type
            }
        ) as trace:
            # Your campaign logic
            campaign = self._design_campaign(target_products, campaign_type)

            trace.add_output_data({
                "campaign_id": campaign.id,
                "expected_impact": "15%_sales_increase",
                "target_audience": "premium_customers"
            })

            return campaign
```

## Dashboard Integration

### Accessing Langfuse Dashboard

1. **Log into Langfuse**: Navigate to your Langfuse instance
2. **Select Project**: Choose the project configured in your environment
3. **View Traces**: Access traces in the "Traces" section
4. **Monitor Performance**: Check the "Dashboard" for metrics

### Custom Dashboards

Create custom views for retail optimization:

```python
# Configure custom dashboard views
from config.langfuse_dashboard_config import DashboardConfig

dashboard_config = DashboardConfig(
    name="Retail Optimization Dashboard",
    filters={
        "agent_types": ["inventory_agent", "pricing_agent", "promotion_agent"],
        "time_range": "last_24_hours",
        "trace_types": ["simulation_event", "agent_operation", "collaboration"]
    },
    metrics=[
        "average_response_time",
        "success_rate",
        "throughput",
        "error_rate"
    ]
)
```

## Best Practices

### 1. Trace Naming Conventions

- Use descriptive, hierarchical names: `agent_type.operation.sub_operation`
- Include relevant context: `inventory_agent.forecast_demand.by_product`
- Be consistent across similar operations

### 2. Performance Considerations

- Use sampling for high-frequency operations
- Set appropriate flush intervals
- Monitor tracing overhead
- Use async processing for non-critical traces

### 3. Security

- Never log sensitive data (PII, credentials)
- Use data masking for user information
- Configure appropriate data retention policies
- Use HTTPS for all communications

### 4. Error Handling

- Always use try/catch blocks around tracing operations
- Log errors with sufficient context
- Implement graceful degradation
- Monitor trace success rates

### 5. Monitoring

- Track tracing performance impact
- Monitor trace success rates
- Alert on high error rates
- Review trace volume and costs

## Troubleshooting

See the [Troubleshooting Guide](./troubleshooting.md) for common issues and solutions.

## Next Steps

1. **Review Examples**: Check the [examples directory](./examples/) for complete implementations
2. **Customize Integration**: Adapt the integration to your specific use cases
3. **Monitor Performance**: Set up monitoring for the tracing system
4. **Extend Coverage**: Add tracing to additional agents and operations