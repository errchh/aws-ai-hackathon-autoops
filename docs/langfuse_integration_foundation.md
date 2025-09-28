# Langfuse Integration Foundation

This document describes the Langfuse integration foundation that provides workflow visualization for the AutoOps retail optimization system.

## Overview

The Langfuse integration foundation consists of three main components:

1. **Configuration Management** (`config/langfuse_config.py`)
2. **Integration Service** (`config/langfuse_integration.py`)
3. **Settings Integration** (`config/settings.py`)

## Features

### Langfuse v3 Support
- Full compatibility with Langfuse Python SDK v3
- OpenTelemetry-based tracing
- Enhanced configuration options

### Graceful Degradation
- System continues operating without tracing if Langfuse is unavailable
- Automatic fallback mode when connection fails
- No impact on core system functionality

### Error Handling
- Comprehensive error handling for connection failures
- Retry logic and timeout management
- Detailed logging for troubleshooting

## Configuration

### Environment Variables

Add the following variables to your `.env` file:

```bash
# Required
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here

# Optional (with defaults)
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED=true
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_DEBUG=false
LANGFUSE_FLUSH_INTERVAL=5.0
LANGFUSE_FLUSH_AT=15
LANGFUSE_TIMEOUT=60
LANGFUSE_ENVIRONMENT=development
LANGFUSE_RELEASE=0.1.0
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT=4
```

### Advanced Configuration

```bash
# Performance tuning
LANGFUSE_MAX_LATENCY_MS=100
LANGFUSE_ENABLE_SAMPLING=true
LANGFUSE_BUFFER_SIZE=1000
LANGFUSE_MAX_RETRIES=3

# Optional features
LANGFUSE_BLOCKED_INSTRUMENTATION_SCOPES=scope1,scope2
LANGFUSE_ADDITIONAL_HEADERS={"X-Custom-Header": "value"}
```

## Usage

### Basic Usage

```python
from config.langfuse_integration import get_langfuse_integration

# Get the integration service
service = get_langfuse_integration()

# Create a simulation trace
trace_id = service.create_simulation_trace({
    "type": "demand_spike",
    "source": "iot_sensor",
    "product_id": "PROD_123"
})

# Start an agent span
span_id = service.start_agent_span(
    "inventory_agent",
    "forecast_demand",
    parent_trace_id=trace_id,
    input_data={"product_id": "PROD_123"}
)

# End the span with results
service.end_agent_span(span_id, {
    "forecast_result": "high_demand",
    "confidence": 0.85
})

# Finalize the trace
service.finalize_trace(trace_id, {
    "final_outcome": "restock_initiated"
})
```

### Context Manager Usage

```python
# Use context manager for automatic trace management
with service.trace_operation("market_analysis") as trace:
    # Your operation code here
    pass  # Trace is automatically finalized
```

### Agent Decision Logging

```python
service.log_agent_decision(
    "pricing_agent",
    {
        "type": "price_optimization",
        "inputs": {"current_price": 29.99, "competitor_prices": [27.99, 31.99]},
        "outputs": {"recommended_price": 28.99},
        "confidence": 0.92,
        "reasoning": "Competitive positioning strategy"
    }
)
```

### Collaboration Tracking

```python
# Track multi-agent workflows
collab_trace_id = service.track_collaboration(
    "inventory_pricing_sync",
    ["inventory_agent", "pricing_agent"],
    {"trigger": "stock_level_change"}
)
```

## Health Monitoring

```python
# Check system health
health = service.health_check()
print(health)
# Output: {
#     "initialized": True,
#     "available": True,
#     "connection_status": "healthy",
#     "active_traces": 2,
#     "active_spans": 1,
#     "integration_service": "ready"
# }
```

## Testing

Run the integration tests:

```bash
uv run python -m pytest tests/test_langfuse_integration.py -v
```

Run the demonstration:

```bash
uv run python examples/langfuse_foundation_demo.py
```

## Architecture

### Component Relationships

```
┌─────────────────────┐
│   Application       │
│   Components        │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│ LangfuseIntegration │
│ Service             │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│ LangfuseClient      │
│ (Wrapper)           │
└─────────┬───────────┘
          │
┌─────────▼───────────┐
│ Langfuse SDK v3     │
│ (OpenTelemetry)     │
└─────────────────────┘
```

### Key Classes

- **`LangfuseConfig`**: Configuration data class with v3 parameters
- **`LangfuseClient`**: Client wrapper with error handling and health checks
- **`LangfuseIntegrationService`**: High-level service for trace management
- **`LangfuseSettings`**: Pydantic settings for environment configuration

## Error Handling

The integration includes comprehensive error handling:

1. **Connection Errors**: Graceful degradation when Langfuse is unavailable
2. **Configuration Errors**: Clear error messages for invalid settings
3. **Runtime Errors**: Automatic retry and fallback mechanisms
4. **Performance Errors**: Sampling and throttling for high-load scenarios

## Performance Considerations

- **Asynchronous Processing**: Traces are processed asynchronously to minimize latency
- **Intelligent Sampling**: Configurable sampling rates to reduce overhead
- **Connection Pooling**: Efficient HTTP connection management
- **Graceful Degradation**: Zero impact when tracing is disabled

## Next Steps

1. **Agent Instrumentation**: Add tracing to individual agents (Task 2)
2. **Simulation Integration**: Connect simulation events to traces (Task 3)
3. **Dashboard Configuration**: Set up Langfuse dashboard views
4. **Performance Optimization**: Implement sampling strategies

## Troubleshooting

### Common Issues

1. **"Langfuse credentials not found"**
   - Ensure `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set in `.env`

2. **"Connection test failed"**
   - Check network connectivity to Langfuse host
   - Verify credentials are correct
   - Check firewall settings

3. **High latency impact**
   - Reduce `LANGFUSE_SAMPLE_RATE`
   - Enable `LANGFUSE_ENABLE_SAMPLING`
   - Increase `LANGFUSE_FLUSH_INTERVAL`

### Debug Mode

Enable debug mode for detailed logging:

```bash
LANGFUSE_DEBUG=true
```

This will provide detailed information about trace creation, connection status, and error conditions.