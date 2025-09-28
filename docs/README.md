# Langfuse Integration Documentation

This directory contains comprehensive documentation for the Langfuse workflow visualization integration in the AutoOps retail optimization system.

## Overview

The Langfuse integration provides comprehensive observability and tracing capabilities for the multi-agent retail optimization system. It tracks agent interactions, decision-making processes, and performance metrics triggered by the simulation engine, enabling real-time monitoring and analysis of system behavior.

## Documentation Structure

### Core Documentation
- **[Integration Guide](./integration-guide.md)** - Complete setup and configuration guide
- **[API Reference](./api-reference.md)** - Detailed API documentation for all integration components
- **[Architecture Overview](./architecture.md)** - System architecture and design principles

### User Guides
- **[Dashboard Guide](./dashboard-guide.md)** - How to use the Langfuse dashboard for monitoring
- **[Workflow Visualization](./workflow-visualization.md)** - Understanding trace visualizations and workflows
- **[Performance Monitoring](./performance-monitoring.md)** - Monitoring system performance and metrics

### Developer Resources
- **[Developer Guide](./developer-guide.md)** - Extending tracing to new agents and components
- **[Troubleshooting Guide](./troubleshooting.md)** - Common issues and solutions
- **[Examples](./examples/)** - Code examples and sample implementations

### Examples Directory
- **[Basic Examples](./examples/basic-usage.md)** - Simple usage examples
- **[Advanced Examples](./examples/advanced-usage.md)** - Complex workflow examples
- **[Troubleshooting Examples](./examples/troubleshooting.md)** - Examples of common issues and solutions

## Quick Start

1. **Configure Langfuse credentials** in your `.env` file:
   ```bash
   LANGFUSE_PUBLIC_KEY=your_public_key
   LANGFUSE_SECRET_KEY=your_secret_key
   LANGFUSE_HOST=https://cloud.langfuse.com
   LANGFUSE_ENABLED=true
   ```

2. **Initialize the integration** in your application:
   ```python
   from config.langfuse_integration import get_langfuse_integration

   service = get_langfuse_integration()
   ```

3. **Start tracing workflows**:
   ```python
   # Create a simulation trace
   trace_id = service.create_simulation_trace(event_data)

   # Start agent operations
   with service.trace_operation("agent_task", agent_id="my_agent") as trace:
       # Your agent logic here
       pass
   ```

## Key Features

- **Real-time Workflow Tracing**: Track agent interactions and decision paths
- **Performance Monitoring**: Monitor response times, success rates, and throughput
- **Error Tracking**: Capture and analyze errors with full context
- **Dashboard Integration**: Visualize workflows and metrics in Langfuse dashboard
- **Graceful Degradation**: System continues operating even if Langfuse is unavailable
- **Configurable Sampling**: Control tracing overhead with intelligent sampling

## Requirements

- Python 3.9+
- Langfuse Python SDK v3
- AWS Bedrock access for agent operations
- Network connectivity to Langfuse service

## Support

For issues and questions:
- Check the [Troubleshooting Guide](./troubleshooting.md)
- Review the [Examples](./examples/) for common patterns
- Report issues at https://github.com/sst/opencode/issues