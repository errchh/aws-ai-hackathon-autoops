# Langfuse Integration Examples

This directory contains practical examples demonstrating how to use the Langfuse integration in various scenarios.

## Example Categories

### Basic Usage Examples
- **[Simple Tracing](./basic-usage.md)** - Basic trace creation and management
- **[Agent Integration](./agent-integration.md)** - Adding tracing to existing agents
- **[Error Handling](./error-handling.md)** - Proper error handling with tracing

### Advanced Examples
- **[Workflow Tracing](./workflow-tracing.md)** - Complex multi-step workflows
- **[Performance Monitoring](./performance-monitoring.md)** - Monitoring and optimization
- **[Custom Dashboards](./custom-dashboards.md)** - Creating custom visualizations

### Troubleshooting Examples
- **[Common Issues](./troubleshooting.md)** - Solutions to common problems
- **[Debug Examples](./debug-examples.md)** - Debugging and diagnostics

## Running Examples

Most examples can be run directly:

```bash
cd /home/er/Documents/project/aws-ai-hackathon-autoops
python docs/examples/basic-usage/simple_tracing.py
```

Some examples may require:
- Langfuse credentials in `.env`
- Specific dependencies
- Mock data or test scenarios

## Key Concepts Demonstrated

- **Trace Lifecycle**: Creating, updating, and finalizing traces
- **Context Management**: Using context managers for automatic cleanup
- **Error Handling**: Graceful degradation when tracing fails
- **Performance Optimization**: Sampling and async processing
- **Data Masking**: Protecting sensitive information
- **Custom Metadata**: Adding domain-specific information to traces

## Best Practices Shown

- **Consistent Naming**: Using descriptive trace and span names
- **Rich Context**: Including relevant metadata and context
- **Error Recovery**: Handling tracing failures without breaking business logic
- **Performance Awareness**: Minimizing tracing overhead
- **Security**: Proper data masking and access control

## Contributing Examples

When adding new examples:

1. **Follow Existing Patterns**: Use similar structure to existing examples
2. **Include Documentation**: Add docstrings and comments explaining the example
3. **Handle Edge Cases**: Show both success and failure scenarios
4. **Test Independently**: Examples should run without external dependencies when possible
5. **Update This README**: Add your example to the appropriate category

## Example Structure

Each example should include:

```python
#!/usr/bin/env python3
"""
Example demonstrating [specific concept].

This example shows how to [brief description].
"""

# Imports
import sys
sys.path.insert(0, '../..')  # Adjust path as needed

# Example code with detailed comments

if __name__ == "__main__":
    # Demonstration code
    pass
```

See existing examples for detailed patterns and conventions.