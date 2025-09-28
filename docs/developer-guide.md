# Developer Guide: Extending Tracing to New Agents

This guide provides instructions for developers who want to extend the Langfuse tracing integration to new agents or components in the AutoOps retail optimization system.

## Overview

The Langfuse integration provides a flexible framework for adding observability to any component. The key components are:

- **Integration Service**: Central coordination of tracing
- **Agent Tracers**: Decorators and context managers for agent operations
- **Data Masking**: Automatic protection of sensitive data
- **Metrics Collection**: Performance and usage tracking

## Basic Integration Pattern

### 1. Import Required Modules

```python
from config.langfuse_integration import get_langfuse_integration
from functools import wraps
from typing import Optional, Dict, Any
```

### 2. Get Integration Service

```python
class MyNewAgent:
    def __init__(self):
        self.service = get_langfuse_integration()
        self.agent_id = "my_new_agent"
```

### 3. Add Tracing to Methods

```python
class MyNewAgent:
    def __init__(self):
        self.service = get_langfuse_integration()
        self.agent_id = "my_new_agent"

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request with tracing."""
        with self.service.trace_operation(
            operation_name="process_request",
            input_data=request_data,
            metadata={"agent_id": self.agent_id, "request_type": "standard"}
        ) as trace:
            try:
                # Your processing logic here
                result = self._perform_processing(request_data)

                # Add success metrics to trace
                if trace:
                    trace.add_output_data({
                        "result": result,
                        "processing_time_ms": 150,
                        "success": True
                    })

                return result

            except Exception as e:
                # Log error in trace
                if trace:
                    trace.add_error_data({
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "failed_operation": "process_request"
                    })

                # Re-raise the exception
                raise
```

## Advanced Integration Patterns

### 1. Custom Tracer Decorator

Create reusable decorators for common operations:

```python
def trace_agent_operation(operation_name: str, include_args: bool = True):
    """Decorator to automatically trace agent operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            service = get_langfuse_integration()

            # Prepare input data
            input_data = {}
            if include_args and args:
                input_data["args"] = [str(arg) for arg in args]
            if kwargs:
                input_data["kwargs"] = {k: str(v) for k, v in kwargs.items()}

            with service.trace_operation(
                operation_name=operation_name,
                input_data=input_data,
                metadata={"agent_id": getattr(self, 'agent_id', 'unknown')}
            ) as trace:
                try:
                    result = func(self, *args, **kwargs)

                    if trace:
                        trace.add_output_data({
                            "result": str(result) if not isinstance(result, dict) else result,
                            "success": True
                        })

                    return result

                except Exception as e:
                    if trace:
                        trace.add_error_data({
                            "error": str(e),
                            "error_type": type(e).__name__
                        })
                    raise

        return wrapper
    return decorator

# Usage
class MyNewAgent:
    @trace_agent_operation("analyze_data")
    def analyze_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Your analysis logic
        return {"analysis": "complete", "insights": []}
```

### 2. Context Manager for Complex Operations

For operations that need manual trace control:

```python
class MyNewAgent:
    def complex_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complex workflow with multiple traced steps."""
        service = get_langfuse_integration()

        # Start root trace
        trace_id = service.create_simulation_trace({
            "type": "complex_workflow",
            "agent_id": self.agent_id,
            "workflow_data": workflow_data
        })

        try:
            # Step 1: Data validation
            validation_span = service.start_agent_span(
                agent_id=self.agent_id,
                operation="validate_input",
                parent_trace_id=trace_id,
                input_data={"data": workflow_data}
            )

            is_valid = self._validate_input(workflow_data)
            service.end_agent_span(validation_span, {"valid": is_valid})

            if not is_valid:
                raise ValueError("Invalid input data")

            # Step 2: Processing
            processing_span = service.start_agent_span(
                agent_id=self.agent_id,
                operation="process_data",
                parent_trace_id=trace_id,
                input_data={"validated_data": workflow_data}
            )

            result = self._process_data(workflow_data)
            service.end_agent_span(processing_span, {"result": result})

            # Step 3: Output generation
            output_span = service.start_agent_span(
                agent_id=self.agent_id,
                operation="generate_output",
                parent_trace_id=trace_id,
                input_data={"processed_data": result}
            )

            final_output = self._generate_output(result)
            service.end_agent_span(output_span, {"output": final_output})

            # Finalize root trace
            service.finalize_trace(trace_id, {
                "status": "completed",
                "steps_completed": 3,
                "final_output": final_output
            })

            return final_output

        except Exception as e:
            # Log error and finalize trace
            service.log_error(trace_id, e, {"operation": "complex_workflow"})
            service.finalize_trace(trace_id, {
                "status": "failed",
                "error": str(e)
            })
            raise
```

### 3. Tool-Level Tracing

Trace individual tools or functions within agents:

```python
class MyNewAgent:
    def __init__(self):
        self.service = get_langfuse_integration()
        self.agent_id = "my_new_agent"

    @trace_agent_operation("external_api_call")
    def call_external_api(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call external API with tracing."""
        import requests

        with self.service.trace_operation(
            "http_request",
            input_data={"endpoint": endpoint, "method": "POST"},
            metadata={"tool": "external_api", "agent_id": self.agent_id}
        ) as trace:
            try:
                response = requests.post(endpoint, json=data, timeout=30)

                if trace:
                    trace.add_output_data({
                        "status_code": response.status_code,
                        "response_time_ms": response.elapsed.total_seconds() * 1000,
                        "success": response.status_code < 400
                    })

                return response.json()

            except requests.RequestException as e:
                if trace:
                    trace.add_error_data({
                        "error": str(e),
                        "error_type": "network_error"
                    })
                raise

    @trace_agent_operation("data_transformation")
    def transform_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform raw data with tracing."""
        with self.service.trace_operation(
            "data_processing",
            input_data={"data_size": len(str(raw_data))},
            metadata={"tool": "data_transformer", "agent_id": self.agent_id}
        ) as trace:
            # Your transformation logic
            transformed = self._apply_transformations(raw_data)

            if trace:
                trace.add_output_data({
                    "transformed_size": len(str(transformed)),
                    "transformation_success": True
                })

            return transformed
```

## Error Handling and Resilience

### 1. Graceful Degradation

Ensure your agent works even if tracing fails:

```python
class MyNewAgent:
    def safe_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Operation that works with or without tracing."""
        service = get_langfuse_integration()

        try:
            if service.is_available:
                with service.trace_operation(
                    "safe_operation",
                    input_data=data,
                    metadata={"agent_id": self.agent_id}
                ) as trace:
                    result = self._perform_operation(data)

                    if trace:
                        trace.add_output_data({"success": True, "result": result})

                    return result
            else:
                # Fallback without tracing
                return self._perform_operation(data)

        except Exception as e:
            # Log error if tracing is available
            if service.is_available:
                try:
                    service.log_error(None, e, {"operation": "safe_operation"})
                except:
                    pass  # Ignore tracing errors

            # Re-raise original exception
            raise
```

### 2. Error Context Enhancement

Add rich error context to traces:

```python
class MyNewAgent:
    def operation_with_rich_errors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Operation with detailed error context."""
        service = get_langfuse_integration()

        with self.service.trace_operation(
            "rich_error_operation",
            input_data=data,
            metadata={"agent_id": self.agent_id}
        ) as trace:
            try:
                # Step 1: Validate input
                if not self._validate_input(data):
                    raise ValueError("Input validation failed")

                # Step 2: Process data
                intermediate_result = self._process_data(data)

                # Step 3: Final transformation
                result = self._finalize_result(intermediate_result)

                if trace:
                    trace.add_output_data({
                        "success": True,
                        "steps_completed": 3,
                        "result": result
                    })

                return result

            except ValueError as e:
                # Business logic error
                if trace:
                    trace.add_error_data({
                        "error": str(e),
                        "error_type": "validation_error",
                        "error_category": "business_logic",
                        "recoverable": True,
                        "suggested_action": "Check input data format"
                    })
                raise

            except ConnectionError as e:
                # Infrastructure error
                if trace:
                    trace.add_error_data({
                        "error": str(e),
                        "error_type": "connection_error",
                        "error_category": "infrastructure",
                        "recoverable": True,
                        "retry_after_seconds": 30,
                        "suggested_action": "Check network connectivity"
                    })
                raise

            except Exception as e:
                # Unexpected error
                if trace:
                    trace.add_error_data({
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_category": "unexpected",
                        "recoverable": False,
                        "suggested_action": "Investigate system logs"
                    })
                raise
```

## Performance Optimization

### 1. Sampling for High-Frequency Operations

```python
class HighFrequencyAgent:
    def __init__(self):
        self.service = get_langfuse_integration()
        self.operation_count = 0

    def high_frequency_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """High-frequency operation with intelligent sampling."""
        self.operation_count += 1

        # Sample every 10th operation
        should_trace = (self.operation_count % 10 == 0)

        if should_trace and self.service.is_available:
            with self.service.trace_operation(
                "high_frequency_op",
                input_data={"operation_count": self.operation_count},
                metadata={"agent_id": self.agent_id, "sampled": True}
            ) as trace:
                result = self._perform_operation(data)

                if trace:
                    trace.add_output_data({
                        "sampled": True,
                        "total_operations": self.operation_count
                    })

                return result
        else:
            # Skip tracing for performance
            return self._perform_operation(data)
```

### 2. Async Tracing

For non-blocking tracing in high-performance scenarios:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncAgent:
    def __init__(self):
        self.service = get_langfuse_integration()
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def async_operation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Async operation with non-blocking tracing."""
        loop = asyncio.get_event_loop()

        # Submit tracing to thread pool
        trace_future = loop.run_in_executor(
            self.executor,
            self._perform_traced_operation,
            data
        )

        # Continue with main operation
        result = await self._perform_main_operation(data)

        # Wait for tracing to complete (with timeout)
        try:
            await asyncio.wait_for(trace_future, timeout=1.0)
        except asyncio.TimeoutError:
            # Tracing is taking too long, continue without it
            pass

        return result

    def _perform_traced_operation(self, data: Dict[str, Any]) -> None:
        """Perform tracing in background thread."""
        try:
            if self.service.is_available:
                with self.service.trace_operation(
                    "async_background_trace",
                    input_data=data,
                    metadata={"async": True, "background": True}
                ) as trace:
                    # Simulate background work
                    import time
                    time.sleep(0.1)

                    if trace:
                        trace.add_output_data({"background_success": True})
        except Exception as e:
            # Log tracing errors but don't fail main operation
            print(f"Tracing error: {e}")
```

## Testing Integration

### 1. Unit Tests with Tracing

```python
import unittest
from unittest.mock import Mock, patch
from my_new_agent import MyNewAgent

class TestMyNewAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MyNewAgent()

    @patch('config.langfuse_integration.get_langfuse_integration')
    def test_operation_with_tracing(self, mock_get_service):
        # Mock the integration service
        mock_service = Mock()
        mock_service.is_available = True
        mock_service.trace_operation.return_value.__enter__ = Mock(return_value=Mock())
        mock_service.trace_operation.return_value.__exit__ = Mock(return_value=None)
        mock_get_service.return_value = mock_service

        # Test the operation
        result = self.agent.process_request({"test": "data"})

        # Verify tracing was called
        mock_service.trace_operation.assert_called_once()
        self.assertIsNotNone(result)

    @patch('config.langfuse_integration.get_langfuse_integration')
    def test_operation_without_tracing(self, mock_get_service):
        # Mock service as unavailable
        mock_service = Mock()
        mock_service.is_available = False
        mock_get_service.return_value = mock_service

        # Test the operation still works
        result = self.agent.process_request({"test": "data"})

        # Verify no tracing calls were made
        mock_service.trace_operation.assert_not_called()
        self.assertIsNotNone(result)
```

### 2. Integration Tests

```python
import pytest
from my_new_agent import MyNewAgent

@pytest.mark.integration
class TestAgentIntegration:
    def test_full_workflow_tracing(self):
        """Test complete workflow with tracing enabled."""
        agent = MyNewAgent()

        # This test requires actual Langfuse credentials
        if not agent.service.is_available:
            pytest.skip("Langfuse not available for integration test")

        # Perform workflow
        result = agent.complex_workflow({"input": "test_data"})

        # Verify traces were created
        traces = agent.service.get_recent_traces(agent_id="my_new_agent")
        assert len(traces) > 0

        # Verify trace structure
        workflow_trace = next(
            (t for t in traces if t["name"] == "complex_workflow"),
            None
        )
        assert workflow_trace is not None
        assert workflow_trace["status"] == "completed"
```

## Best Practices

### 1. Trace Naming Conventions

- **Be Descriptive**: `agent_type.operation.sub_operation`
- **Include Context**: `inventory_agent.forecast_demand.by_product`
- **Use Consistent Patterns**: `agent.operation.tool` or `agent.tool.operation`

### 2. Data Management

- **Limit Payload Size**: Keep trace data under 1KB when possible
- **Use Sampling**: For high-frequency operations, trace only a percentage
- **Mask Sensitive Data**: Never include PII or credentials in traces
- **Structure Data**: Use consistent data structures for similar operations

### 3. Performance Considerations

- **Async When Possible**: Use async tracing for non-critical operations
- **Batch Operations**: Group related operations into single traces
- **Monitor Overhead**: Track the performance impact of tracing
- **Graceful Degradation**: Ensure system works without tracing

### 4. Error Handling

- **Never Fail on Tracing Errors**: Tracing failures shouldn't break agent logic
- **Rich Error Context**: Include helpful debugging information in error traces
- **Recovery Actions**: Suggest remediation steps in error metadata
- **Error Classification**: Categorize errors (business_logic, infrastructure, etc.)

### 5. Security

- **Data Masking**: Always mask sensitive information
- **Access Control**: Implement appropriate authorization for trace access
- **Audit Logging**: Track who accesses trace data
- **Compliance**: Ensure traces meet data protection requirements

## Common Patterns

### 1. Agent Initialization Pattern

```python
class BaseAgent:
    """Base class with tracing setup."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.service = get_langfuse_integration()

    def _get_trace_metadata(self) -> Dict[str, Any]:
        """Get standard metadata for traces."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "version": getattr(self, "version", "1.0")
        }
```

### 2. Tool Integration Pattern

```python
class TracedTool:
    """Mixin for adding tracing to tools."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.service = get_langfuse_integration()

    def execute_with_tracing(self, operation: str, **kwargs):
        """Execute tool operation with tracing."""
        with self.service.trace_operation(
            f"{self.tool_name}.{operation}",
            input_data=kwargs,
            metadata={"tool": self.tool_name}
        ) as trace:
            try:
                result = self._execute(operation, **kwargs)

                if trace:
                    trace.add_output_data({
                        "tool_result": result,
                        "execution_time_ms": 100
                    })

                return result
            except Exception as e:
                if trace:
                    trace.add_error_data({"tool_error": str(e)})
                raise
```

### 3. Workflow Pattern

```python
class WorkflowTracer:
    """Helper for tracing complex workflows."""

    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.service = get_langfuse_integration()
        self.trace_id = None
        self.spans = []

    def __enter__(self):
        if self.service.is_available:
            self.trace_id = self.service.create_simulation_trace({
                "type": "workflow",
                "workflow_name": self.workflow_name
            })
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace_id and self.service.is_available:
            status = "failed" if exc_type else "completed"
            self.service.finalize_trace(self.trace_id, {"status": status})

    def add_step(self, step_name: str, **kwargs):
        """Add a workflow step."""
        if self.service.is_available and self.trace_id:
            span_id = self.service.start_agent_span(
                agent_id="workflow",
                operation=step_name,
                parent_trace_id=self.trace_id,
                input_data=kwargs
            )
            self.spans.append(span_id)
            return span_id
        return None

    def complete_step(self, span_id: str, result: Any = None):
        """Complete a workflow step."""
        if span_id and self.service.is_available:
            self.service.end_agent_span(span_id, {"result": result})
```

## Example: Complete Agent Implementation

```python
from typing import Dict, Any, List
from config.langfuse_integration import get_langfuse_integration
from functools import wraps

class RecommendationAgent:
    """Example agent with comprehensive tracing."""

    def __init__(self):
        self.agent_id = "recommendation_agent"
        self.service = get_langfuse_integration()
        self.operation_count = 0

    def trace_operation(self, operation_name: str):
        """Decorator for tracing operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.operation_count += 1

                # Intelligent sampling for high-frequency operations
                should_trace = (
                    self.operation_count % 10 == 0 or  # Sample every 10th
                    operation_name in ["critical_operation", "user_facing"]  # Always trace critical ops
                )

                if should_trace and self.service.is_available:
                    with self.service.trace_operation(
                        operation_name,
                        input_data={"args": args, "kwargs": kwargs},
                        metadata={
                            "agent_id": self.agent_id,
                            "operation_count": self.operation_count,
                            "sampled": should_trace
                        }
                    ) as trace:
                        try:
                            result = func(*args, **kwargs)

                            if trace:
                                trace.add_output_data({
                                    "success": True,
                                    "result_summary": str(result)[:100]  # Limit size
                                })

                            return result

                        except Exception as e:
                            if trace:
                                trace.add_error_data({
                                    "error": str(e),
                                    "error_type": type(e).__name__,
                                    "operation_count": self.operation_count
                                })
                            raise
                else:
                    # Execute without tracing
                    return func(*args, **kwargs)

            return wrapper
        return decorator

    @trace_operation("analyze_user_behavior")
    def analyze_user_behavior(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user behavior patterns."""
        # Your analysis logic here
        return {"behavior_profile": "premium_shopper", "confidence": 0.85}

    @trace_operation("generate_recommendations")
    def generate_recommendations(self, user_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations."""
        # Your recommendation logic here
        return [
            {"product_id": "PROD_001", "score": 0.95, "reason": "High engagement"},
            {"product_id": "PROD_002", "score": 0.87, "reason": "Similar preferences"}
        ]

    @trace_operation("send_notifications")
    def send_notifications(self, recommendations: List[Dict[str, Any]]) -> bool:
        """Send recommendations to user."""
        # Your notification logic here
        return True

    def full_recommendation_workflow(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete workflow with comprehensive tracing."""
        service = get_langfuse_integration()

        # Start workflow trace
        workflow_trace_id = service.create_simulation_trace({
            "type": "recommendation_workflow",
            "user_id": user_data.get("user_id"),  # Will be masked
            "agent_id": self.agent_id
        })

        try:
            # Step 1: Analyze behavior
            behavior_span = service.start_agent_span(
                agent_id=self.agent_id,
                operation="analyze_behavior",
                parent_trace_id=workflow_trace_id,
                input_data={"user_data_size": len(str(user_data))}
            )

            user_profile = self.analyze_user_behavior(user_data)
            service.end_agent_span(behavior_span, {"profile": user_profile})

            # Step 2: Generate recommendations
            rec_span = service.start_agent_span(
                agent_id=self.agent_id,
                operation="generate_recommendations",
                parent_trace_id=workflow_trace_id,
                input_data={"profile": user_profile}
            )

            recommendations = self.generate_recommendations(user_profile)
            service.end_agent_span(rec_span, {
                "recommendation_count": len(recommendations),
                "top_score": max(r["score"] for r in recommendations) if recommendations else 0
            })

            # Step 3: Send notifications
            notify_span = service.start_agent_span(
                agent_id=self.agent_id,
                operation="send_notifications",
                parent_trace_id=workflow_trace_id,
                input_data={"recommendation_count": len(recommendations)}
            )

            success = self.send_notifications(recommendations)
            service.end_agent_span(notify_span, {"notification_success": success})

            # Finalize workflow
            service.finalize_trace(workflow_trace_id, {
                "status": "completed",
                "steps_completed": 3,
                "recommendations_sent": len(recommendations),
                "success": success
            })

            return {
                "recommendations": recommendations,
                "success": success,
                "workflow_id": workflow_trace_id
            }

        except Exception as e:
            # Log error and finalize with failure status
            service.log_error(workflow_trace_id, e, {
                "operation": "full_recommendation_workflow",
                "user_id": user_data.get("user_id")
            })

            service.finalize_trace(workflow_trace_id, {
                "status": "failed",
                "error": str(e),
                "failed_step": "unknown"
            })

            raise
```

This developer guide provides a comprehensive foundation for extending the Langfuse tracing integration to new agents and components. The patterns and examples shown here can be adapted to fit your specific use cases while maintaining consistency with the existing tracing infrastructure.