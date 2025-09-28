# Troubleshooting Guide

This guide provides solutions to common issues encountered when using the Langfuse integration in the AutoOps retail optimization system.

## Common Issues and Solutions

### 1. Langfuse Connection Issues

#### Problem: "Langfuse client not available" or connection failures

**Symptoms:**
- Traces not appearing in dashboard
- `is_available` returns `False`
- Connection timeout errors in logs

**Root Causes:**
- Invalid credentials
- Network connectivity issues
- Firewall blocking Langfuse endpoints
- Incorrect host configuration

**Solutions:**

1. **Verify Credentials:**
   ```bash
   # Check environment variables
   echo $LANGFUSE_PUBLIC_KEY
   echo $LANGFUSE_SECRET_KEY
   echo $LANGFUSE_HOST
   ```

2. **Test Network Connectivity:**
   ```bash
   # Test connection to Langfuse host
   curl -I https://cloud.langfuse.com
   ```

3. **Validate Configuration:**
   ```python
   from config.langfuse_config import get_langfuse_client

   client = get_langfuse_client()
   health = client.health_check()
   print(health)
   ```

4. **Check Logs:**
   ```bash
   # Look for connection errors
   grep -r "langfuse" /path/to/logs/ | grep -i "error\|fail"
   ```

#### Problem: Authentication failures

**Symptoms:**
- 401 Unauthorized errors
- Invalid credentials messages

**Solutions:**
1. **Regenerate API Keys** in Langfuse dashboard
2. **Update Environment Variables** with new keys
3. **Restart Application** to pick up new credentials
4. **Verify Key Format** - ensure keys start with `pk-lf-` and `sk-lf-`

### 2. Performance Issues

#### Problem: High tracing overhead or slow operations

**Symptoms:**
- Agent operations taking longer than expected
- High CPU/memory usage
- Traces showing extended durations

**Root Causes:**
- Excessive trace creation
- Large trace payloads
- Synchronous flushing
- Insufficient sampling

**Solutions:**

1. **Enable Sampling:**
   ```python
   # Configure sampling in environment
   export LANGFUSE_SAMPLE_RATE=0.1  # Sample 10% of traces
   ```

2. **Optimize Trace Data:**
   ```python
   # Reduce trace payload size
   service = get_langfuse_integration()
   service.configure(max_payload_size=1000)  # Limit payload size
   ```

3. **Use Async Processing:**
   ```python
   # Enable async trace processing
   service.configure(async_processing=True, flush_interval=10)
   ```

4. **Monitor Performance Impact:**
   ```python
   # Check tracing overhead
   metrics = service.get_system_metrics()
   print(f"Tracing overhead: {metrics.get('tracing_overhead', 0)}%")
   ```

#### Problem: Memory usage growing over time

**Symptoms:**
- Increasing memory consumption
- Out of memory errors
- Performance degradation over time

**Solutions:**
1. **Reduce Buffer Size:**
   ```bash
   export LANGFUSE_BUFFER_SIZE=500  # Smaller buffer
   ```

2. **Increase Flush Frequency:**
   ```bash
   export LANGFUSE_FLUSH_INTERVAL=2  # Flush every 2 seconds
   ```

3. **Monitor Memory Usage:**
   ```python
   import psutil
   import os

   process = psutil.Process(os.getpid())
   memory_mb = process.memory_info().rss / 1024 / 1024
   print(f"Memory usage: {memory_mb} MB")
   ```

### 3. Data Quality Issues

#### Problem: Missing or incomplete traces

**Symptoms:**
- Partial traces in dashboard
- Missing operations in workflows
- Inconsistent trace data

**Root Causes:**
- Agent failures during tracing
- Network interruptions
- Data masking issues
- Buffer overflow

**Solutions:**

1. **Check Agent Error Handling:**
   ```python
   # Ensure agents handle tracing errors gracefully
   try:
       result = perform_operation()
       service.log_operation_success(trace_id, result)
   except Exception as e:
       service.log_operation_error(trace_id, e)
       # Don't let tracing errors break agent logic
   ```

2. **Verify Data Masking:**
   ```python
   from config.langfuse_data_masking import get_data_masker

   masker = get_data_masker()
   test_data = {"user_id": "12345", "email": "user@example.com"}
   masked = masker.mask_data(test_data)
   print(masked)  # Should mask sensitive fields
   ```

3. **Monitor Buffer Status:**
   ```python
   service = get_langfuse_integration()
   health = service.health_check()
   print(f"Buffer usage: {health.get('buffer_usage', 0)}%")
   ```

#### Problem: Incorrect trace hierarchy or relationships

**Symptoms:**
- Orphaned spans
- Incorrect parent-child relationships
- Missing dependency links

**Solutions:**
1. **Validate Trace Context Propagation:**
   ```python
   # Ensure trace context is properly passed
   with service.trace_operation("parent_operation") as parent_trace:
       # Pass trace context to child operations
       child_result = child_operation(parent_trace.id)
   ```

2. **Check Span Timing:**
   ```python
   # Ensure spans are properly nested
   span_id = service.start_agent_span("agent", "operation", parent_trace_id)
   try:
       # Operation logic
       result = perform_operation()
   finally:
       service.end_agent_span(span_id, {"result": result})
   ```

### 4. Dashboard and Visualization Issues

#### Problem: Traces not appearing in dashboard

**Symptoms:**
- Traces created locally but not visible in Langfuse dashboard
- Delayed trace appearance
- Missing trace data

**Root Causes:**
- Flush not called
- Network connectivity issues
- Dashboard filters hiding traces
- Time zone differences

**Solutions:**

1. **Force Flush:**
   ```python
   service = get_langfuse_integration()
   service.flush()  # Manually flush pending traces
   ```

2. **Check Dashboard Filters:**
   - Verify time range includes trace timestamps
   - Check if project/environment filters are applied
   - Ensure trace types are not filtered out

3. **Verify Time Synchronization:**
   ```python
   # Check time difference
   import datetime
   local_time = datetime.datetime.now()
   print(f"Local time: {local_time}")
   # Compare with Langfuse dashboard time zone
   ```

#### Problem: Incorrect metrics or aggregations

**Symptoms:**
- Wrong success rates
- Incorrect duration calculations
- Missing performance data

**Solutions:**
1. **Validate Metrics Collection:**
   ```python
   service = get_langfuse_integration()
   metrics = service.get_system_metrics()
   print(f"Total events: {metrics['total_events_processed']}")
   ```

2. **Check Time Windows:**
   ```python
   # Ensure metrics are calculated over correct time periods
   recent_metrics = service.get_agent_metrics("inventory_agent")
   print(f"Recent performance: {recent_metrics}")
   ```

### 5. Security and Privacy Issues

#### Problem: Sensitive data appearing in traces

**Symptoms:**
- PII or confidential data in trace logs
- Security audit failures
- Data compliance violations

**Solutions:**
1. **Review Data Masking Configuration:**
   ```python
   from config.langfuse_data_masking import get_data_masker

   masker = get_data_masker()
   # Check what fields are being masked
   print("Masked fields:", masker.masked_fields)
   ```

2. **Add Custom Masking Rules:**
   ```python
   # Configure additional sensitive fields
   masker.add_sensitive_field("credit_card_number")
   masker.add_sensitive_field("ssn")
   masker.add_sensitive_field("medical_record_id")
   ```

3. **Audit Trace Content:**
   ```python
   # Review traces for sensitive data
   service = get_langfuse_integration()
   traces = service.get_recent_traces(limit=10)
   for trace in traces:
       audit_trace_content(trace)
   ```

#### Problem: Access control issues

**Symptoms:**
- Unauthorized trace access
- Missing audit logs
- Security policy violations

**Solutions:**
1. **Configure Access Controls:**
   ```python
   from config.langfuse_config import LangfuseConfig

   config = LangfuseConfig(
       audit_trace_access=True,
       require_user_roles=True,
       allowed_roles=["admin", "analyst"]
   )
   ```

2. **Review Audit Logs:**
   ```python
   # Check trace access audit trail
   audit_logs = service.get_audit_logs()
   for log in audit_logs:
       print(f"Access: {log['user']} -> {log['trace_id']} at {log['timestamp']}")
   ```

### 6. Integration Issues

#### Problem: Agent operations failing due to tracing

**Symptoms:**
- Agent logic breaking when tracing is enabled
- Import errors for tracing modules
- Dependency conflicts

**Solutions:**
1. **Implement Graceful Degradation:**
   ```python
   # Ensure agents work without tracing
   try:
       service = get_langfuse_integration()
       if service.is_available:
           with service.trace_operation("operation") as trace:
               result = perform_operation()
       else:
           result = perform_operation()
   except ImportError:
       # Fallback if tracing modules not available
       result = perform_operation()
   ```

2. **Check Dependencies:**
   ```bash
   # Verify all required packages are installed
   python -c "import langfuse; print('Langfuse OK')"
   python -c "from config.langfuse_integration import get_langfuse_integration; print('Integration OK')"
   ```

#### Problem: Configuration conflicts

**Symptoms:**
- Multiple configuration sources conflicting
- Environment variables not being read
- Configuration changes not taking effect

**Solutions:**
1. **Validate Configuration Loading:**
   ```python
   from config.settings import get_settings

   settings = get_settings()
   print(f"Langfuse enabled: {settings.langfuse_enabled}")
   print(f"Sample rate: {settings.langfuse_sample_rate}")
   ```

2. **Check Configuration Priority:**
   ```python
   # Environment variables should override defaults
   import os
   print(f"ENV sample rate: {os.getenv('LANGFUSE_SAMPLE_RATE')}")
   print(f"Settings sample rate: {settings.langfuse_sample_rate}")
   ```

### 7. Monitoring and Alerting Issues

#### Problem: Alerts not triggering

**Symptoms:**
- Performance issues not being detected
- No alerts for failures
- Monitoring dashboards showing stale data

**Solutions:**
1. **Check Alert Configuration:**
   ```python
   from config.langfuse_monitoring_alerting import get_alert_manager

   alert_manager = get_alert_manager()
   alerts = alert_manager.get_active_alerts()
   print(f"Active alerts: {len(alerts)}")
   ```

2. **Test Alert Thresholds:**
   ```python
   # Simulate threshold breach
   alert_manager.test_alert_thresholds()
   ```

3. **Verify Monitoring Data:**
   ```python
   # Check if monitoring data is being collected
   service = get_langfuse_integration()
   health = service.health_check()
   print(f"Monitoring status: {health}")
   ```

## Diagnostic Tools

### 1. Health Check Script

```python
#!/usr/bin/env python3
"""Comprehensive health check for Langfuse integration."""

from config.langfuse_integration import get_langfuse_integration
from config.langfuse_config import get_langfuse_client

def run_health_check():
    print("=== Langfuse Integration Health Check ===")

    # Check client
    client = get_langfuse_client()
    print(f"Client available: {client.is_available}")
    print(f"Client health: {client.health_check()}")

    # Check integration service
    service = get_langfuse_integration()
    print(f"Service available: {service.is_available}")
    print(f"Service health: {service.health_check()}")

    # Check metrics
    try:
        metrics = service.get_system_metrics()
        print(f"System metrics: {metrics}")
    except Exception as e:
        print(f"Metrics error: {e}")

    # Check recent traces
    try:
        traces = service.get_recent_traces(limit=5)
        print(f"Recent traces: {len(traces)}")
    except Exception as e:
        print(f"Trace retrieval error: {e}")

    print("=== Health Check Complete ===")

if __name__ == "__main__":
    run_health_check()
```

### 2. Performance Benchmark Script

```python
#!/usr/bin/env python3
"""Benchmark tracing performance impact."""

import time
import statistics
from config.langfuse_integration import get_langfuse_integration

def benchmark_tracing():
    service = get_langfuse_integration()

    # Benchmark with tracing
    times_with_tracing = []
    for i in range(100):
        start = time.time()
        with service.trace_operation(f"benchmark_{i}") as trace:
            time.sleep(0.001)  # Small operation
        duration = time.time() - start
        times_with_tracing.append(duration)

    # Benchmark without tracing
    times_without_tracing = []
    for i in range(100):
        start = time.time()
        time.sleep(0.001)  # Same operation
        duration = time.time() - start
        times_without_tracing.append(duration)

    # Calculate statistics
    tracing_overhead = (
        statistics.mean(times_with_tracing) - statistics.mean(times_without_tracing)
    ) / statistics.mean(times_without_tracing) * 100

    print(f"Tracing overhead: {tracing_overhead".2f"}%")
    print(f"With tracing: {statistics.mean(times_with_tracing)*1000".2f"}ms avg")
    print(f"Without tracing: {statistics.mean(times_without_tracing)*1000".2f"}ms avg")

if __name__ == "__main__":
    benchmark_tracing()
```

### 3. Log Analysis Script

```python
#!/usr/bin/env python3
"""Analyze Langfuse-related logs for issues."""

import re
from pathlib import Path

def analyze_logs(log_file="/var/log/app.log"):
    print(f"=== Analyzing logs: {log_file} ===")

    # Common error patterns
    error_patterns = [
        (r"langfuse.*error", "Langfuse errors"),
        (r"trace.*fail", "Trace failures"),
        (r"connection.*refused", "Connection issues"),
        (r"timeout", "Timeout errors"),
        (r"authentication.*fail", "Auth failures"),
        (r"permission.*denied", "Permission issues"),
    ]

    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()

        for pattern, description in error_patterns:
            matches = [line for line in lines if re.search(pattern, line, re.I)]
            if matches:
                print(f"{description}: {len(matches)} occurrences")
                # Show recent examples
                for line in matches[-3:]:
                    print(f"  {line.strip()}")

    except FileNotFoundError:
        print(f"Log file not found: {log_file}")

if __name__ == "__main__":
    analyze_logs()
```

## Getting Help

### 1. Check Documentation
- Review the [Integration Guide](./integration-guide.md)
- Check the [API Reference](./api-reference.md)
- Look at [Examples](./examples/)

### 2. Community Resources
- Langfuse Documentation: https://langfuse.com/docs
- GitHub Issues: https://github.com/langfuse/langfuse/issues
- Stack Overflow: Tag `langfuse`

### 3. Support Channels
- Report issues at https://github.com/sst/opencode/issues
- Contact the development team for urgent issues
- Check system status at https://status.langfuse.com

### 4. Debug Mode
Enable debug logging for detailed troubleshooting:

```python
import logging

# Enable debug logging
logging.getLogger("config.langfuse").setLevel(logging.DEBUG)

# Or set environment variable
export LANGFUSE_DEBUG=true
```

This troubleshooting guide covers the most common issues encountered with the Langfuse integration. If you encounter an issue not covered here, please report it with detailed logs and system information.