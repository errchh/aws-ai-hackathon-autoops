# Langfuse Integration Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for the Langfuse observability integration in the AutoOps Retail Optimization System. It covers common issues, diagnostic steps, and resolution procedures.

## Quick Diagnostic Commands

### 1. Health Check
```bash
python scripts/deploy_langfuse.py --health-check
```

### 2. Configuration Validation
```bash
python scripts/deploy_langfuse.py --validate-only
```

### 3. View Logs
```bash
# Application logs
tail -f logs/app.log | grep -i langfuse

# Langfuse-specific logs
tail -f logs/langfuse.log
```

### 4. Test Connection
```python
from config.langfuse_integration import get_langfuse_integration

service = get_langfuse_integration()
health = service.health_check()
print("Health Status:", health)
```

## Common Issues and Solutions

### Issue 1: "Langfuse credentials not found"

**Symptoms**:
- Application fails to start
- Error: `LangfuseConfigError: Missing required credentials`
- Health check shows `credentials_missing: true`

**Root Causes**:
1. Missing environment variables
2. Incorrect environment variable names
3. `.env` file not loaded properly

**Resolution Steps**:

1. **Check Environment Variables**:
```bash
echo $LANGFUSE_PUBLIC_KEY
echo $LANGFUSE_SECRET_KEY
echo $LANGFUSE_HOST
```

2. **Verify .env File**:
```bash
cat .env | grep -E "(LANGFUSE_PUBLIC_KEY|LANGFUSE_SECRET_KEY|LANGFUSE_HOST)"
```

3. **Fix Environment Variables**:
```bash
# Add to .env file
echo "LANGFUSE_PUBLIC_KEY=pk-lf-your_key_here" >> .env
echo "LANGFUSE_SECRET_KEY=sk-lf-your_key_here" >> .env
echo "LANGFUSE_HOST=https://cloud.langfuse.com" >> .env
```

4. **Restart Application**:
```bash
# Reload environment
source .env
python main.py
```

### Issue 2: "Connection test failed"

**Symptoms**:
- Health check shows `connection_status: failed`
- Error: `ConnectionError: Failed to connect to Langfuse`
- Traces not appearing in dashboard

**Root Causes**:
1. Network connectivity issues
2. Incorrect host URL
3. Firewall blocking connections
4. SSL/TLS certificate issues

**Resolution Steps**:

1. **Test Network Connectivity**:
```bash
# Test basic connectivity
curl -I https://cloud.langfuse.com

# Test with custom host if using self-hosted
curl -I https://your-langfuse-instance.com
```

2. **Check Host Configuration**:
```bash
# Verify LANGFUSE_HOST is correct
echo $LANGFUSE_HOST

# Test with different hosts
export LANGFUSE_HOST=https://us.cloud.langfuse.com
python scripts/deploy_langfuse.py --health-check
```

3. **Check Firewall Settings**:
```bash
# Test if port 443 is open
telnet cloud.langfuse.com 443

# Check for proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY
```

4. **SSL/TLS Issues**:
```bash
# Test SSL connection
openssl s_client -connect cloud.langfuse.com:443 -servername cloud.langfuse.com

# Disable SSL verification (temporary debugging)
export LANGFUSE_SSL_VERIFY=false
```

### Issue 3: "High latency impact"

**Symptoms**:
- Agent response times increased significantly
- Health check shows `avg_latency_ms: > 100`
- System performance degradation

**Root Causes**:
1. Excessive tracing overhead
2. Network latency to Langfuse
3. Insufficient sampling
4. Synchronous trace processing

**Resolution Steps**:

1. **Enable Sampling**:
```bash
# Reduce sampling rate
export LANGFUSE_SAMPLE_RATE=0.1  # Trace only 10% of events
export LANGFUSE_ENABLE_SAMPLING=true
```

2. **Adjust Buffer Settings**:
```bash
# Increase buffer size and flush interval
export LANGFUSE_BUFFER_SIZE=2000
export LANGFUSE_FLUSH_INTERVAL=10.0
export LANGFUSE_FLUSH_AT=20
```

3. **Enable Async Processing**:
```python
# Ensure async processing is enabled in config
service = get_langfuse_integration()
# Traces are processed asynchronously by default
```

4. **Monitor Performance**:
```python
# Check performance metrics
metrics = service.get_performance_metrics()
print(f"Success Rate: {metrics['success_rate']}")
print(f"Avg Latency: {metrics['avg_latency_ms']}ms")
print(f"Throughput: {metrics['throughput']} traces/sec")
```

### Issue 4: "Traces not appearing in dashboard"

**Symptoms**:
- Traces are created but not visible in Langfuse dashboard
- Health check shows traces are being created
- No errors in application logs

**Root Causes**:
1. Incorrect project configuration in Langfuse
2. Trace data not being flushed
3. Sampling excluding traces
4. Time synchronization issues

**Resolution Steps**:

1. **Check Project Configuration**:
   - Verify you're looking at the correct project in Langfuse dashboard
   - Check if traces are in the correct environment
   - Verify release version matches

2. **Force Flush Traces**:
```python
# Manually flush pending traces
service = get_langfuse_integration()
service.flush()  # Force immediate flush

# Check flush status
print(f"Pending traces: {service.get_pending_trace_count()}")
```

3. **Verify Sampling**:
```bash
# Check current sampling rate
echo $LANGFUSE_SAMPLE_RATE

# Temporarily disable sampling for testing
export LANGFUSE_SAMPLE_RATE=1.0
```

4. **Check Time Settings**:
```python
# Verify system time is correct
date

# Check if Langfuse dashboard timezone matches
# Dashboard shows traces in UTC by default
```

### Issue 5: "Memory usage increasing"

**Symptoms**:
- Application memory usage growing over time
- Health check shows high buffer usage
- OutOfMemory errors

**Root Causes**:
1. Trace buffer not being flushed
2. Memory leaks in trace data
3. Excessive trace volume
4. Large trace payloads

**Resolution Steps**:

1. **Check Buffer Status**:
```python
service = get_langfuse_integration()
buffer_stats = service.get_buffer_stats()
print(f"Buffer size: {buffer_stats['current_size']}")
print(f"Max buffer size: {buffer_stats['max_size']}")
print(f"Flush pending: {buffer_stats['flush_pending']}")
```

2. **Force Buffer Flush**:
```bash
# Reduce buffer size and increase flush frequency
export LANGFUSE_BUFFER_SIZE=500
export LANGFUSE_FLUSH_INTERVAL=2.0
export LANGFUSE_FLUSH_AT=10

# Restart application to apply changes
```

3. **Monitor Memory Usage**:
```bash
# Monitor Python process memory
ps aux | grep python

# Check garbage collection
import gc
print(f"Objects collected: {gc.collect()}")
```

4. **Reduce Trace Volume**:
```bash
# Enable aggressive sampling
export LANGFUSE_SAMPLE_RATE=0.05  # 5% sampling
export LANGFUSE_ENABLE_SAMPLING=true
```

### Issue 6: "Authentication errors"

**Symptoms**:
- Error: `AuthenticationError: Invalid credentials`
- Health check shows `auth_failed: true`
- 401/403 HTTP status codes

**Root Causes**:
1. Invalid API keys
2. Expired credentials
3. Incorrect key format
4. Permission issues

**Resolution Steps**:

1. **Verify API Keys**:
   - Check if keys are copied correctly (no extra spaces)
   - Verify keys haven't expired
   - Regenerate keys if necessary

2. **Test Key Format**:
```bash
# Keys should start with pk-lf- and sk-lf-
echo $LANGFUSE_PUBLIC_KEY | grep "^pk-lf-"
echo $LANGFUSE_SECRET_KEY | grep "^sk-lf-"
```

3. **Check Permissions**:
   - Verify keys have write permissions for traces
   - Check if project allows the IP address
   - Verify project isn't in read-only mode

4. **Regenerate Keys**:
```bash
# In Langfuse dashboard:
# 1. Go to Project Settings
# 2. Generate new API keys
# 3. Update .env file with new keys
# 4. Restart application
```

### Issue 7: "Configuration validation errors"

**Symptoms**:
- Error: `LangfuseConfigError: Invalid configuration`
- Application fails during startup
- Configuration validation fails

**Root Causes**:
1. Invalid configuration values
2. Missing required fields
3. Incorrect data types
4. Malformed environment variables

**Resolution Steps**:

1. **Validate Configuration**:
```bash
python scripts/deploy_langfuse.py --validate-only
```

2. **Check Configuration Values**:
```python
from config.langfuse_config import LangfuseConfig

# Test configuration creation
try:
    config = LangfuseConfig.from_env()
    print("Configuration valid:", config)
except Exception as e:
    print("Configuration error:", e)
```

3. **Fix Common Issues**:
```bash
# Ensure numeric values are numbers, not strings
export LANGFUSE_SAMPLE_RATE=0.1  # not "0.1"
export LANGFUSE_FLUSH_INTERVAL=5.0  # not "5.0"
export LANGFUSE_MAX_RETRIES=3  # not "3"
```

4. **Reset to Defaults**:
```bash
# Use default configuration
export LANGFUSE_ENABLED=false
export LANGFUSE_DEBUG=true
```

## Advanced Troubleshooting

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
export LANGFUSE_DEBUG=true
export PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from config.langfuse_integration import get_langfuse_integration
service = get_langfuse_integration()
print('Debug mode enabled')
"
```

### Network Packet Capture

Capture network traffic to diagnose connection issues:

```bash
# Install tcpdump if not available
sudo apt-get install tcpdump

# Capture Langfuse traffic
sudo tcpdump -i any -A -s 0 host cloud.langfuse.com and port 443
```

### Database Connection Issues

If using self-hosted Langfuse, check database connectivity:

```bash
# Test database connection
psql -h your-db-host -U langfuse_user -d langfuse_db

# Check database logs
tail -f /var/log/postgresql/postgresql.log | grep langfuse
```

### Performance Profiling

Profile the application to identify bottlenecks:

```python
import cProfile
import pstats

# Profile trace creation
profiler = cProfile.Profile()
profiler.enable()

# Your trace creation code here
service.create_simulation_trace({"test": "data"})

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

## Error Code Reference

### Common Error Codes

| Error Code | Description | Resolution |
|------------|-------------|------------|
| `CONFIG_ERROR` | Configuration validation failed | Check environment variables |
| `CONNECTION_ERROR` | Cannot connect to Langfuse | Check network and credentials |
| `AUTH_ERROR` | Authentication failed | Verify API keys |
| `TIMEOUT_ERROR` | Request timeout | Check network latency |
| `RATE_LIMIT_ERROR` | Too many requests | Enable sampling |
| `VALIDATION_ERROR` | Invalid trace data | Check trace format |
| `FLUSH_ERROR` | Failed to flush traces | Check buffer settings |

### HTTP Status Codes

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 400 | Bad Request | Check trace data format |
| 401 | Unauthorized | Verify API keys |
| 403 | Forbidden | Check project permissions |
| 404 | Not Found | Verify endpoint URL |
| 429 | Rate Limited | Enable sampling |
| 500 | Server Error | Check Langfuse service status |
| 503 | Service Unavailable | Retry later |

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Connection Success Rate**: Should be >99%
2. **Average Latency**: Should be <100ms
3. **Trace Success Rate**: Should be >95%
4. **Buffer Usage**: Should be <80% of max
5. **Error Rate**: Should be <1%

### Alert Thresholds

```python
ALERT_THRESHOLDS = {
    "connection_success_rate": 0.99,  # Alert if <99%
    "avg_latency_ms": 100,           # Alert if >100ms
    "trace_success_rate": 0.95,      # Alert if <95%
    "buffer_usage_percent": 0.8,     # Alert if >80%
    "error_rate": 0.01               # Alert if >1%
}
```

### Automated Health Checks

Set up automated monitoring:

```bash
# Add to cron for regular health checks
*/5 * * * * /path/to/project/scripts/deploy_langfuse.py --health-check >> /var/log/langfuse_health.log 2>&1
```

## Getting Help

### Support Channels

1. **Documentation**: Check this troubleshooting guide
2. **Logs**: Review application and Langfuse logs
3. **Community**: Langfuse community forums
4. **Support**: Contact Langfuse support for account issues
5. **Development Team**: Contact the AutoOps development team

### Reporting Issues

When reporting issues, include:

1. **Error messages** (full stack traces)
2. **Configuration** (sanitized environment variables)
3. **Steps to reproduce**
4. **System information** (OS, Python version, Langfuse version)
5. **Recent changes** that might have caused the issue

### Log Collection

Collect comprehensive logs for debugging:

```bash
# Collect all relevant logs
mkdir -p /tmp/langfuse_debug
cp logs/app.log /tmp/langfuse_debug/
cp logs/langfuse.log /tmp/langfuse_debug/
env > /tmp/langfuse_debug/environment.txt
python scripts/deploy_langfuse.py --health-check --verbose > /tmp/langfuse_debug/health_check.txt 2>&1
```

## Prevention

### Best Practices

1. **Environment Separation**: Use different projects for dev/staging/prod
2. **Regular Health Checks**: Monitor integration health continuously
3. **Sampling Strategy**: Implement appropriate sampling for production
4. **Error Handling**: Ensure graceful degradation when Langfuse is unavailable
5. **Configuration Management**: Use version control for configuration
6. **Documentation**: Keep configuration and troubleshooting docs updated

### Maintenance Tasks

1. **Weekly**: Review error logs and performance metrics
2. **Monthly**: Validate configuration and test failover scenarios
3. **Quarterly**: Review and update sampling strategies
4. **Annually**: Audit data retention and security settings

## Recovery Procedures

### Disaster Recovery

If Langfuse becomes completely unavailable:

1. **Enable Fallback Mode**:
```bash
export LANGFUSE_ENABLED=false
```

2. **Buffer Local Traces**:
```python
# Traces will be buffered locally
service = get_langfuse_integration()
service.enable_local_buffering()
```

3. **Manual Flush When Service Restored**:
```python
# When Langfuse is back online
service.flush_buffered_traces()
```

### Data Recovery

If trace data is lost:

1. **Check Local Buffers**: Look for buffered trace files
2. **Application Logs**: Extract trace information from logs
3. **Database Backup**: Restore from application database if available
4. **Re-run Simulations**: Re-execute key workflows to regenerate traces

## Testing Troubleshooting

### Test Scenarios

1. **Network Failure**: Simulate network outages
2. **Invalid Credentials**: Test with wrong API keys
3. **High Load**: Test with excessive trace volume
4. **Service Unavailable**: Test Langfuse service downtime
5. **Configuration Errors**: Test with invalid configuration

### Automated Tests

```python
def test_langfuse_failure_scenarios():
    """Test various failure scenarios."""
    service = get_langfuse_integration()

    # Test 1: Invalid credentials
    with mock_invalid_credentials():
        health = service.health_check()
        assert health["fallback_mode"] == True

    # Test 2: Network failure
    with mock_network_failure():
        trace_id = service.create_simulation_trace({"test": "data"})
        assert trace_id is not None  # Should work in fallback mode

    # Test 3: Service overload
    with mock_high_load():
        service.log_agent_decision("test_agent", {"test": "data"})
        # Should handle gracefully with sampling
```

This comprehensive troubleshooting guide should help resolve most Langfuse integration issues. If problems persist, don't hesitate to reach out to the development team or Langfuse support.