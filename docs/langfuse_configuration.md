# Langfuse Configuration Guide

## Overview

This guide provides detailed information on configuring Langfuse observability integration for the AutoOps Retail Optimization System.

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LANGFUSE_PUBLIC_KEY` | Your Langfuse public key | `pk-lf-...` |
| `LANGFUSE_SECRET_KEY` | Your Langfuse secret key | `sk-lf-...` |

### Optional Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `LANGFUSE_HOST` | Langfuse host URL | `https://cloud.langfuse.com` | `https://your-instance.langfuse.com` |
| `LANGFUSE_ENABLED` | Enable/disable tracing | `true` | `false` |
| `LANGFUSE_SAMPLE_RATE` | Sampling rate (0.0-1.0) | `1.0` | `0.1` |
| `LANGFUSE_DEBUG` | Enable debug mode | `false` | `true` |
| `LANGFUSE_FLUSH_INTERVAL` | Flush interval in seconds | `5.0` | `10.0` |
| `LANGFUSE_FLUSH_AT` | Events to trigger flush | `15` | `10` |
| `LANGFUSE_TIMEOUT` | HTTP timeout in seconds | `60` | `30` |
| `LANGFUSE_MAX_RETRIES` | Maximum retry attempts | `3` | `5` |
| `LANGFUSE_MAX_LATENCY_MS` | Max acceptable latency | `100` | `50` |
| `LANGFUSE_ENABLE_SAMPLING` | Enable intelligent sampling | `true` | `false` |
| `LANGFUSE_BUFFER_SIZE` | Buffer size for traces | `1000` | `500` |
| `LANGFUSE_ENVIRONMENT` | Environment name | `development` | `production` |
| `LANGFUSE_RELEASE` | Release version | `0.1.0` | `1.0.0` |
| `LANGFUSE_TRACING_ENABLED` | Enable OpenTelemetry tracing | `true` | `false` |
| `LANGFUSE_MEDIA_UPLOAD_THREAD_COUNT` | Media upload threads | `4` | `2` |

### Advanced Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LANGFUSE_BLOCKED_INSTRUMENTATION_SCOPES` | Comma-separated blocked scopes | `scope1,scope2` |
| `LANGFUSE_ADDITIONAL_HEADERS` | Additional HTTP headers (JSON) | `{"X-Custom": "value"}` |

## Setup Instructions

### 1. Get Langfuse Credentials

1. Sign up at [Langfuse](https://langfuse.com)
2. Create a new project
3. Copy your public and secret keys from the project settings

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and set your Langfuse credentials:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-your_public_key_here
LANGFUSE_SECRET_KEY=sk-lf-your_secret_key_here
LANGFUSE_ENVIRONMENT=production
LANGFUSE_RELEASE=1.0.0
```

### 3. Validate Configuration

Run the deployment script to validate your setup:

```bash
python scripts/deploy_langfuse.py --validate-only
```

### 4. Deploy Integration

Deploy the Langfuse integration:

```bash
python scripts/deploy_langfuse.py --setup-langfuse
```

## Deployment Scripts

### Validation Only

```bash
python scripts/deploy_langfuse.py --validate-only
```

### Full Setup

```bash
python scripts/deploy_langfuse.py --setup-langfuse
```

### Create Environment File

```bash
python scripts/deploy_langfuse.py --create-env
```

### Health Check

```bash
python scripts/deploy_langfuse.py --health-check
```

## Troubleshooting

### Common Issues

1. **Missing Credentials**: Ensure `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
2. **Connection Errors**: Check `LANGFUSE_HOST` and network connectivity
3. **Permission Errors**: Verify your Langfuse keys have the correct permissions
4. **High Latency**: Adjust `LANGFUSE_MAX_LATENCY_MS` or enable sampling

### Debug Mode

Enable debug mode for detailed logging:

```bash
LANGFUSE_DEBUG=true
```

### Health Check

Perform a health check:

```bash
python scripts/deploy_langfuse.py --health-check
```

## Security Considerations

1. **Credential Storage**: Store keys in environment variables, not in code
2. **Access Control**: Configure appropriate permissions in Langfuse
3. **Data Privacy**: Review what data is being traced and consider sampling
4. **Network Security**: Use HTTPS for all Langfuse communications

## Performance Tuning

### Sampling

Reduce tracing overhead with sampling:

```bash
LANGFUSE_SAMPLE_RATE=0.1  # Trace 10% of events
LANGFUSE_ENABLE_SAMPLING=true
```

### Buffer Management

Adjust buffer settings for your workload:

```bash
LANGFUSE_BUFFER_SIZE=500
LANGFUSE_FLUSH_AT=10
LANGFUSE_FLUSH_INTERVAL=10.0
```

### Latency Control

Set maximum acceptable latency:

```bash
LANGFUSE_MAX_LATENCY_MS=50
```

## Monitoring

Monitor the integration using the built-in health check:

```bash
python scripts/deploy_langfuse.py --health-check
```

Check the application logs for Langfuse-related messages.

## Support

For issues with Langfuse integration:

1. Check the [Langfuse Documentation](https://langfuse.com/docs)
2. Review application logs for error messages
3. Run the health check script
4. Contact the development team