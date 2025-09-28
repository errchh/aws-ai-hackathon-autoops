# Security and Data Masking Features

This document describes the security and data masking features implemented for the Langfuse workflow visualization system.

## Overview

The security features ensure that sensitive information is properly masked in traces while maintaining the functionality of the observability system. This includes PII filtering, data retention policies, access controls, and secure credential management.

## Data Masking

### Masking Strategies

The system supports several masking strategies:

- **REPLACE**: Replace sensitive data with a fixed string (e.g., `********` for passwords)
- **PARTIAL**: Show partial information while masking sensitive parts (e.g., `test@***.com` for emails)
- **REDACT**: Completely remove sensitive data
- **HASH**: Hash sensitive data for anonymization

### Default Masking Rules

The system includes default rules for common sensitive data types:

| Field Path | Strategy | Description |
|------------|----------|-------------|
| `user.email` | PARTIAL | Email addresses |
| `user.phone` | PARTIAL | Phone numbers |
| `user.ssn` | REPLACE | Social Security Numbers |
| `payment.card_number` | PARTIAL | Credit card numbers |
| `payment.cvv` | REPLACE | CVV codes |
| `auth.password` | REPLACE | Passwords |
| `auth.api_key` | PARTIAL | API keys |
| `location.address` | PARTIAL | Street addresses |
| `business.profit_margin` | REPLACE | Profit margins |

### Custom Masking Rules

You can add custom masking rules by extending the `MaskingRule` class:

```python
from config.langfuse_data_masking import MaskingRule, MaskingStrategy

custom_rule = MaskingRule(
    field_path="custom.sensitive_field",
    strategy=MaskingStrategy.REPLACE,
    replacement="[SENSITIVE]",
    description="Custom sensitive field"
)
```

## Configuration

### Environment Variables

Add the following environment variables to configure security features:

```bash
# Data Masking
LANGFUSE_ENABLE_DATA_MASKING=true
LANGFUSE_PII_FILTERING_ENABLED=true

# Data Retention
LANGFUSE_DATA_RETENTION_DAYS=90

# Access Control
LANGFUSE_ALLOWED_TRACE_TYPES=simulation,agent_decision,collaboration
LANGFUSE_AUDIT_TRACE_ACCESS=true

# Encryption
LANGFUSE_ENCRYPTION_ENABLED=false
```

### Configuration in Code

```python
from config.langfuse_config import LangfuseConfig

config = LangfuseConfig(
    # ... other config
    enable_data_masking=True,
    pii_filtering_enabled=True,
    data_retention_days=90,
    audit_trace_access=True,
    allowed_trace_types=["simulation", "agent_decision"]
)
```

## Access Controls

### Trace Type Restrictions

You can restrict which trace types are allowed based on user roles:

```python
from config.langfuse_data_masking import get_secure_trace_manager

manager = get_secure_trace_manager()
manager.set_access_control("simulation", ["admin", "analyst"])
manager.set_access_control("agent_decision", ["admin"])
```

### User Role Checking

When creating traces, provide user roles for access control:

```python
from config.langfuse_integration import get_langfuse_integration

integration = get_langfuse_integration()
trace_id = integration.create_simulation_trace(
    event_data={"type": "market_event", "data": {...}},
    user_roles=["admin"]
)
```

## Data Retention Policies

### Setting Retention Policies

Configure how long different types of data should be retained:

```python
manager = get_secure_trace_manager()
manager.set_retention_policy("user.email", 30)  # 30 days
manager.set_retention_policy("payment.card_number", 365)  # 1 year
```

### Automatic Cleanup

The system can automatically remove expired data based on retention policies. This is handled during trace creation and processing.

## Credential Security

### Secure Storage

Credentials are stored securely using environment variables:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
```

### Validation

The system validates credentials before use:

```python
from config.langfuse_data_masking import get_secure_trace_manager

manager = get_secure_trace_manager()
is_valid = manager.validate_credentials({
    "public_key": "pk-lf-...",
    "secret_key": "sk-lf-..."
})
```

## Audit Logging

### Enabling Audit Logs

Enable audit logging to track trace access:

```bash
LANGFUSE_AUDIT_TRACE_ACCESS=true
```

### Audit Events

The system logs the following audit events:

- Trace creation
- Trace access
- Data masking operations
- Access control violations

## Testing

### Running Security Tests

```bash
pytest tests/test_langfuse_security.py -v
```

### Test Coverage

The test suite covers:

- Data masking functionality
- Access control mechanisms
- Credential validation
- Retention policy enforcement
- Integration with Langfuse traces

## Best Practices

### 1. Principle of Least Privilege

- Only grant access to trace types that users actually need
- Use specific roles rather than broad permissions

### 2. Data Minimization

- Only collect and store necessary data
- Apply masking to all potentially sensitive fields
- Set appropriate retention policies

### 3. Regular Audits

- Regularly review masking rules
- Monitor audit logs for suspicious activity
- Update retention policies as needed

### 4. Secure Configuration

- Store credentials in secure environment variables
- Use strong, unique keys for production
- Regularly rotate credentials

## Troubleshooting

### Common Issues

1. **Traces not appearing**: Check access controls and user roles
2. **Data not masked**: Verify masking rules are correctly configured
3. **Performance impact**: Monitor masking overhead and adjust rules if needed

### Debug Mode

Enable debug logging to troubleshoot security features:

```python
import logging
logging.getLogger('config.langfuse_data_masking').setLevel(logging.DEBUG)
```

## Compliance

These security features help with compliance for:

- **GDPR**: Data minimization and user consent
- **PCI DSS**: Payment information protection
- **HIPAA**: Healthcare data protection (if applicable)
- **SOX**: Audit trail requirements

## Future Enhancements

- Encryption at rest for stored traces
- Advanced pattern matching for custom data types
- Integration with external secret management systems
- Real-time alerting for security violations