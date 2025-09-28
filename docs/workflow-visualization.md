# Workflow Visualization Guide

This guide explains how to interpret and analyze the workflow visualizations created by the Langfuse integration in the AutoOps retail optimization system.

## Understanding Trace Visualizations

### Trace Structure

Each workflow in the system generates a hierarchical trace structure:

```
Simulation Event (Root Trace)
├── Inventory Agent Operations
│   ├── Demand Forecasting
│   ├── Stock Level Analysis
│   └── Restock Recommendations
├── Pricing Agent Operations
│   ├── Market Analysis
│   ├── Price Optimization
│   └── Competitor Analysis
└── Promotion Agent Operations
    ├── Campaign Planning
    ├── Target Analysis
    └── Promotion Execution
```

### Trace Components

#### 1. Root Traces (Simulation Events)
- **Purpose**: Capture the initial trigger that starts a workflow
- **Data**: Event type, source, magnitude, affected products/locations
- **Duration**: Total workflow execution time
- **Status**: Success/failure of entire workflow

#### 2. Agent Spans
- **Purpose**: Track individual agent operations within a workflow
- **Data**: Agent ID, operation type, input/output data
- **Duration**: Time spent on specific operation
- **Dependencies**: Shows which operations depend on others

#### 3. Collaboration Spans
- **Purpose**: Track cross-agent communication and coordination
- **Data**: Participating agents, message flow, conflict resolution
- **Duration**: Time spent on coordination activities

## Dashboard Views

### 1. Timeline View

The timeline view shows the chronological sequence of operations:

```
Time → [Simulation Event] → [Inventory Agent] → [Pricing Agent] → [Promotion Agent]
       |                     |                   |                  |
       |                     |                   |                  |
       └─ Demand Spike       └─ Forecast         └─ Optimize        └─ Campaign
          Detected              Generated          Prices            Created
```

**Key Insights:**
- **Bottlenecks**: Long gaps between operations indicate waiting or blocking
- **Parallel Processing**: Overlapping operations show concurrent execution
- **Error Patterns**: Failed operations appear as red segments

### 2. Dependency Graph

Shows the flow of data and control between operations:

```
Simulation Event
    ↓
Inventory Agent (Demand Forecast)
    ↓
├── Pricing Agent (Price Optimization)
└── Promotion Agent (Campaign Planning)
    ↓
Collaboration (Joint Decision)
```

**Key Insights:**
- **Critical Path**: Longest chain of dependent operations
- **Parallel Opportunities**: Operations that can run concurrently
- **Failure Impact**: Which failures affect downstream operations

### 3. Performance Heatmap

Color-coded visualization of operation performance:

- **Green**: Fast operations (< 100ms)
- **Yellow**: Moderate operations (100ms - 1s)
- **Red**: Slow operations (> 1s)
- **Gray**: Failed operations

### 4. Agent Performance Dashboard

Individual agent performance metrics:

```
Inventory Agent:
├── Success Rate: 95.2%
├── Avg Response Time: 245ms
├── Operations/Hour: 1,247
└── Error Breakdown:
    ├── Data Issues: 2.1%
    ├── API Timeouts: 1.8%
    └── Logic Errors: 0.9%

Pricing Agent:
├── Success Rate: 97.8%
├── Avg Response Time: 189ms
├── Operations/Hour: 892
└── Error Breakdown:
    ├── Market Data: 1.2%
    └── Calculation: 1.0%
```

## Analyzing Workflow Patterns

### 1. Normal Workflow Pattern

**Characteristics:**
- Sequential execution: Simulation → Inventory → Pricing → Promotion
- Consistent timing between operations
- High success rates across all agents
- Minimal collaboration overhead

**Expected Duration:** 2-5 seconds

### 2. Bottleneck Pattern

**Characteristics:**
- One agent consistently slower than others
- Queue buildup before slow operation
- Downstream operations waiting
- Overall workflow duration extended

**Common Causes:**
- Resource contention
- Complex calculations
- External API dependencies
- Insufficient agent capacity

### 3. Error Cascade Pattern

**Characteristics:**
- Single failure triggers multiple downstream failures
- Error propagation through dependency chain
- Recovery attempts visible in traces
- Partial success in some branches

**Common Causes:**
- Data dependency failures
- Network issues
- Configuration problems
- Resource exhaustion

### 4. High Collaboration Pattern

**Characteristics:**
- Multiple collaboration spans
- Complex dependency graphs
- Extended coordination periods
- Cross-agent conflict resolution

**Common Causes:**
- Conflicting recommendations
- Resource contention
- Complex business rules
- Multi-stakeholder decisions

## Performance Metrics

### Key Performance Indicators (KPIs)

#### System-Level KPIs
- **Throughput**: Events processed per minute
- **Latency**: Average workflow completion time
- **Success Rate**: Percentage of successful workflows
- **Error Rate**: Percentage of failed operations

#### Agent-Level KPIs
- **Response Time**: Average time per operation
- **Success Rate**: Operations completed successfully
- **Utilization**: Percentage of time actively processing
- **Queue Length**: Pending operations waiting

#### Collaboration KPIs
- **Coordination Efficiency**: Time spent on coordination vs. execution
- **Conflict Rate**: Frequency of conflicting recommendations
- **Resolution Time**: Time to resolve conflicts
- **Agreement Rate**: Percentage of collaborative decisions

### Benchmarking

Compare performance against baselines:

```python
# Example performance benchmarks
benchmarks = {
    "inventory_agent": {
        "avg_response_time_ms": 200,
        "success_rate": 0.95,
        "throughput_per_hour": 1000
    },
    "pricing_agent": {
        "avg_response_time_ms": 150,
        "success_rate": 0.98,
        "throughput_per_hour": 800
    },
    "promotion_agent": {
        "avg_response_time_ms": 300,
        "success_rate": 0.92,
        "throughput_per_hour": 600
    }
}
```

## Troubleshooting Common Patterns

### 1. Slow Performance

**Symptoms:**
- Extended trace durations
- Yellow/red segments in heatmap
- Bottlenecks in dependency graph

**Investigation Steps:**
1. Identify the slowest operation in the trace
2. Check resource utilization during the operation
3. Review external dependencies (APIs, databases)
4. Analyze input data complexity

**Solutions:**
- Optimize slow algorithms
- Add caching for expensive operations
- Scale agent capacity
- Implement async processing

### 2. High Error Rates

**Symptoms:**
- Red segments in traces
- Failed operations in dependency chains
- Error events in logs

**Investigation Steps:**
1. Examine error details in trace metadata
2. Check input data quality
3. Review external service health
4. Analyze error patterns by operation type

**Solutions:**
- Improve error handling and recovery
- Add input validation
- Implement retry logic with exponential backoff
- Monitor external dependencies

### 3. Excessive Collaboration

**Symptoms:**
- Complex dependency graphs
- Multiple collaboration spans
- Extended coordination periods

**Investigation Steps:**
1. Review business rules for conflicts
2. Analyze decision criteria
3. Check agent recommendation consistency
4. Evaluate coordination overhead

**Solutions:**
- Refine decision criteria
- Improve agent alignment
- Optimize coordination workflows
- Add conflict prediction

### 4. Resource Contention

**Symptoms:**
- Intermittent slow performance
- Queue buildup
- Resource exhaustion events

**Investigation Steps:**
1. Monitor system resource usage
2. Check concurrent workflow volume
3. Analyze resource allocation
4. Review scaling configuration

**Solutions:**
- Implement auto-scaling
- Add resource pooling
- Optimize concurrent processing
- Load balancing improvements

## Advanced Analysis Techniques

### 1. Comparative Analysis

Compare similar workflows to identify patterns:

```python
# Compare successful vs failed workflows
successful_traces = get_traces_by_status("success")
failed_traces = get_traces_by_status("failed")

# Analyze differences in:
# - Operation durations
# - Resource usage
# - Error patterns
# - Collaboration frequency
```

### 2. Trend Analysis

Track performance over time:

```python
# Analyze performance trends
daily_metrics = get_daily_metrics()
weekly_trends = analyze_trends(daily_metrics)

# Identify:
# - Performance degradation
# - Seasonal patterns
# - Improvement opportunities
# - Anomaly detection
```

### 3. Root Cause Analysis

For failed workflows, trace back through the dependency chain:

```python
# Analyze failure propagation
failed_trace = get_trace(failed_trace_id)

# Walk backward through dependencies
for operation in reversed(failed_trace.operations):
    if operation.failed:
        # Analyze failure cause
        # Check dependencies
        # Review input data
        # Examine error context
```

## Custom Visualizations

### Creating Custom Dashboards

```python
from config.langfuse_dashboard_config import DashboardConfig

# Create custom dashboard for retail optimization
dashboard = DashboardConfig(
    name="Retail Optimization Performance",
    filters={
        "time_range": "last_7_days",
        "agent_types": ["inventory", "pricing", "promotion"],
        "operation_types": ["forecast", "optimize", "plan"]
    },
    widgets=[
        {
            "type": "timeline",
            "title": "Workflow Timeline",
            "span": "full_width"
        },
        {
            "type": "heatmap",
            "title": "Performance Heatmap",
            "span": "half_width"
        },
        {
            "type": "metrics",
            "title": "Key Metrics",
            "metrics": ["success_rate", "avg_duration", "throughput"]
        }
    ]
)
```

### Exporting Data for Analysis

```python
# Export trace data for external analysis
from config.langfuse_integration import get_langfuse_integration

service = get_langfuse_integration()

# Get all traces for analysis
all_traces = service.export_metrics_for_dashboard()

# Export to various formats
export_to_csv(all_traces, "workflow_analysis.csv")
export_to_json(all_traces, "workflow_analysis.json")
```

## Best Practices for Visualization

### 1. Dashboard Design
- **Focus on Key Metrics**: Show 3-5 most important KPIs
- **Use Color Consistently**: Green=good, Red=bad, Yellow=warning
- **Provide Context**: Include benchmarks and targets
- **Enable Drill-Down**: Allow users to explore details

### 2. Performance Monitoring
- **Set Baselines**: Establish normal performance ranges
- **Monitor Trends**: Track changes over time
- **Alert on Anomalies**: Notify when metrics exceed thresholds
- **Regular Reviews**: Weekly/monthly performance reviews

### 3. User Training
- **Educate Users**: Train on interpretation of visualizations
- **Provide Context**: Explain what metrics mean for business
- **Share Insights**: Regular reports on findings and improvements
- **Gather Feedback**: Continuously improve dashboard usability

## Integration with Business Intelligence

### Connecting to BI Tools

```python
# Export data for BI integration
from config.langfuse_dashboard_api import export_for_bi

# Export to various BI formats
export_for_tableau(all_traces, "tableau_export.hyper")
export_for_powerbi(all_traces, "powerbi_export.pbix")
export_for_excel(all_traces, "excel_export.xlsx")
```

### Scheduled Reports

```python
# Set up automated reporting
from config.langfuse_dashboard_realtime import schedule_reports

# Daily performance summary
schedule_reports(
    report_type="daily_summary",
    recipients=["ops-team@company.com"],
    schedule="daily_at_8am"
)

# Weekly trend analysis
schedule_reports(
    report_type="weekly_trends",
    recipients=["management@company.com"],
    schedule="monday_at_9am"
)
```

This visualization guide provides the foundation for understanding and optimizing your retail optimization workflows through comprehensive trace analysis and performance monitoring.