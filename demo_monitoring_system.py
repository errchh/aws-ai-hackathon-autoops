#!/usr/bin/env python3
"""Demonstration of the Langfuse monitoring and alerting system."""

import sys
import os
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🚀 Langfuse Monitoring and Alerting System Demo")
print("=" * 50)

try:
    # Import the monitoring components
    print("📦 Importing monitoring components...")

    # Import each component individually to avoid circular imports
    from config.langfuse_performance_monitor import (
        PerformanceMonitor,
        PerformanceThresholds,
    )
    from config.langfuse_alerting import (
        AlertingEngine,
        LangfuseAlertManager,
        AlertSeverity,
        AlertRule,
    )
    from config.langfuse_error_handler import LangfuseErrorHandler, DegradationLevel
    from config.metrics_collector import MetricsCollector
    from config.langfuse_integration import LangfuseIntegrationService

    print("✅ Successfully imported all monitoring components")

    # Initialize components
    print("\n🔧 Initializing monitoring components...")

    # Create mock error handler for demo
    class MockLangfuseClient:
        def __init__(self):
            self.is_available = True
            self.client = None
            self.config = None

    mock_client = MockLangfuseClient()
    error_handler = LangfuseErrorHandler(mock_client)
    print("✅ Error handler initialized")

    # Initialize metrics collector
    metrics_collector = MetricsCollector()
    print("✅ Metrics collector initialized")

    # Initialize performance monitor
    thresholds = PerformanceThresholds()
    performance_monitor = PerformanceMonitor(
        error_handler=error_handler, thresholds=thresholds
    )
    print("✅ Performance monitor initialized")

    # Initialize alerting engine
    alerting_engine = AlertingEngine()
    alert_manager = LangfuseAlertManager(alerting_engine)
    print("✅ Alerting system initialized")

    # Initialize integration service
    integration_service = LangfuseIntegrationService()
    print("✅ Integration service initialized")

    # Demonstrate monitoring functionality
    print("\n📊 Demonstrating monitoring functionality...")

    # Simulate some operations
    print("Simulating agent operations...")
    for i in range(5):
        operation_id = f"demo_op_{i}"
        metrics_collector.start_operation(operation_id, f"agent_{i}", "demo_operation")

        # Simulate some processing time
        time.sleep(0.1)

        # End operation successfully
        metrics_collector.end_operation(operation_id, success=True)

    print("✅ Simulated 5 agent operations")

    # Get metrics summary
    metrics_summary = metrics_collector.get_metrics_summary()
    print("✅ Retrieved metrics summary:")
    print(f"   - Total Operations: {metrics_summary['total_operations']}")
    print(f"   - Active Workflows: {metrics_summary['active_workflows']}")

    # Get performance summary
    perf_summary = performance_monitor.get_performance_summary()
    print("✅ Retrieved performance summary:")
    print(f"   - Current CPU Usage: {perf_summary['current']['cpu_usage']:.1f}%")
    print(f"   - Current Memory Usage: {perf_summary['current']['memory_usage']:.1f}%")

    # Get alert stats
    alert_stats = alerting_engine.get_alert_stats()
    print("✅ Retrieved alert statistics:")
    print(f"   - Total Rules: {alert_stats['total_rules']}")
    print(f"   - Enabled Rules: {alert_stats['enabled_rules']}")

    # Demonstrate health check simulation
    print("\n🏥 Simulating health check...")

    # Simulate health check results
    health_status = "healthy"
    connectivity_tests = [
        {"name": "langfuse_connectivity", "success": True, "duration_ms": 45.2},
        {"name": "dns_resolution", "success": True, "duration_ms": 12.1},
        {"name": "port_connectivity", "success": True, "duration_ms": 8.7},
    ]

    print("✅ Health check simulation completed:")
    print(f"   - Overall Status: {health_status}")
    print(
        f"   - Connectivity Tests Passed: {len([t for t in connectivity_tests if t['success']])}/{len(connectivity_tests)}"
    )

    # Demonstrate alerting
    print("\n🚨 Demonstrating alerting system...")

    # Create a test alert
    test_alert = alerting_engine._create_alert(
        AlertRule(
            name="demo_high_latency",
            description="Demo high latency alert",
            severity=AlertSeverity.WARNING,
            threshold=100.0,
            metric_name="average_response_time",
        ),
        150.0,  # Metric value above threshold
        datetime.now(),
    )

    print("✅ Created test alert:")
    print(f"   - Alert ID: {test_alert.id}")
    print(f"   - Severity: {test_alert.severity.value}")
    print(f"   - Title: {test_alert.title}")

    # Get active alerts
    active_alerts = alerting_engine.get_active_alerts()
    print(f"✅ Active alerts count: {len(active_alerts)}")

    # Clean up
    performance_monitor.shutdown()
    print("\n🧹 Cleanup completed")

    print("\n🎉 Monitoring and Alerting System Demo Completed Successfully!")
    print("\n📋 Summary of Implemented Features:")
    print("   ✅ Performance monitoring with CPU, memory, and latency tracking")
    print("   ✅ Comprehensive alerting system with configurable rules")
    print("   ✅ Error handling with graceful degradation")
    print("   ✅ Metrics collection for agents and workflows")
    print("   ✅ Health check system with connectivity diagnostics")
    print("   ✅ Dashboard integration for monitoring data")
    print("   ✅ Debug logging and troubleshooting tools")

except Exception as e:
    print(f"❌ Demo failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
