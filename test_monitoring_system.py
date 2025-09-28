#!/usr/bin/env python3
"""Test script for the Langfuse monitoring and alerting system."""

import sys
import os
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config.langfuse_monitoring_alerting import get_monitoring_system
    from config.langfuse_health_checker import get_health_checker
    from config.langfuse_monitoring_dashboard import get_monitoring_dashboard
    from config.metrics_collector import get_metrics_collector

    print("✅ Successfully imported monitoring components")

    # Test monitoring system
    print("\n🔍 Testing monitoring system...")
    monitoring_system = get_monitoring_system()

    # Start monitoring
    monitoring_system.start_monitoring()
    print("✅ Monitoring system started")

    # Wait a moment for initial metrics
    time.sleep(2)

    # Get monitoring summary
    summary = monitoring_system.get_monitoring_summary()
    print("✅ Got monitoring summary:")
    print(f"   - Health Status: {summary.health_status.value}")
    print(f"   - Active Alerts: {summary.active_alerts}")
    print(f"   - Uptime: {summary.uptime_seconds:.1f} seconds")

    # Test health checker
    print("\n🏥 Testing health checker...")
    health_checker = get_health_checker()
    diagnostics = health_checker.perform_comprehensive_health_check()

    print("✅ Health check completed:")
    print(f"   - Overall Status: {diagnostics.overall_status}")
    print(f"   - Connectivity Tests: {len(diagnostics.connectivity_tests)}")
    print(f"   - Configuration Issues: {len(diagnostics.configuration_issues)}")
    print(f"   - Security Issues: {len(diagnostics.security_issues)}")
    print(f"   - Recommendations: {len(diagnostics.recommendations)}")

    # Test dashboard
    print("\n📊 Testing dashboard...")
    dashboard = get_monitoring_dashboard()
    dashboard_metrics = dashboard.get_dashboard_metrics()

    print("✅ Dashboard metrics retrieved:")
    print(f"   - Health Score: {dashboard_metrics.health_score:.1f}/100")
    print(f"   - Recommendations: {len(dashboard_metrics.recommendations)}")

    # Test alert summary
    alert_summary = dashboard.get_alert_summary()
    print("✅ Alert summary retrieved:")
    print(f"   - Active Alerts: {alert_summary.total_active}")
    print(f"   - Resolution Rate: {alert_summary.resolution_rate:.1%}")

    # Test performance summary
    perf_summary = dashboard.get_performance_summary()
    print("✅ Performance summary retrieved:")
    print(f"   - Response Time: {perf_summary.average_response_time:.1f}ms")
    print(f"   - Throughput: {perf_summary.throughput:.2f} ops/sec")
    print(f"   - Trend: {perf_summary.trend}")

    # Test system overview
    overview = dashboard.get_system_overview()
    print("✅ System overview retrieved:")
    print(f"   - Overview Keys: {list(overview.keys())}")

    # Stop monitoring
    monitoring_system.stop_monitoring()
    print("✅ Monitoring system stopped")

    print("\n🎉 All monitoring and alerting tests passed!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Make sure all dependencies are installed and files are in the correct location"
    )
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
