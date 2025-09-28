#!/usr/bin/env python3
"""Demo script showing metrics collection functionality."""

import time
from config.metrics_collector import get_metrics_collector
from config.langfuse_integration import get_langfuse_integration


def demo_metrics_collection():
    """Demonstrate metrics collection capabilities."""
    print("🚀 Starting Metrics Collection Demo")

    # Get instances
    metrics_collector = get_metrics_collector()
    langfuse_integration = get_langfuse_integration()

    # Simulate agent operations
    print("\n📊 Simulating agent operations...")

    # Agent 1: Inventory Agent
    inventory_agent = "inventory_agent"
    langfuse_integration.start_agent_span(inventory_agent, "forecast_demand", "sim_001")
    time.sleep(0.1)  # Simulate work
    langfuse_integration.end_agent_span(
        "inventory_agent_forecast_demand_001", {"forecast": "high"}
    )

    # Agent 2: Pricing Agent
    pricing_agent = "pricing_agent"
    langfuse_integration.start_agent_span(pricing_agent, "calculate_price", "sim_001")
    time.sleep(0.05)  # Simulate work
    langfuse_integration.end_agent_span(
        "pricing_agent_calculate_price_001", {"price": 99.99}
    )

    # Agent 3: Promotion Agent
    promotion_agent = "promotion_agent"
    langfuse_integration.start_agent_span(promotion_agent, "create_campaign", "sim_001")
    time.sleep(0.08)  # Simulate work
    langfuse_integration.end_agent_span(
        "promotion_agent_create_campaign_001", {"campaign": "summer_sale"}
    )

    # Simulate workflow
    print("\n🔄 Simulating workflow...")
    workflow_id = "workflow_001"
    langfuse_integration._metrics_collector.start_workflow(
        workflow_id, [inventory_agent, pricing_agent, promotion_agent]
    )
    time.sleep(0.2)
    langfuse_integration._metrics_collector.end_workflow(
        workflow_id, success=True, coordination_events=3
    )

    # Record collaboration
    metrics_collector.record_agent_collaboration(inventory_agent)
    metrics_collector.record_agent_collaboration(pricing_agent)

    # Get and display metrics
    print("\n📈 Agent Metrics:")
    for agent_id in [inventory_agent, pricing_agent, promotion_agent]:
        agent_metrics = langfuse_integration.get_agent_metrics(agent_id)
        if agent_metrics:
            print(f"  {agent_id}:")
            print(f"    Operations: {agent_metrics['operation_count']}")
            print(
                f"    Avg Response Time: {agent_metrics['average_response_time']:.3f}s"
            )
            print(f"    Success Rate: {agent_metrics['success_rate']:.1%}")
            print(f"    Collaborations: {agent_metrics['collaboration_count']}")

    print("\n📊 System Metrics:")
    system_metrics = langfuse_integration.get_system_metrics()
    print(f"  Total Events: {system_metrics['total_events_processed']}")
    print(f"  Workflows Completed: {system_metrics['total_workflows_completed']}")
    print(
        f"  Avg Workflow Duration: {system_metrics['average_workflow_duration']:.3f}s"
    )
    print(f"  System Throughput: {system_metrics['system_throughput']:.3f} events/sec")
    print(f"  Error Rate: {system_metrics['error_rate']:.1%}")

    # Export for dashboard
    print("\n📋 Dashboard Export Sample:")
    export_data = langfuse_integration.export_metrics_for_dashboard()
    print(f"  Export contains {len(export_data['agents'])} agents")
    print(
        f"  System throughput: {export_data['system']['system_throughput']:.3f} events/sec"
    )

    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    demo_metrics_collection()
