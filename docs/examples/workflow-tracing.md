# Workflow Tracing Examples

This section demonstrates advanced workflow tracing patterns for complex, multi-step operations.

## Complex Workflow Example

```python
#!/usr/bin/env python3
"""
Advanced workflow tracing example demonstrating complex multi-agent scenarios.

This example shows how to:
- Trace complex workflows with multiple agents
- Handle collaboration and conflict resolution
- Track performance metrics across workflow steps
- Implement sophisticated error handling and recovery
"""

import sys
import os
import time
import random
from datetime import datetime
from typing import Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.langfuse_integration import get_langfuse_integration

class RetailOptimizationWorkflow:
    """Example of a complex retail optimization workflow with comprehensive tracing."""

    def __init__(self):
        self.service = get_langfuse_integration()
        self.workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def execute_full_workflow(self, market_event: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete retail optimization workflow with tracing."""
        print("=== Complex Workflow Example ===")
        print(f"Starting workflow: {self.workflow_id}")

        # Start the root workflow trace
        workflow_trace_id = self.service.create_simulation_trace({
            "type": "retail_optimization",
            "trigger": market_event.get("type", "unknown"),
            "workflow_id": self.workflow_id,
            "market_conditions": market_event
        })

        if not workflow_trace_id:
            print("Failed to create workflow trace")
            return {"status": "failed", "error": "trace_creation_failed"}

        try:
            # Phase 1: Market Analysis
            print("\n1. Market Analysis Phase...")
            market_analysis = self._phase_market_analysis(workflow_trace_id, market_event)

            # Phase 2: Agent Coordination
            print("\n2. Agent Coordination Phase...")
            coordination_result = self._phase_agent_coordination(
                workflow_trace_id, market_analysis
            )

            # Phase 3: Decision Implementation
            print("\n3. Decision Implementation Phase...")
            implementation_result = self._phase_decision_implementation(
                workflow_trace_id, coordination_result
            )

            # Phase 4: Performance Monitoring
            print("\n4. Performance Monitoring Phase...")
            monitoring_result = self._phase_performance_monitoring(
                workflow_trace_id, implementation_result
            )

            # Finalize workflow
            final_outcome = {
                "status": "completed",
                "workflow_id": self.workflow_id,
                "phases_completed": 4,
                "total_agents_involved": 3,
                "decisions_made": len(coordination_result.get("decisions", [])),
                "expected_impact": monitoring_result.get("expected_impact", {}),
                "processing_time_ms": int(time.time() * 1000) - int(datetime.now().timestamp() * 1000)
            }

            self.service.finalize_trace(workflow_trace_id, final_outcome)

            print(f"\n✓ Workflow completed successfully: {self.workflow_id}")
            return final_outcome

        except Exception as e:
            print(f"\n✗ Workflow failed: {e}")

            # Log error and finalize with failure status
            self.service.log_error(workflow_trace_id, e, {
                "workflow_id": self.workflow_id,
                "failed_phase": "unknown"
            })

            self.service.finalize_trace(workflow_trace_id, {
                "status": "failed",
                "error": str(e),
                "workflow_id": self.workflow_id
            })

            return {"status": "failed", "error": str(e)}

    def _phase_market_analysis(self, workflow_trace_id: str, market_event: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Analyze market conditions and trigger appropriate agents."""
        phase_trace_id = self.service.track_collaboration(
            workflow_id=f"{self.workflow_id}_market_analysis",
            participating_agents=["market_analyzer"],
            workflow_data={"phase": "market_analysis", "input": market_event}
        )

        try:
            # Simulate market analysis
            analysis_span = self.service.start_agent_span(
                agent_id="market_analyzer",
                operation="analyze_market_conditions",
                parent_trace_id=phase_trace_id,
                input_data=market_event
            )

            # Simulate analysis work
            time.sleep(0.1)

            market_analysis = {
                "market_trend": "increasing_demand" if random.random() > 0.3 else "stable_demand",
                "competitor_activity": "high" if random.random() > 0.5 else "normal",
                "seasonal_factor": 1.2 if datetime.now().month in [11, 12] else 1.0,
                "economic_indicators": {
                    "consumer_confidence": random.uniform(0.7, 0.9),
                    "inflation_rate": random.uniform(0.02, 0.05)
                },
                "recommendations": [
                    "monitor_demand_closely",
                    "prepare_for_price_adjustments",
                    "consider_promotional_campaigns"
                ]
            }

            self.service.end_agent_span(analysis_span, {
                "analysis_result": market_analysis,
                "confidence": 0.85,
                "processing_time_ms": 100
            })

            # Log the market analysis decision
            self.service.log_agent_decision("market_analyzer", {
                "decision_type": "market_assessment",
                "inputs": market_event,
                "outputs": market_analysis,
                "confidence": 0.85,
                "reasoning": "Market analysis based on current conditions and historical trends"
            })

            self.service.finalize_trace(phase_trace_id, {
                "status": "completed",
                "analysis_complete": True,
                "agents_involved": 1
            })

            return market_analysis

        except Exception as e:
            self.service.finalize_trace(phase_trace_id, {
                "status": "failed",
                "error": str(e)
            })
            raise

    def _phase_agent_coordination(self, workflow_trace_id: str, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Coordinate multiple agents to make optimization decisions."""
        phase_trace_id = self.service.track_collaboration(
            workflow_id=f"{self.workflow_id}_agent_coordination",
            participating_agents=["inventory_agent", "pricing_agent", "promotion_agent"],
            workflow_data={"phase": "agent_coordination", "market_analysis": market_analysis}
        )

        try:
            coordination_decisions = []

            # Inventory Agent Decision
            inventory_span = self.service.start_agent_span(
                agent_id="inventory_agent",
                operation="optimize_inventory_levels",
                parent_trace_id=phase_trace_id,
                input_data={"market_analysis": market_analysis}
            )

            inventory_decision = {
                "agent": "inventory_agent",
                "action": "increase_safety_stock",
                "product_categories": ["electronics", "seasonal_items"],
                "increase_percentage": 0.25,
                "reasoning": "Market analysis indicates increasing demand trend",
                "confidence": 0.82,
                "expected_impact": {"stockout_reduction": 0.15}
            }

            time.sleep(0.05)  # Simulate processing time
            coordination_decisions.append(inventory_decision)

            self.service.end_agent_span(inventory_span, inventory_decision)

            # Pricing Agent Decision
            pricing_span = self.service.start_agent_span(
                agent_id="pricing_agent",
                operation="calculate_optimal_pricing",
                parent_trace_id=phase_trace_id,
                input_data={"market_analysis": market_analysis, "inventory_decision": inventory_decision}
            )

            pricing_decision = {
                "agent": "pricing_agent",
                "action": "dynamic_pricing_adjustment",
                "price_changes": [
                    {"category": "high_demand", "adjustment": 0.05, "direction": "increase"},
                    {"category": "low_demand", "adjustment": -0.08, "direction": "decrease"}
                ],
                "reasoning": "Balance between demand trends and inventory levels",
                "confidence": 0.78,
                "expected_impact": {"revenue_increase": 0.12, "margin_improvement": 0.03}
            }

            time.sleep(0.05)
            coordination_decisions.append(pricing_decision)

            self.service.end_agent_span(pricing_span, pricing_decision)

            # Promotion Agent Decision
            promotion_span = self.service.start_agent_span(
                agent_id="promotion_agent",
                operation="plan_promotional_campaigns",
                parent_trace_id=phase_trace_id,
                input_data={
                    "market_analysis": market_analysis,
                    "inventory_decision": inventory_decision,
                    "pricing_decision": pricing_decision
                }
            )

            promotion_decision = {
                "agent": "promotion_agent",
                "action": "targeted_promotions",
                "campaigns": [
                    {
                        "type": "flash_sale",
                        "target_products": ["high_demand_items"],
                        "discount_percentage": 0.15,
                        "duration_hours": 24
                    }
                ],
                "reasoning": "Capitalize on increased demand with strategic promotions",
                "confidence": 0.75,
                "expected_impact": {"sales_boost": 0.20, "customer_engagement": 0.10}
            }

            time.sleep(0.05)
            coordination_decisions.append(promotion_decision)

            self.service.end_agent_span(promotion_span, promotion_decision)

            # Log collaboration outcome
            self.service.log_agent_decision("coordination_manager", {
                "decision_type": "multi_agent_coordination",
                "inputs": {"market_analysis": market_analysis},
                "outputs": {"coordination_decisions": coordination_decisions},
                "confidence": 0.80,
                "reasoning": "Coordinated decisions across inventory, pricing, and promotion agents"
            })

            self.service.finalize_trace(phase_trace_id, {
                "status": "completed",
                "decisions_made": len(coordination_decisions),
                "agents_involved": 3,
                "coordination_success": True
            })

            return {"decisions": coordination_decisions, "coordination_success": True}

        except Exception as e:
            self.service.finalize_trace(phase_trace_id, {
                "status": "failed",
                "error": str(e),
                "agents_involved": 3
            })
            raise

    def _phase_decision_implementation(self, workflow_trace_id: str, coordination_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Implement the coordinated decisions."""
        phase_trace_id = self.service.track_collaboration(
            workflow_id=f"{self.workflow_id}_decision_implementation",
            participating_agents=["implementation_manager"],
            workflow_data={"phase": "implementation", "decisions": coordination_result}
        )

        try:
            implementation_span = self.service.start_agent_span(
                agent_id="implementation_manager",
                operation="implement_decisions",
                parent_trace_id=phase_trace_id,
                input_data={"decisions": coordination_result["decisions"]}
            )

            # Simulate implementation work
            time.sleep(0.08)

            implementation_results = []
            for decision in coordination_result["decisions"]:
                result = {
                    "agent": decision["agent"],
                    "action_implemented": True,
                    "implementation_time_ms": random.randint(50, 150),
                    "systems_updated": ["inventory_system", "pricing_engine", "promotion_platform"],
                    "rollback_available": True
                }
                implementation_results.append(result)

            self.service.end_agent_span(implementation_span, {
                "implementation_results": implementation_results,
                "total_actions": len(implementation_results),
                "success_rate": 1.0
            })

            self.service.finalize_trace(phase_trace_id, {
                "status": "completed",
                "implementations_completed": len(implementation_results),
                "success_rate": 1.0
            })

            return {"implementation_results": implementation_results, "success": True}

        except Exception as e:
            self.service.finalize_trace(phase_trace_id, {
                "status": "failed",
                "error": str(e)
            })
            raise

    def _phase_performance_monitoring(self, workflow_trace_id: str, implementation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Monitor the impact of implemented decisions."""
        phase_trace_id = self.service.track_collaboration(
            workflow_id=f"{self.workflow_id}_performance_monitoring",
            participating_agents=["monitoring_agent"],
            workflow_data={"phase": "monitoring", "implementation": implementation_result}
        )

        try:
            monitoring_span = self.service.start_agent_span(
                agent_id="monitoring_agent",
                operation="track_performance_impact",
                parent_trace_id=phase_trace_id,
                input_data={"implementation": implementation_result}
            )

            # Simulate monitoring work
            time.sleep(0.03)

            # Generate mock performance metrics
            performance_metrics = {
                "sales_impact": {
                    "revenue_change": 0.12,
                    "units_sold_change": 0.08,
                    "customer_satisfaction": 0.05
                },
                "operational_impact": {
                    "inventory_turnover": 0.15,
                    "stockout_rate": -0.10,
                    "operational_efficiency": 0.07
                },
                "expected_vs_actual": {
                    "revenue_variance": 0.02,
                    "timing_variance_days": 1,
                    "overall_accuracy": 0.88
                }
            }

            self.service.end_agent_span(monitoring_span, {
                "performance_metrics": performance_metrics,
                "monitoring_complete": True
            })

            self.service.finalize_trace(phase_trace_id, {
                "status": "completed",
                "metrics_collected": True,
                "impact_assessed": True
            })

            return {"performance_metrics": performance_metrics, "monitoring_complete": True}

def run_workflow_example():
    """Run the complex workflow example."""
    # Sample market event
    market_event = {
        "type": "seasonal_demand_increase",
        "magnitude": 1.8,
        "affected_categories": ["electronics", "seasonal_items"],
        "duration_days": 30,
        "trigger_source": "market_analysis",
        "timestamp": datetime.now().isoformat()
    }

    # Execute workflow
    workflow = RetailOptimizationWorkflow()
    result = workflow.execute_full_workflow(market_event)

    # Display results
    print("
=== Workflow Results ===")
    print(f"Status: {result.get('status')}")
    print(f"Workflow ID: {result.get('workflow_id')}")
    print(f"Processing time: {result.get('processing_time_ms', 0)}ms")

    if result.get('status') == 'completed':
        print("✓ Workflow executed successfully")
        print(f"✓ {result.get('phases_completed', 0)} phases completed")
        print(f"✓ {result.get('total_agents_involved', 0)} agents involved")
    else:
        print(f"✗ Workflow failed: {result.get('error')}")

    return result

def main():
    """Main function to run the workflow example."""
    print("Advanced Workflow Tracing Example")
    print("=" * 50)

    try:
        result = run_workflow_example()

        print("\n" + "=" * 50)
        print("Example completed successfully!")

        if result.get('status') == 'completed':
            print("\nNext steps:")
            print("1. Check Langfuse dashboard for detailed trace visualization")
            print("2. Analyze performance metrics and timing")
            print("3. Review agent coordination patterns")
            print("4. Examine error handling and recovery mechanisms")

    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

## Workflow Visualization

When this example runs with Langfuse properly configured, it creates a comprehensive trace hierarchy:

```
Retail Optimization Workflow (Root)
├── Market Analysis Phase
│   └── Market Analyzer: analyze_market_conditions
├── Agent Coordination Phase
│   ├── Inventory Agent: optimize_inventory_levels
│   ├── Pricing Agent: calculate_optimal_pricing
│   └── Promotion Agent: plan_promotional_campaigns
├── Decision Implementation Phase
│   └── Implementation Manager: implement_decisions
└── Performance Monitoring Phase
    └── Monitoring Agent: track_performance_impact
```

## Key Features Demonstrated

### 1. Multi-Phase Workflow Tracing
- Each phase has its own collaboration trace
- Clear separation of concerns between phases
- Proper error propagation and handling

### 2. Agent Coordination
- Multiple agents working on related decisions
- Cross-agent data dependencies
- Coordinated decision making

### 3. Rich Metadata
- Detailed input/output data for each operation
- Performance metrics and timing information
- Decision confidence and reasoning

### 4. Error Handling
- Graceful failure handling at each phase
- Proper trace finalization even on errors
- Detailed error context for debugging

### 5. Performance Monitoring
- End-to-end workflow timing
- Per-phase performance metrics
- Impact assessment and validation

## Expected Dashboard View

In the Langfuse dashboard, you should see:

1. **Timeline View**: Clear progression through 4 phases
2. **Dependency Graph**: Shows data flow between agents
3. **Performance Metrics**: Response times for each operation
4. **Error Tracking**: Any failures with full context
5. **Custom Metadata**: Business-specific information

This example demonstrates how to implement comprehensive tracing for complex, real-world workflows while maintaining performance and providing rich observability data.