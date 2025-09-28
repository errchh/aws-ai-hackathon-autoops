#!/usr/bin/env python3
"""
Performance testing script for the autoops retail optimization system.

This script runs comprehensive performance tests to validate system behavior
under various load conditions, measuring response times, throughput, and
system stability.
"""

import asyncio
import time
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import json

# Mock strands imports since the package has build issues
from unittest.mock import MagicMock
import sys

sys.modules["strands"] = MagicMock()
sys.modules["strands.models"] = MagicMock()

from agents.orchestrator import RetailOptimizationOrchestrator
from scenarios.data_generator import ScenarioDataGenerator


class PerformanceTestSuite:
    """Comprehensive performance testing suite for the retail optimization system."""

    def __init__(self):
        """Initialize the performance test suite."""
        self.orchestrator = RetailOptimizationOrchestrator()
        self.scenario_generator = ScenarioDataGenerator(seed=42)

        # Register agents
        try:
            from agents.pricing_agent import pricing_agent
            from agents.inventory_agent import inventory_agent
            from agents.promotion_agent import promotion_agent

            agents = [pricing_agent, inventory_agent, promotion_agent]
            success = self.orchestrator.register_agents(agents)
            if not success:
                print(
                    "⚠️  Warning: Failed to register agents, tests may not work properly"
                )
        except Exception as e:
            print(f"⚠️  Warning: Could not import agents: {e}")

    async def run_load_test(
        self, num_events: int = 10, concurrency: int = 5, name: str = "Load Test"
    ) -> Dict[str, Any]:
        """
        Run a load test with specified parameters.

        Args:
            num_events: Number of market events to process
            concurrency: Number of concurrent requests
            name: Name of the test

        Returns:
            Dictionary containing test results
        """
        print(f"🚀 Running {name}: {num_events} events with concurrency {concurrency}")

        # Generate test events
        events = self.scenario_generator.generate_market_events(num_events)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(event):
            async with semaphore:
                event_data = {
                    "event_id": str(event.id),
                    "event_type": event.event_type.value,
                    "affected_products": event.affected_products,
                    "impact_magnitude": event.impact_magnitude,
                    "metadata": event.metadata,
                }
                start_time = time.time()
                try:
                    result = await self.orchestrator.process_market_event(event_data)
                    end_time = time.time()
                    return {
                        "result": result,
                        "response_time": end_time - start_time,
                        "success": result["status"]
                        in ["completed", "no_response_needed"],
                        "error": None,
                    }
                except Exception as e:
                    end_time = time.time()
                    return {
                        "result": None,
                        "response_time": end_time - start_time,
                        "success": False,
                        "error": str(e),
                    }

        # Run concurrent load test
        start_time = time.time()
        tasks = [process_with_semaphore(event) for event in events]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Process results
        successful_results = []
        failed_results = []

        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result), "response_time": 0})
            elif isinstance(result, dict) and result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)

        response_times = [r["response_time"] for r in successful_results]

        # Calculate statistics
        test_results = {
            "test_name": name,
            "total_events": num_events,
            "concurrency": concurrency,
            "successful": len(successful_results),
            "failed": len(failed_results),
            "success_rate": len(successful_results) / num_events
            if num_events > 0
            else 0,
            "total_time": total_time,
            "throughput": num_events / total_time if total_time > 0 else 0,
        }

        if response_times:
            test_results.update(
                {
                    "avg_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "response_time_stddev": statistics.stdev(response_times)
                    if len(response_times) > 1
                    else 0,
                    "p95_response_time": statistics.quantiles(response_times, n=20)[18]
                    if len(response_times) >= 20
                    else max(response_times),
                }
            )
        else:
            test_results.update(
                {
                    "avg_response_time": 0,
                    "median_response_time": 0,
                    "min_response_time": 0,
                    "max_response_time": 0,
                    "response_time_stddev": 0,
                    "p95_response_time": 0,
                }
            )

        return test_results

    async def run_comprehensive_performance_test(self):
        """Run comprehensive performance tests with multiple scenarios."""
        print("🧪 Starting Comprehensive Performance Test Suite")
        print("=" * 60)

        test_scenarios = [
            {"num_events": 5, "concurrency": 2, "name": "Light Load Test"},
            {"num_events": 10, "concurrency": 5, "name": "Medium Load Test"},
            {"num_events": 20, "concurrency": 10, "name": "Heavy Load Test"},
            {"num_events": 50, "concurrency": 5, "name": "High Volume Test"},
            {"num_events": 10, "concurrency": 20, "name": "High Concurrency Test"},
        ]

        all_results = []

        for scenario in test_scenarios:
            try:
                result = await self.run_load_test(**scenario)
                all_results.append(result)
                self._print_test_results(result)
                print()

                # Small delay between tests
                await asyncio.sleep(1)

            except Exception as e:
                print(f"❌ Test {scenario['name']} failed: {e}")
                continue

        # Print summary
        self._print_performance_summary(all_results)

        return all_results

    def _print_test_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        print(f"📊 {results['test_name']} Results:")
        print(
            f"  Events: {results['total_events']} | Concurrency: {results['concurrency']}"
        )
        print(f"  Success Rate: {results['success_rate']:.1%}")
        print(f"  Total Time: {results['total_time']:.2f}s")
        print(f"  Throughput: {results['throughput']:.3f} req/s")
        print(f"  Success/Fail: {results['successful']}/{results['failed']}")
        if results["successful"] > 0:
            print("  Response Times:")
            print(f"    Average: {results['avg_response_time']:.3f}s")
            print(f"    Median: {results['median_response_time']:.3f}s")
            print(f"    Min: {results['min_response_time']:.3f}s")
            print(f"    Max: {results['max_response_time']:.3f}s")
            print(f"    P95: {results['p95_response_time']:.3f}s")
        else:
            print("  ❌ No successful requests")

    def _print_performance_summary(self, all_results: List[Dict[str, Any]]):
        """Print comprehensive performance summary."""
        print("📈 Performance Test Summary")
        print("=" * 60)

        if not all_results:
            print("❌ No test results available")
            return

        # Overall statistics
        total_events = sum(r["total_events"] for r in all_results)
        total_successful = sum(r["successful"] for r in all_results)
        overall_success_rate = (
            total_successful / total_events if total_events > 0 else 0
        )

        avg_response_times = [
            r["avg_response_time"] for r in all_results if r["successful"] > 0
        ]
        overall_avg_response = (
            statistics.mean(avg_response_times) if avg_response_times else 0
        )

        throughputs = [r["throughput"] for r in all_results]
        avg_throughput = statistics.mean(throughputs) if throughputs else 0

        print("🎯 Overall Performance:")
        print(f"  Overall Success Rate: {overall_success_rate:.1%}")
        print(f"  Average Response Time: {overall_avg_response:.3f}s")
        print(f"  Average Throughput: {avg_throughput:.2f} req/s")
        print(f"  Total Events Processed: {total_events}")
        print()

        # Test-specific results
        print("📋 Individual Test Results:")
        for result in all_results:
            status = (
                "✅"
                if result["success_rate"] >= 0.8
                else "⚠️"
                if result["success_rate"] >= 0.5
                else "❌"
            )
            print(
                f"  {status} {result['test_name']}: {result['success_rate']:.1%} success, "
                f"{result['avg_response_time']:.3f}s avg, {result['throughput']:.2f} req/s"
            )

        print()
        print("🎖️  Performance Assessment:")
        if overall_success_rate >= 0.95 and overall_avg_response < 5.0:
            print("  🟢 EXCELLENT: System handles load well with fast responses")
        elif overall_success_rate >= 0.80 and overall_avg_response < 10.0:
            print("  🟡 GOOD: System performs adequately under load")
        elif overall_success_rate >= 0.50:
            print("  🟠 FAIR: System needs optimization for better performance")
        else:
            print("  🔴 POOR: System requires significant performance improvements")

    async def run_memory_performance_test(self):
        """Test memory system performance under load."""
        print("🧠 Testing Memory System Performance")
        print("-" * 40)

        try:
            from agents.memory import agent_memory
            from scenarios.data_generator import ScenarioDataGenerator

            generator = ScenarioDataGenerator(seed=123)
            decisions = generator.generate_agent_decisions(20)

            # Test memory storage performance
            store_times = []
            for decision in decisions:
                context = {
                    "test": "performance",
                    "product_id": f"TEST_{decision.agent_id}",
                }

                start_time = time.time()
                memory_id = agent_memory.store_decision(
                    agent_id=decision.agent_id, decision=decision, context=context
                )
                end_time = time.time()
                store_times.append(end_time - start_time)

            # Test memory retrieval performance
            retrieve_times = []
            for _ in range(10):
                start_time = time.time()
                results = agent_memory.retrieve_similar_decisions(
                    agent_id="pricing_agent",
                    current_context={"test": "performance"},
                    limit=5,
                )
                end_time = time.time()
                retrieve_times.append(end_time - start_time)

            # Calculate statistics
            memory_results = {
                "memory_operations": len(decisions) + 10,
                "store_avg_time": statistics.mean(store_times),
                "store_max_time": max(store_times),
                "retrieve_avg_time": statistics.mean(retrieve_times),
                "retrieve_max_time": max(retrieve_times),
                "total_memory_entries": len(decisions),
            }

            print("  Memory Storage Performance:")
            print(f"    Average store time: {memory_results['store_avg_time']:.4f}s")
            print(f"    Max store time: {memory_results['store_max_time']:.4f}s")
            print("  Memory Retrieval Performance:")
            print(
                f"    Average retrieve time: {memory_results['retrieve_avg_time']:.4f}s"
            )
            print(f"    Max retrieve time: {memory_results['retrieve_max_time']:.4f}s")
            print(f"  Total entries stored: {memory_results['total_memory_entries']}")

            return memory_results

        except Exception as e:
            print(f"❌ Memory performance test failed: {e}")
            return None

    async def run_end_to_end_workflow_test(self):
        """Test complete end-to-end workflows."""
        print("🔄 Testing End-to-End Workflows")
        print("-" * 40)

        workflow_scenarios = [
            {
                "name": "Demand Spike Response",
                "events": [
                    {
                        "event_type": "demand_spike",
                        "affected_products": ["COFFEE_001", "COFFEE_002"],
                        "impact_magnitude": 2.5,
                        "metadata": {"viral_trend": True},
                    }
                ],
            },
            {
                "name": "Multi-Agent Coordination",
                "events": [
                    {
                        "event_type": "competitor_price_change",
                        "affected_products": ["LAPTOP_001"],
                        "impact_magnitude": -0.15,
                        "metadata": {"competitor": "major_brand"},
                    },
                    {
                        "event_type": "demand_spike",
                        "affected_products": ["LAPTOP_001"],
                        "impact_magnitude": 1.8,
                        "metadata": {"back_to_school": True},
                    },
                ],
            },
        ]

        workflow_results = []

        for scenario in workflow_scenarios:
            print(f"  Testing {scenario['name']}...")

            scenario_times = []
            success_count = 0

            for event in scenario["events"]:
                start_time = time.time()
                result = await self.orchestrator.process_market_event(event)
                end_time = time.time()

                scenario_times.append(end_time - start_time)
                if result["status"] in ["completed", "no_response_needed"]:
                    success_count += 1

            workflow_result = {
                "scenario": scenario["name"],
                "events_processed": len(scenario["events"]),
                "successful_events": success_count,
                "success_rate": success_count / len(scenario["events"]),
                "avg_processing_time": statistics.mean(scenario_times),
                "total_time": sum(scenario_times),
            }

            workflow_results.append(workflow_result)

            status = "✅" if workflow_result["success_rate"] == 1.0 else "⚠️"
            print(
                f"    {status} {workflow_result['success_rate']:.0%} success, "
                f"{workflow_result['avg_processing_time']:.3f}s avg"
            )

        return workflow_results


async def main():
    """Run the complete performance test suite."""
    print("🚀 AutoOps Retail Optimization - Performance Test Suite")
    print("=" * 80)

    # Initialize test suite
    test_suite = PerformanceTestSuite()

    try:
        # Run comprehensive performance tests
        performance_results = await test_suite.run_comprehensive_performance_test()

        # Run memory performance tests
        memory_results = await test_suite.run_memory_performance_test()

        # Run end-to-end workflow tests
        workflow_results = await test_suite.run_end_to_end_workflow_test()

        # Final summary
        print("\n" + "=" * 80)
        print("✅ Performance Testing Complete!")
        print("=" * 80)

        print("📊 Test Summary:")
        print(f"  • Performance scenarios tested: {len(performance_results)}")
        print(
            f"  • Memory operations tested: {memory_results['memory_operations'] if memory_results else 0}"
        )
        print(f"  • Workflow scenarios tested: {len(workflow_results)}")

        # Save results to file
        results_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_tests": performance_results,
            "memory_tests": memory_results,
            "workflow_tests": workflow_results,
        }

        with open("performance_test_results.json", "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        print("💾 Results saved to performance_test_results.json")

    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
