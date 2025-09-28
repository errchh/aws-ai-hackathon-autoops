"""
Advanced load testing for high-frequency simulation events in Langfuse workflow visualization.

This module contains comprehensive load testing scenarios to validate system performance
under high-frequency simulation events, stress conditions, and sustained load.
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Deque
from unittest.mock import Mock, patch

import pytest

from config.langfuse_integration import LangfuseIntegrationService
from config.simulation_event_capture import SimulationEventCapture
from config.orchestrator_tracing import OrchestrationTracer
from tests.test_data_generators import TestDataGenerator, MockLangfuseClient


class LoadTestMetricsCollector:
    """Collects and analyzes load test metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.response_times: Deque[float] = deque(maxlen=10000)
        self.throughput_samples: Deque[float] = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.memory_samples: Deque[float] = deque(maxlen=1000)
        self.cpu_samples: Deque[float] = deque(maxlen=1000)
        self.trace_counts: Deque[int] = deque(maxlen=1000)
        self.start_time = time.time()

    def record_response_time(self, response_time: float):
        """Record a response time sample."""
        self.response_times.append(response_time)

    def record_throughput(self, operations_per_second: float):
        """Record throughput sample."""
        self.throughput_samples.append(operations_per_second)

    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1

    def record_system_metrics(self, memory_mb: float, cpu_percent: float):
        """Record system metrics."""
        self.memory_samples.append(memory_mb)
        self.cpu_samples.append(cpu_percent)

    def record_trace_count(self, count: int):
        """Record trace count."""
        self.trace_counts.append(count)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        current_time = time.time()
        duration = current_time - self.start_time

        if not self.response_times:
            return {"error": "No response time data collected"}

        response_stats = {
            "mean": statistics.mean(self.response_times),
            "median": statistics.median(self.response_times),
            "min": min(self.response_times),
            "max": max(self.response_times),
            "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else max(self.response_times),
            "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else max(self.response_times)
        }

        throughput_stats = {
            "mean": statistics.mean(self.throughput_samples) if self.throughput_samples else 0,
            "current": self.throughput_samples[-1] if self.throughput_samples else 0
        }

        memory_stats = {
            "mean": statistics.mean(self.memory_samples) if self.memory_samples else 0,
            "max": max(self.memory_samples) if self.memory_samples else 0,
            "current": self.memory_samples[-1] if self.memory_samples else 0
        }

        cpu_stats = {
            "mean": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            "max": max(self.cpu_samples) if self.cpu_samples else 0,
            "current": self.cpu_samples[-1] if self.cpu_samples else 0
        }

        total_operations = len(self.response_times)
        total_errors = sum(self.error_counts.values())
        error_rate = total_errors / total_operations if total_operations > 0 else 0

        return {
            "duration_seconds": duration,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "response_times": response_stats,
            "throughput": throughput_stats,
            "memory_usage_mb": memory_stats,
            "cpu_usage_percent": cpu_stats,
            "error_breakdown": dict(self.error_counts)
        }


class LoadTestScenario:
    """Defines a load testing scenario."""

    def __init__(
        self,
        name: str,
        duration_seconds: int,
        target_throughput: float,
        ramp_up_seconds: int = 30,
        ramp_down_seconds: int = 30,
        description: str = ""
    ):
        """Initialize load test scenario."""
        self.name = name
        self.duration_seconds = duration_seconds
        self.target_throughput = target_throughput
        self.ramp_up_seconds = ramp_up_seconds
        self.ramp_down_seconds = ramp_down_seconds
        self.description = description

    def get_target_rate_at_time(self, elapsed_seconds: float) -> float:
        """Get target operation rate at given elapsed time."""
        if elapsed_seconds < self.ramp_up_seconds:
            # Ramp up phase
            return (elapsed_seconds / self.ramp_up_seconds) * self.target_throughput
        elif elapsed_seconds < self.duration_seconds - self.ramp_down_seconds:
            # Sustained load phase
            return self.target_throughput
        elif elapsed_seconds < self.duration_seconds:
            # Ramp down phase
            remaining_ramp_down = self.duration_seconds - elapsed_seconds
            return (remaining_ramp_down / self.ramp_down_seconds) * self.target_throughput
        else:
            # Test completed
            return 0.0


class HighFrequencyLoadTester:
    """Advanced load tester for high-frequency simulation events."""

    def __init__(self):
        """Initialize load tester."""
        self.test_generator = TestDataGenerator(seed=42)
        self.mock_client = MockLangfuseClient()
        self.metrics_collector = LoadTestMetricsCollector()

    async def run_high_frequency_load_test(
        self,
        scenario: LoadTestScenario,
        integration_service: LangfuseIntegrationService,
        system_metrics_interval: float = 1.0
    ) -> Dict[str, Any]:
        """Run high-frequency load test scenario."""
        print(f"Running load test: {scenario.name}")
        print(f"  Duration: {scenario.duration_seconds}s")
        print(f"  Target throughput: {scenario.target_throughput} ops/sec")
        print(f"  Description: {scenario.description}")

        # System metrics collection task
        async def collect_system_metrics():
            """Collect system metrics periodically."""
            import psutil
            process = psutil.Process()

            while True:
                try:
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()

                    self.metrics_collector.record_system_metrics(memory_mb, cpu_percent)
                    self.metrics_collector.record_trace_count(len(self.mock_client.traces))

                    await asyncio.sleep(system_metrics_interval)
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
                    break

        # Start system metrics collection
        metrics_task = asyncio.create_task(collect_system_metrics())

        # Load generation task
        async def generate_load():
            """Generate load according to scenario."""
            start_time = time.time()
            operation_count = 0

            while True:
                elapsed = time.time() - start_time

                if elapsed >= scenario.duration_seconds:
                    break

                # Get target rate for current time
                target_rate = scenario.get_target_rate_at_time(elapsed)

                if target_rate > 0:
                    # Calculate delay between operations
                    delay = 1.0 / target_rate

                    # Execute operation
                    await self._execute_single_operation(integration_service, operation_count)
                    operation_count += 1

                    # Wait for next operation (with jitter to avoid thundering herd)
                    jitter = delay * 0.1 * (2 * asyncio.get_event_loop().time() % 1 - 0.5)
                    await asyncio.sleep(max(0, delay + jitter))
                else:
                    await asyncio.sleep(0.1)

        # Start load generation
        try:
            await asyncio.wait_for(generate_load(), timeout=scenario.duration_seconds + 10)
        except asyncio.TimeoutError:
            print("Load test timed out")

        # Stop metrics collection
        metrics_task.cancel()

        # Get final metrics
        summary = self.metrics_collector.get_summary_stats()

        # Add scenario information
        summary.update({
            "scenario_name": scenario.name,
            "scenario_description": scenario.description,
            "target_throughput": scenario.target_throughput,
            "actual_throughput": summary["total_operations"] / max(summary["duration_seconds"], 1),
            "throughput_efficiency": (
                summary["total_operations"] / max(summary["duration_seconds"], 1)
            ) / scenario.target_throughput if scenario.target_throughput > 0 else 0
        })

        return summary

    async def _execute_single_operation(
        self,
        integration_service: LangfuseIntegrationService,
        operation_id: int
    ):
        """Execute a single load test operation."""
        start_time = time.time()

        try:
            with patch.object(integration_service, '_client', self.mock_client):
                # Generate test event
                events = self.test_generator.generate_market_events(1)
                event = events[0]

                # Create trace
                trace_id = integration_service.create_simulation_trace({
                    "event_type": "load_test",
                    "event_id": f"load_{operation_id}",
                    "operation_id": operation_id,
                    "timestamp": datetime.now().isoformat()
                })

                if trace_id:
                    # Create multiple spans to simulate complex workflow
                    for span_idx in range(3):
                        span_id = integration_service.start_agent_span(
                            f"load_agent_{span_idx}",
                            f"load_operation_{span_idx}",
                            input_data={"operation_id": operation_id, "span_index": span_idx}
                        )

                        if span_id:
                            integration_service.end_agent_span(span_id, {
                                "status": "completed",
                                "operation_id": operation_id
                            })

                    integration_service.finalize_trace(trace_id)

                # Record success
                end_time = time.time()
                self.metrics_collector.record_response_time(end_time - start_time)

                # Update throughput tracking
                current_throughput = 1.0 / (end_time - start_time) if end_time > start_time else 0
                self.metrics_collector.record_throughput(current_throughput)

        except Exception as e:
            # Record error
            end_time = time.time()
            self.metrics_collector.record_response_time(end_time - start_time)
            self.metrics_collector.record_error(f"exception_{type(e).__name__}")

    async def run_stress_test(
        self,
        max_concurrent_operations: int = 50,
        duration_seconds: int = 120,
        integration_service: Optional[LangfuseIntegrationService] = None
    ) -> Dict[str, Any]:
        """Run stress test with maximum concurrent operations."""
        print(f"Running stress test: {max_concurrent_operations} concurrent ops for {duration_seconds}s")

        if integration_service is None:
            integration_service = LangfuseIntegrationService()

        # Stress test with increasing concurrency
        stress_levels = [
            {"operations": 5, "duration": 20},
            {"operations": 15, "duration": 30},
            {"operations": max_concurrent_operations, "duration": 40},
            {"operations": 5, "duration": 30}  # Cool down
        ]

        all_results = []

        for level in stress_levels:
            print(f"  Stress level: {level['operations']} operations for {level['duration']}s")

            # Create scenario for this stress level
            scenario = LoadTestScenario(
                name=f"stress_level_{level['operations']}",
                duration_seconds=level["duration"],
                target_throughput=level["operations"],
                ramp_up_seconds=5,
                ramp_down_seconds=5,
                description=f"Stress test with {level['operations']} concurrent operations"
            )

            # Run this stress level
            level_results = await self.run_high_frequency_load_test(scenario, integration_service)
            all_results.append(level_results)

            # Brief pause between stress levels
            await asyncio.sleep(2)

        # Analyze stress test results
        stress_analysis = self._analyze_stress_test_results(all_results)

        return {
            "stress_test_results": all_results,
            "stress_analysis": stress_analysis,
            "max_concurrent_operations": max_concurrent_operations,
            "total_duration_seconds": duration_seconds,
            "recommendations": self._generate_stress_test_recommendations(stress_analysis)
        }

    def _analyze_stress_test_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stress test results across all levels."""
        if not results:
            return {"error": "No stress test results to analyze"}

        # Extract key metrics from each level
        throughput_values = [r.get("actual_throughput", 0) for r in results]
        error_rates = [r.get("error_rate", 0) for r in results]
        response_times = [r.get("response_times", {}).get("mean", 0) for r in results]
        memory_usage = [r.get("memory_usage_mb", {}).get("max", 0) for r in results]

        # Find breaking point (where performance degrades significantly)
        breaking_point = None
        baseline_throughput = throughput_values[0] if throughput_values else 0

        for i, (throughput, error_rate, response_time) in enumerate(zip(throughput_values, error_rates, response_times)):
            # Detect performance degradation
            throughput_degradation = (baseline_throughput - throughput) / baseline_throughput if baseline_throughput > 0 else 0
            high_error_rate = error_rate > 0.1
            high_response_time = response_time > 5.0  # 5 second threshold

            if throughput_degradation > 0.3 or high_error_rate or high_response_time:
                breaking_point = i
                break

        return {
            "breaking_point_level": breaking_point,
            "max_sustainable_throughput": max(throughput_values[:breaking_point]) if breaking_point else max(throughput_values),
            "throughput_scaling": {
                "baseline": baseline_throughput,
                "peak": max(throughput_values),
                "degradation_threshold": 0.3
            },
            "error_analysis": {
                "max_error_rate": max(error_rates),
                "error_rate_threshold": 0.1
            },
            "performance_analysis": {
                "max_response_time": max(response_times),
                "response_time_threshold": 5.0
            },
            "memory_analysis": {
                "max_memory_usage": max(memory_usage),
                "memory_threshold_mb": 500  # 500MB threshold
            }
        }

    def _generate_stress_test_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on stress test analysis."""
        recommendations = []

        if analysis.get("breaking_point_level") is not None:
            breaking_level = analysis["breaking_point_level"]
            max_sustainable = analysis["max_sustainable_throughput"]

            recommendations.append(
                f"System breaks at stress level {breaking_level}. "
                f"Maximum sustainable throughput: {max_sustainable".1f"} ops/sec"
            )
        else:
            recommendations.append(
                "System maintained performance across all stress levels. "
                "Consider increasing max_concurrent_operations for further testing."
            )

        max_error_rate = analysis["error_analysis"]["max_error_rate"]
        if max_error_rate > 0.1:
            recommendations.append(
                f"High error rate detected ({max_error_rate".2%"}) under stress. "
                "Consider implementing circuit breakers or retry logic."
            )

        max_response_time = analysis["performance_analysis"]["max_response_time"]
        if max_response_time > 5.0:
            recommendations.append(
                f"Response times exceed 5s threshold ({max_response_time".2f"}s) under stress. "
                "Consider performance optimization or load balancing."
            )

        max_memory = analysis["memory_analysis"]["max_memory_usage"]
        if max_memory > 500:
            recommendations.append(
                f"Memory usage exceeds 500MB threshold ({max_memory".1f"}MB) under stress. "
                "Implement memory management and cleanup strategies."
            )

        return recommendations


class TestLangfuseLoadTesting:
    """Pytest test class for Langfuse load testing."""

    @pytest.fixture
    def load_tester(self):
        """Create load tester instance."""
        return HighFrequencyLoadTester()

    @pytest.fixture
    def integration_service(self):
        """Create Langfuse integration service."""
        return LangfuseIntegrationService()

    @pytest.mark.load
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_basic_load_scenario(self, load_tester, integration_service):
        """Test basic load scenario."""
        scenario = LoadTestScenario(
            name="basic_load_test",
            duration_seconds=30,
            target_throughput=5.0,
            description="Basic load test with 5 operations per second"
        )

        results = await load_tester.run_high_frequency_load_test(scenario, integration_service)

        # Validate results
        assert results["scenario_name"] == "basic_load_test"
        assert results["total_operations"] > 0
        assert results["duration_seconds"] >= 25  # Allow some tolerance
        assert results["error_rate"] < 0.2  # Less than 20% error rate

        print(f"Basic load test: {results['actual_throughput']".2f"} ops/sec actual vs {results['target_throughput']} target")

    @pytest.mark.load
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_ramp_up_ramp_down_scenario(self, load_tester, integration_service):
        """Test load scenario with ramp up and ramp down phases."""
        scenario = LoadTestScenario(
            name="ramp_test",
            duration_seconds=60,
            target_throughput=10.0,
            ramp_up_seconds=15,
            ramp_down_seconds=15,
            description="Load test with ramp up/down phases"
        )

        results = await load_tester.run_high_frequency_load_test(scenario, integration_service)

        # Validate ramp behavior
        assert results["throughput_efficiency"] > 0.7  # At least 70% efficiency
        assert results["error_rate"] < 0.15  # Less than 15% error rate

        print(f"Ramp test efficiency: {results['throughput_efficiency']".2%"}")

    @pytest.mark.load
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_high_frequency_scenario(self, load_tester, integration_service):
        """Test high-frequency load scenario."""
        scenario = LoadTestScenario(
            name="high_frequency_test",
            duration_seconds=45,
            target_throughput=20.0,
            ramp_up_seconds=10,
            ramp_down_seconds=10,
            description="High-frequency load test with 20 operations per second"
        )

        results = await load_tester.run_high_frequency_load_test(scenario, integration_service)

        # Validate high-frequency performance
        assert results["actual_throughput"] > 10  # At least 10 ops/sec actual
        assert results["response_times"]["p95"] < 2.0  # 95th percentile under 2 seconds

        print(f"High frequency test: {results['actual_throughput']".2f"} ops/sec, p95: {results['response_times']['p95']".3f"}s")

    @pytest.mark.stress
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_stress_test_scenario(self, load_tester, integration_service):
        """Test stress test with maximum concurrent operations."""
        results = await load_tester.run_stress_test(
            max_concurrent_operations=30,
            duration_seconds=90,
            integration_service=integration_service
        )

        # Validate stress test results
        assert len(results["stress_test_results"]) == 4  # 4 stress levels
        assert "stress_analysis" in results
        assert "recommendations" in results
        assert len(results["recommendations"]) > 0

        analysis = results["stress_analysis"]
        if analysis.get("breaking_point_level") is not None:
            print(f"Breaking point detected at level {analysis['breaking_point_level']}")
        else:
            print("No breaking point detected - system handled all stress levels")

        print(f"Max sustainable throughput: {analysis['max_sustainable_throughput']".1f"} ops/sec")

    @pytest.mark.load
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load_scenario(self, load_tester, integration_service):
        """Test sustained load over extended period."""
        scenario = LoadTestScenario(
            name="sustained_load_test",
            duration_seconds=120,
            target_throughput=8.0,
            ramp_up_seconds=20,
            ramp_down_seconds=20,
            description="Sustained load test over 2 minutes"
        )

        results = await load_tester.run_high_frequency_load_test(scenario, integration_service)

        # Validate sustained performance
        assert results["duration_seconds"] >= 100  # At least 100 seconds of data
        assert results["throughput_efficiency"] > 0.8  # At least 80% efficiency over time
        assert results["memory_usage_mb"]["max"] < 300  # Memory usage under control

        print(f"Sustained load test: {results['actual_throughput']".2f"} ops/sec over {results['duration_seconds']".1f"}s")

    @pytest.mark.load
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, load_tester, integration_service):
        """Test for memory leaks under sustained load."""
        # Run sustained load test
        scenario = LoadTestScenario(
            name="memory_leak_test",
            duration_seconds=60,
            target_throughput=5.0,
            description="Memory leak detection test"
        )

        results = await load_tester.run_high_frequency_load_test(scenario, integration_service)

        # Analyze memory usage pattern
        memory_samples = load_tester.metrics_collector.memory_samples

        if len(memory_samples) > 10:
            # Check for memory growth trend
            first_quarter = statistics.mean(list(memory_samples)[:len(memory_samples)//4])
            last_quarter = statistics.mean(list(memory_samples)[-len(memory_samples)//4:])

            memory_growth = last_quarter - first_quarter
            memory_growth_rate = memory_growth / results["duration_seconds"]

            # Memory should not grow more than 50MB over the test duration
            assert memory_growth < 50, f"Potential memory leak detected: {memory_growth".1f"}MB growth"
            assert memory_growth_rate < 1.0, f"High memory growth rate: {memory_growth_rate".2f"}MB/s"

            print(f"Memory growth: {memory_growth".1f"}MB over {results['duration_seconds']}s")
            print(f"Memory growth rate: {memory_growth_rate".3f"}MB/s")

    @pytest.mark.load
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_comprehensive_load_suite(self, load_tester, integration_service):
        """Run comprehensive load testing suite."""
        print("Running comprehensive load testing suite...")

        scenarios = [
            LoadTestScenario("light_load", 30, 3.0, description="Light load baseline"),
            LoadTestScenario("medium_load", 45, 8.0, description="Medium load test"),
            LoadTestScenario("heavy_load", 60, 15.0, description="Heavy load test"),
            LoadTestScenario("sustained_load", 90, 6.0, description="Sustained load test")
        ]

        suite_results = []

        for scenario in scenarios:
            results = await load_tester.run_high_frequency_load_test(scenario, integration_service)
            suite_results.append(results)

            # Brief pause between scenarios
            await asyncio.sleep(5)

        # Analyze suite results
        suite_analysis = {
            "total_scenarios": len(suite_results),
            "total_operations": sum(r["total_operations"] for r in suite_results),
            "total_duration": sum(r["duration_seconds"] for r in suite_results),
            "average_throughput": statistics.mean([r["actual_throughput"] for r in suite_results]),
            "average_error_rate": statistics.mean([r["error_rate"] for r in suite_results]),
            "max_memory_usage": max(r["memory_usage_mb"]["max"] for r in suite_results),
            "performance_trend": self._analyze_performance_trend(suite_results)
        }

        # Validate suite results
        assert suite_analysis["average_error_rate"] < 0.1  # Less than 10% average error rate
        assert suite_analysis["max_memory_usage"] < 400  # Memory usage under control

        print(f"Load suite: {suite_analysis['average_throughput']".2f"} avg ops/sec, {suite_analysis['average_error_rate']".2%"} error rate")

        return {"suite_results": suite_results, "suite_analysis": suite_analysis}

    def _analyze_performance_trend(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends across test scenarios."""
        throughputs = [r["actual_throughput"] for r in results]
        error_rates = [r["error_rate"] for r in results]

        # Simple trend analysis
        if len(throughputs) >= 2:
            throughput_trend = throughputs[-1] - throughputs[0]
            error_trend = error_rates[-1] - error_rates[0]
        else:
            throughput_trend = error_trend = 0

        return {
            "throughput_trend": throughput_trend,
            "error_rate_trend": error_trend,
            "performance_stable": abs(throughput_trend) < 1.0 and abs(error_trend) < 0.05,
            "performance_improving": throughput_trend > 0 and error_trend < 0,
            "performance_degrading": throughput_trend < 0 or error_trend > 0
        }


if __name__ == "__main__":
    # Allow running load tests directly
    async def run_load_test_demo():
        """Demo of load testing capabilities."""
        tester = HighFrequencyLoadTester()
        service = LangfuseIntegrationService()

        print("=== Langfuse Load Testing Demo ===")

        # Run a quick load test
        scenario = LoadTestScenario(
            name="demo_load_test",
            duration_seconds=20,
            target_throughput=5.0,
            description="Demo load test"
        )

        results = await tester.run_high_frequency_load_test(scenario, service)

        print("
Load test results:")
        print(f"  Actual throughput: {results['actual_throughput']".2f"} ops/sec")
        print(f"  Error rate: {results['error_rate']".2%"}")
        print(f"  Avg response time: {results['response_times']['mean']".3f"}s")
        print(f"  Max memory usage: {results['memory_usage_mb']['max']".1f"} MB")

        return results

    # Run demo
    results = asyncio.run(run_load_test_demo())
    print("\n=== Load Testing Demo Complete ===")