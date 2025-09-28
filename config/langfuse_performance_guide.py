"""Performance optimization guide and benchmarks for Langfuse integration."""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import psutil
import threading

logger = logging.getLogger(__name__)


@dataclass
class PerformanceThresholds:
    """Performance thresholds for different system states."""

    # CPU thresholds (percentage)
    cpu_normal: float = 60.0
    cpu_warning: float = 80.0
    cpu_critical: float = 95.0

    # Memory thresholds (percentage)
    memory_normal: float = 70.0
    memory_warning: float = 85.0
    memory_critical: float = 95.0

    # Latency thresholds (milliseconds)
    latency_normal: float = 100.0
    latency_warning: float = 500.0
    latency_critical: float = 2000.0

    # Throughput thresholds (traces per second)
    throughput_normal: float = 50.0
    throughput_warning: float = 20.0
    throughput_critical: float = 5.0

    # Queue size thresholds
    queue_normal: int = 100
    queue_warning: int = 500
    queue_critical: int = 1000


@dataclass
class OptimizationRecommendation:
    """A performance optimization recommendation."""

    category: str  # "sampling", "compression", "async", "buffering", "monitoring"
    priority: str  # "low", "medium", "high", "critical"
    title: str
    description: str
    impact: str  # "low", "medium", "high"
    effort: str  # "low", "medium", "high"
    config_changes: Dict[str, Any] = field(default_factory=dict)
    enabled_by_default: bool = True


class PerformanceAnalyzer:
    """Analyzes system performance and provides optimization recommendations."""

    def __init__(self, thresholds: Optional[PerformanceThresholds] = None):
        """Initialize the performance analyzer.

        Args:
            thresholds: Custom performance thresholds
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self.baseline_metrics = {}
        self.optimization_history = []

    def analyze_current_performance(self) -> Dict[str, Any]:
        """Analyze current system performance and return detailed metrics.

        Returns:
            Dictionary containing current performance analysis
        """
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Process-specific metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()

        # Disk and I/O metrics
        disk = psutil.disk_usage("/")
        io_counters = process.io_counters()

        # Network metrics (if available)
        try:
            net_io = psutil.net_io_counters()
            network_stats = {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            }
        except:
            network_stats = {}

        current_metrics = {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk.percent,
                "memory_available_mb": memory.available / 1024 / 1024,
                "memory_used_mb": memory.used / 1024 / 1024,
            },
            "process": {
                "cpu_percent": process_cpu,
                "memory_rss_mb": process_memory.rss / 1024 / 1024,
                "memory_vms_mb": process_memory.vms / 1024 / 1024,
                "io_read_bytes": io_counters.read_bytes,
                "io_write_bytes": io_counters.write_bytes,
                "thread_count": process.num_threads(),
            },
            "network": network_stats,
        }

        return current_metrics

    def get_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate a performance score (0-100) based on current metrics.

        Args:
            metrics: Performance metrics from analyze_current_performance

        Returns:
            Performance score where 100 is best
        """
        system = metrics.get("system", {})
        process = metrics.get("process", {})

        # Calculate individual scores (0-100, higher is better)
        cpu_score = max(0, 100 - (system.get("cpu_percent", 0) / self.thresholds.cpu_critical * 100))
        memory_score = max(0, 100 - (system.get("memory_percent", 0) / self.thresholds.memory_critical * 100))
        process_cpu_score = max(0, 100 - (process.get("cpu_percent", 0) / 50 * 100))  # Normalize to 50% max

        # Weighted average
        overall_score = (cpu_score * 0.4 + memory_score * 0.4 + process_cpu_score * 0.2)

        return min(100, max(0, overall_score))

    def get_system_state(self, metrics: Dict[str, Any]) -> str:
        """Determine the current system state based on performance metrics.

        Args:
            metrics: Performance metrics from analyze_current_performance

        Returns:
            System state: "optimal", "good", "warning", "critical"
        """
        system = metrics.get("system", {})
        process = metrics.get("process", {})

        cpu_percent = system.get("cpu_percent", 0)
        memory_percent = system.get("memory_percent", 0)
        process_cpu = process.get("cpu_percent", 0)

        # Check critical thresholds first
        if (cpu_percent >= self.thresholds.cpu_critical or
            memory_percent >= self.thresholds.memory_critical or
            process_cpu >= 80):
            return "critical"

        # Check warning thresholds
        if (cpu_percent >= self.thresholds.cpu_warning or
            memory_percent >= self.thresholds.memory_warning or
            process_cpu >= 60):
            return "warning"

        # Check if we're in good range
        if (cpu_percent <= self.thresholds.cpu_normal and
            memory_percent <= self.thresholds.memory_normal and
            process_cpu <= 30):
            return "good"

        return "optimal"

    def generate_optimization_recommendations(
        self,
        current_metrics: Dict[str, Any],
        langfuse_stats: Optional[Dict[str, Any]] = None,
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on current performance.

        Args:
            current_metrics: Current system performance metrics
            langfuse_stats: Optional Langfuse-specific statistics

        Returns:
            List of optimization recommendations
        """
        recommendations = []
        system_state = self.get_system_state(current_metrics)

        # Base recommendations by system state
        if system_state == "critical":
            recommendations.extend(self._get_critical_optimizations())
        elif system_state == "warning":
            recommendations.extend(self._get_warning_optimizations())
        elif system_state == "good":
            recommendations.extend(self._get_good_optimizations())
        else:
            recommendations.extend(self._get_optimal_optimizations())

        # Langfuse-specific recommendations
        if langfuse_stats:
            recommendations.extend(self._get_langfuse_optimizations(langfuse_stats))

        # Sort by priority and impact
        priority_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        recommendations.sort(
            key=lambda r: (priority_order.get(r.priority, 0), r.impact),
            reverse=True
        )

        return recommendations

    def _get_critical_optimizations(self) -> List[OptimizationRecommendation]:
        """Get optimizations for critical system state."""
        return [
            OptimizationRecommendation(
                category="sampling",
                priority="critical",
                title="Enable Aggressive Sampling",
                description="System is critically overloaded. Enable aggressive sampling to reduce tracing overhead.",
                impact="high",
                effort="low",
                config_changes={
                    "LANGFUSE_SAMPLE_RATE": 0.1,
                    "LANGFUSE_ENABLE_SAMPLING": True,
                },
            ),
            OptimizationRecommendation(
                category="async",
                priority="critical",
                title="Enable Async Processing",
                description="Switch to fully asynchronous processing to prevent blocking operations.",
                impact="high",
                effort="low",
                config_changes={
                    "LANGFUSE_ASYNC_PROCESSING": True,
                    "LANGFUSE_BATCH_SIZE": 10,
                },
            ),
            OptimizationRecommendation(
                category="buffering",
                priority="critical",
                title="Enable Local Buffering",
                description="Buffer traces locally to reduce immediate system load.",
                impact="medium",
                effort="low",
                config_changes={
                    "LANGFUSE_BUFFERING_ENABLED": True,
                    "LANGFUSE_MAX_BUFFER_SIZE": 5000,
                },
            ),
        ]

    def _get_warning_optimizations(self) -> List[OptimizationRecommendation]:
        """Get optimizations for warning system state."""
        return [
            OptimizationRecommendation(
                category="sampling",
                priority="high",
                title="Reduce Sampling Rate",
                description="System showing warning signs. Reduce sampling rate to decrease overhead.",
                impact="medium",
                effort="low",
                config_changes={
                    "LANGFUSE_SAMPLE_RATE": 0.5,
                },
            ),
            OptimizationRecommendation(
                category="compression",
                priority="medium",
                title="Enable Compression",
                description="Enable trace data compression to reduce memory and network usage.",
                impact="medium",
                effort="low",
                config_changes={
                    "LANGFUSE_COMPRESSION_ENABLED": True,
                    "LANGFUSE_COMPRESSION_THRESHOLD": 2048,
                },
            ),
        ]

    def _get_good_optimizations(self) -> List[OptimizationRecommendation]:
        """Get optimizations for good system state."""
        return [
            OptimizationRecommendation(
                category="monitoring",
                priority="medium",
                title="Enable Detailed Monitoring",
                description="System performing well. Enable detailed performance monitoring.",
                impact="low",
                effort="low",
                config_changes={
                    "LANGFUSE_DETAILED_MONITORING": True,
                    "LANGFUSE_PERFORMANCE_MONITORING": True,
                },
            ),
            OptimizationRecommendation(
                category="sampling",
                priority="low",
                title="Optimize Sampling Strategy",
                description="Fine-tune sampling strategy based on trace importance.",
                impact="low",
                effort="medium",
                config_changes={
                    "LANGFUSE_SAMPLING_STRATEGY": "priority_based",
                },
            ),
        ]

    def _get_optimal_optimizations(self) -> List[OptimizationRecommendation]:
        """Get optimizations for optimal system state."""
        return [
            OptimizationRecommendation(
                category="monitoring",
                priority="low",
                title="Full Tracing Enabled",
                description="System is optimal. Full tracing can be safely enabled.",
                impact="low",
                effort="low",
                config_changes={
                    "LANGFUSE_SAMPLE_RATE": 1.0,
                    "LANGFUSE_DETAILED_TRACING": True,
                },
            ),
        ]

    def _get_langfuse_optimizations(self, langfuse_stats: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Get Langfuse-specific optimizations."""
        recommendations = []

        # Check error rates
        error_rate = langfuse_stats.get("error_rate", 0)
        if error_rate > 0.1:
            recommendations.append(
                OptimizationRecommendation(
                    category="monitoring",
                    priority="high",
                    title="High Error Rate Detected",
                    description=f"Langfuse error rate is {error_rate:.1%".1%"}. Check connectivity and configuration.",
                    impact="high",
                    effort="low",
                    config_changes={
                        "LANGFUSE_MAX_RETRIES": 5,
                        "LANGFUSE_RETRY_DELAY": 2.0,
                    },
                )
            )

        # Check latency
        avg_latency = langfuse_stats.get("average_latency_ms", 0)
        if avg_latency > self.thresholds.latency_warning:
            recommendations.append(
                OptimizationRecommendation(
                    category="async",
                    priority="medium",
                    title="High Latency Detected",
                    description=f"Average latency is {avg_latency".1f"}ms. Consider async processing.",
                    impact="medium",
                    effort="medium",
                    config_changes={
                        "LANGFUSE_ASYNC_PROCESSING": True,
                        "LANGFUSE_FLUSH_INTERVAL": 2.0,
                    },
                )
            )

        # Check throughput
        throughput = langfuse_stats.get("throughput_traces_per_sec", 0)
        if throughput < self.thresholds.throughput_warning:
            recommendations.append(
                OptimizationRecommendation(
                    category="buffering",
                    priority="medium",
                    title="Low Throughput Detected",
                    description=f"Throughput is {throughput".1f"} traces/sec. Enable batching.",
                    impact="medium",
                    effort="low",
                    config_changes={
                        "LANGFUSE_BATCH_SIZE": 25,
                        "LANGFUSE_BUFFERING_ENABLED": True,
                    },
                )
            )

        return recommendations

    def create_performance_report(
        self,
        current_metrics: Dict[str, Any],
        langfuse_stats: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[OptimizationRecommendation]] = None,
    ) -> Dict[str, Any]:
        """Create a comprehensive performance report.

        Args:
            current_metrics: Current system performance metrics
            langfuse_stats: Optional Langfuse-specific statistics
            recommendations: Optional list of recommendations

        Returns:
            Comprehensive performance report
        """
        if recommendations is None:
            recommendations = self.generate_optimization_recommendations(current_metrics, langfuse_stats)

        performance_score = self.get_performance_score(current_metrics)
        system_state = self.get_system_state(current_metrics)

        # Count recommendations by priority
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for rec in recommendations:
            priority_counts[rec.priority] += 1

        report = {
            "timestamp": datetime.now().isoformat(),
            "performance_score": round(performance_score, 2),
            "system_state": system_state,
            "current_metrics": current_metrics,
            "recommendations": {
                "total": len(recommendations),
                "by_priority": priority_counts,
                "details": [
                    {
                        "category": rec.category,
                        "priority": rec.priority,
                        "title": rec.title,
                        "description": rec.description,
                        "impact": rec.impact,
                        "effort": rec.effort,
                        "config_changes": rec.config_changes,
                    }
                    for rec in recommendations[:10]  # Limit to top 10
                ],
            },
            "langfuse_stats": langfuse_stats or {},
            "thresholds": {
                "cpu_normal": self.thresholds.cpu_normal,
                "cpu_warning": self.thresholds.cpu_warning,
                "cpu_critical": self.thresholds.cpu_critical,
                "memory_normal": self.thresholds.memory_normal,
                "memory_warning": self.thresholds.memory_warning,
                "memory_critical": self.thresholds.memory_critical,
                "latency_normal": self.thresholds.latency_normal,
                "latency_warning": self.thresholds.latency_warning,
                "latency_critical": self.thresholds.latency_critical,
            },
        }

        return report

    def save_performance_report(self, report: Dict[str, Any], filepath: str) -> bool:
        """Save performance report to file.

        Args:
            report: Performance report to save
            filepath: File path to save to

        Returns:
            True if saved successfully
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Performance report saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save performance report: {e}")
            return False


class PerformanceBenchmarkRunner:
    """Runs performance benchmarks and tracks results."""

    def __init__(self):
        """Initialize the benchmark runner."""
        self.benchmark_results = []
        self.baselines = {}

    def run_comprehensive_benchmark(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Run a comprehensive performance benchmark.

        Args:
            duration_minutes: Duration to run benchmarks

        Returns:
            Comprehensive benchmark results
        """
        logger.info(f"Starting comprehensive performance benchmark for {duration_minutes} minutes")

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        benchmark_data = {
            "start_time": start_time.isoformat(),
            "duration_minutes": duration_minutes,
            "samples": [],
            "summary": {},
        }

        # Collect samples every 10 seconds
        next_sample = start_time + timedelta(seconds=10)

        while datetime.now() < end_time:
            if datetime.now() >= next_sample:
                sample = self._collect_benchmark_sample()
                benchmark_data["samples"].append(sample)
                next_sample += timedelta(seconds=10)

            time.sleep(1)  # Check every second

        # Calculate summary statistics
        benchmark_data["summary"] = self._calculate_benchmark_summary(benchmark_data["samples"])

        logger.info("Comprehensive benchmark completed")
        self.benchmark_results.append(benchmark_data)

        return benchmark_data

    def _collect_benchmark_sample(self) -> Dict[str, Any]:
        """Collect a single benchmark sample."""
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Langfuse-specific metrics (if available)
        langfuse_metrics = self._get_langfuse_metrics()

        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_mb": memory.used / 1024 / 1024,
            },
            "langfuse": langfuse_metrics,
        }

    def _get_langfuse_metrics(self) -> Dict[str, Any]:
        """Get Langfuse-specific metrics for benchmarking."""
        metrics = {}

        try:
            # Try to get metrics from various components
            from .langfuse_sampling import get_langfuse_sampler
            sampler = get_langfuse_sampler()
            if sampler:
                metrics["sampling"] = sampler.get_sampling_stats()

            from .langfuse_async_processor import get_async_trace_processor
            processor = get_async_trace_processor()
            if processor:
                metrics["async_processing"] = processor.get_processing_stats()

            from .langfuse_performance_monitor import get_langfuse_performance_monitor
            monitor = get_langfuse_performance_monitor()
            if monitor:
                metrics["performance_monitor"] = monitor.get_performance_summary()

        except ImportError:
            pass

        return metrics

    def _calculate_benchmark_summary(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from benchmark samples."""
        if not samples:
            return {}

        # Extract metrics
        cpu_values = [s["system"]["cpu_percent"] for s in samples]
        memory_values = [s["system"]["memory_percent"] for s in samples]

        summary = {
            "total_samples": len(samples),
            "duration_seconds": (len(samples) - 1) * 10,  # 10-second intervals
            "system": {
                "cpu_percent": {
                    "mean": sum(cpu_values) / len(cpu_values),
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "p95": sorted(cpu_values)[int(len(cpu_values) * 0.95)] if cpu_values else 0,
                },
                "memory_percent": {
                    "mean": sum(memory_values) / len(memory_values),
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "p95": sorted(memory_values)[int(len(memory_values) * 0.95)] if memory_values else 0,
                },
            },
        }

        return summary


# Global instances
_performance_analyzer: Optional[PerformanceAnalyzer] = None
_benchmark_runner: Optional[PerformanceBenchmarkRunner] = None


def get_performance_analyzer() -> Optional[PerformanceAnalyzer]:
    """Get the global performance analyzer instance."""
    return _performance_analyzer


def initialize_performance_analyzer(**kwargs) -> PerformanceAnalyzer:
    """Initialize the global performance analyzer.

    Args:
        **kwargs: Configuration parameters for PerformanceAnalyzer

    Returns:
        Initialized PerformanceAnalyzer instance
    """
    global _performance_analyzer
    _performance_analyzer = PerformanceAnalyzer(**kwargs)
    return _performance_analyzer


def get_benchmark_runner() -> Optional[PerformanceBenchmarkRunner]:
    """Get the global benchmark runner instance."""
    return _benchmark_runner


def initialize_benchmark_runner() -> PerformanceBenchmarkRunner:
    """Initialize the global benchmark runner.

    Returns:
        Initialized PerformanceBenchmarkRunner instance
    """
    global _benchmark_runner
    _benchmark_runner = PerformanceBenchmarkRunner()
    return _benchmark_runner


def run_quick_performance_check() -> Dict[str, Any]:
    """Run a quick performance check and return recommendations.

    Returns:
        Quick performance analysis with recommendations
    """
    analyzer = get_performance_analyzer() or initialize_performance_analyzer()
    current_metrics = analyzer.analyze_current_performance()
    recommendations = analyzer.generate_optimization_recommendations(current_metrics)

    return {
        "performance_score": analyzer.get_performance_score(current_metrics),
        "system_state": analyzer.get_system_state(current_metrics),
        "recommendations_count": len(recommendations),
        "top_recommendations": [
            {
                "title": rec.title,
                "priority": rec.priority,
                "category": rec.category,
            }
            for rec in recommendations[:3]
        ],
    }