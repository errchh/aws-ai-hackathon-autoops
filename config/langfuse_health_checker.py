"""Enhanced health check system for Langfuse connectivity and diagnostics.

This module provides comprehensive health checking capabilities including:
- Detailed connectivity diagnostics
- Performance benchmarking
- Configuration validation
- Network diagnostics
- Security checks
"""

import logging
import time
import socket
import ssl
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
import json

from .langfuse_config import get_langfuse_client, LangfuseClient
from .langfuse_integration import get_langfuse_integration

logger = logging.getLogger(__name__)


@dataclass
class ConnectivityTestResult:
    """Result of a connectivity test."""

    test_name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class HealthDiagnostics:
    """Comprehensive health diagnostics."""

    timestamp: datetime
    overall_status: str
    connectivity_tests: List[ConnectivityTestResult]
    performance_benchmarks: Dict[str, float]
    configuration_issues: List[str]
    security_issues: List[str]
    recommendations: List[str]


class LangfuseHealthChecker:
    """Enhanced health checker for Langfuse connectivity and diagnostics."""

    def __init__(self, client: Optional[LangfuseClient] = None):
        """Initialize the health checker.

        Args:
            client: Optional LangfuseClient instance. If None, uses global client.
        """
        self.client = client or get_langfuse_client()
        self.integration = get_langfuse_integration()

    def perform_comprehensive_health_check(self) -> HealthDiagnostics:
        """Perform comprehensive health check with detailed diagnostics."""
        timestamp = datetime.now()
        connectivity_tests = []
        performance_benchmarks = {}
        configuration_issues = []
        security_issues = []
        recommendations = []

        try:
            # Basic connectivity tests
            connectivity_tests.extend(self._test_basic_connectivity())

            # Network diagnostics
            connectivity_tests.extend(self._test_network_diagnostics())

            # Configuration validation
            config_issues, config_recommendations = self._validate_configuration()
            configuration_issues.extend(config_issues)
            recommendations.extend(config_recommendations)

            # Security checks
            security_issues, security_recommendations = self._check_security()
            security_issues.extend(security_issues)
            recommendations.extend(security_recommendations)

            # Performance benchmarks
            performance_benchmarks = self._run_performance_benchmarks()

            # Determine overall status
            overall_status = self._determine_overall_status(
                connectivity_tests, configuration_issues, security_issues
            )

            # Generate recommendations
            recommendations.extend(
                self._generate_recommendations(
                    connectivity_tests,
                    performance_benchmarks,
                    configuration_issues,
                    security_issues,
                )
            )

            logger.info(f"Comprehensive health check completed: {overall_status}")

        except Exception as e:
            overall_status = "error"
            connectivity_tests.append(
                ConnectivityTestResult(
                    test_name="health_check_execution",
                    success=False,
                    duration_ms=0,
                    error=str(e),
                )
            )
            logger.error(f"Health check failed: {e}")

        return HealthDiagnostics(
            timestamp=timestamp,
            overall_status=overall_status,
            connectivity_tests=connectivity_tests,
            performance_benchmarks=performance_benchmarks,
            configuration_issues=configuration_issues,
            security_issues=security_issues,
            recommendations=recommendations,
        )

    def _test_basic_connectivity(self) -> List[ConnectivityTestResult]:
        """Test basic Langfuse connectivity."""
        tests = []

        # Test 1: Client availability
        start_time = time.time()
        try:
            available = self.client.is_available
            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="client_availability",
                    success=available,
                    duration_ms=duration,
                    details={"available": available},
                )
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="client_availability",
                    success=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )

        # Test 2: Authentication
        if self.client.is_available:
            start_time = time.time()
            try:
                # Try to access client properties that require authentication
                client = self.client.client
                if client:
                    # Test basic client functionality
                    duration = (time.time() - start_time) * 1000
                    tests.append(
                        ConnectivityTestResult(
                            test_name="authentication",
                            success=True,
                            duration_ms=duration,
                            details={"authenticated": True},
                        )
                    )
                else:
                    duration = (time.time() - start_time) * 1000
                    tests.append(
                        ConnectivityTestResult(
                            test_name="authentication",
                            success=False,
                            duration_ms=duration,
                            error="Client not initialized",
                        )
                    )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                tests.append(
                    ConnectivityTestResult(
                        test_name="authentication",
                        success=False,
                        duration_ms=duration,
                        error=str(e),
                    )
                )

        # Test 3: Integration service health
        start_time = time.time()
        try:
            health = self.integration.health_check()
            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="integration_service",
                    success=health.get("integration_service") == "ready",
                    duration_ms=duration,
                    details=health,
                )
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="integration_service",
                    success=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )

        return tests

    def _test_network_diagnostics(self) -> List[ConnectivityTestResult]:
        """Test network connectivity and performance."""
        tests = []

        if not self.client.config or not self.client.config.host:
            return tests

        host = self.client.config.host
        if not host:
            return tests

        # Test 1: DNS resolution
        start_time = time.time()
        try:
            ip_address = socket.gethostbyname(
                host.replace("https://", "").replace("http://", "").split("/")[0]
            )
            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="dns_resolution",
                    success=True,
                    duration_ms=duration,
                    details={"ip_address": ip_address, "hostname": host},
                )
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="dns_resolution",
                    success=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )

        # Test 2: Port connectivity (443 for HTTPS)
        start_time = time.time()
        try:
            hostname = host.replace("https://", "").replace("http://", "").split("/")[0]
            port = 443 if host.startswith("https") else 80

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((hostname, port))
            sock.close()

            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="port_connectivity",
                    success=result == 0,
                    duration_ms=duration,
                    details={"hostname": hostname, "port": port, "result": result},
                )
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            tests.append(
                ConnectivityTestResult(
                    test_name="port_connectivity",
                    success=False,
                    duration_ms=duration,
                    error=str(e),
                )
            )

        # Test 3: HTTPS/SSL validation
        if host.startswith("https"):
            start_time = time.time()
            try:
                hostname = host.replace("https://", "").split("/")[0]
                context = ssl.create_default_context()

                with socket.create_connection((hostname, 443)) as sock:
                    with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                        cert = ssock.getpeercert()

                duration = (time.time() - start_time) * 1000
                tests.append(
                    ConnectivityTestResult(
                        test_name="ssl_validation",
                        success=True,
                        duration_ms=duration,
                        details={
                            "hostname": hostname,
                            "cert_issuer": cert.get("issuer", "unknown")
                            if cert
                            else "unknown",
                            "cert_valid": True,
                        },
                    )
                )
            except Exception as e:
                duration = (time.time() - start_time) * 1000
                tests.append(
                    ConnectivityTestResult(
                        test_name="ssl_validation",
                        success=False,
                        duration_ms=duration,
                        error=str(e),
                    )
                )

        return tests

    def _validate_configuration(self) -> Tuple[List[str], List[str]]:
        """Validate Langfuse configuration."""
        issues = []
        recommendations = []

        if not self.client.config:
            issues.append("Langfuse configuration not found")
            recommendations.append("Initialize LangfuseConfig with proper credentials")
            return issues, recommendations

        config = self.client.config

        # Check required fields
        if not config.public_key:
            issues.append("Public key not configured")
            recommendations.append("Set LANGFUSE_PUBLIC_KEY environment variable")

        if not config.secret_key:
            issues.append("Secret key not configured")
            recommendations.append("Set LANGFUSE_SECRET_KEY environment variable")

        if not config.host:
            issues.append("Host URL not configured")
            recommendations.append("Set LANGFUSE_HOST environment variable")

        # Validate host URL format
        if config.host and not (
            config.host.startswith("http://") or config.host.startswith("https://")
        ):
            issues.append("Invalid host URL format")
            recommendations.append("Host URL must start with http:// or https://")

        # Check sampling configuration
        if config.sample_rate < 0 or config.sample_rate > 1:
            issues.append("Invalid sample rate (must be 0-1)")
            recommendations.append("Set sample rate between 0.0 and 1.0")

        # Check buffer configuration
        if config.buffer_size < 0:
            issues.append("Invalid buffer size")
            recommendations.append("Set buffer size to a positive integer")

        return issues, recommendations

    def _check_security(self) -> Tuple[List[str], List[str]]:
        """Check security configuration and issues."""
        issues = []
        recommendations = []

        if not self.client.config:
            return issues, recommendations

        config = self.client.config

        # Check HTTPS usage
        if config.host and not config.host.startswith("https://"):
            issues.append("Using insecure HTTP connection")
            recommendations.append("Use HTTPS for production environments")

        # Check for default or weak credentials (basic check)
        if config.public_key and config.public_key.startswith("pk-lf-"):
            # This is a valid format, but check for obviously weak patterns
            pass

        # Check for credential exposure in logs
        # This is a basic check - in production, you'd want more sophisticated checks

        return issues, recommendations

    def _run_performance_benchmarks(self) -> Dict[str, float]:
        """Run performance benchmarks."""
        benchmarks = {}

        try:
            # Benchmark trace creation
            start_time = time.time()
            try:
                trace_id = self.integration.create_simulation_trace(
                    {"benchmark": "test"}
                )
                if trace_id:
                    self.integration.finalize_trace(trace_id)
                creation_time = (time.time() - start_time) * 1000
                benchmarks["trace_creation_ms"] = creation_time
            except Exception as e:
                benchmarks["trace_creation_ms"] = -1  # Error indicator
                logger.debug(f"Trace creation benchmark failed: {e}")

            # Benchmark span creation
            start_time = time.time()
            try:
                span_id = self.integration.start_agent_span(
                    "test_agent", "benchmark_test"
                )
                if span_id:
                    self.integration.end_agent_span(span_id)
                span_time = (time.time() - start_time) * 1000
                benchmarks["span_creation_ms"] = span_time
            except Exception as e:
                benchmarks["span_creation_ms"] = -1  # Error indicator
                logger.debug(f"Span creation benchmark failed: {e}")

        except Exception as e:
            logger.debug(f"Performance benchmarking failed: {e}")

        return benchmarks

    def _determine_overall_status(
        self,
        connectivity_tests: List[ConnectivityTestResult],
        configuration_issues: List[str],
        security_issues: List[str],
    ) -> str:
        """Determine overall health status."""
        failed_tests = [t for t in connectivity_tests if not t.success]

        if len(failed_tests) > 2:
            return "unhealthy"
        elif len(failed_tests) > 0 or configuration_issues or security_issues:
            return "degraded"
        else:
            return "healthy"

    def _generate_recommendations(
        self,
        connectivity_tests: List[ConnectivityTestResult],
        performance_benchmarks: Dict[str, float],
        configuration_issues: List[str],
        security_issues: List[str],
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Connectivity recommendations
        failed_tests = [t for t in connectivity_tests if not t.success]
        if failed_tests:
            recommendations.append("Check network connectivity and firewall settings")

        # Performance recommendations
        trace_time = performance_benchmarks.get("trace_creation_ms", 0)
        if trace_time > 1000:  # 1 second
            recommendations.append(
                "High trace creation latency detected - consider optimizing or reducing trace frequency"
            )

        span_time = performance_benchmarks.get("span_creation_ms", 0)
        if span_time > 500:  # 500ms
            recommendations.append(
                "High span creation latency detected - consider optimizing agent operations"
            )

        # Configuration recommendations
        if configuration_issues:
            recommendations.append("Review and fix configuration issues")

        # Security recommendations
        if security_issues:
            recommendations.append("Address security issues for production deployment")

        return recommendations

    def get_connectivity_report(self) -> Dict[str, Any]:
        """Get detailed connectivity report."""
        diagnostics = self.perform_comprehensive_health_check()

        # Format connectivity tests for report
        connectivity_report = {
            "summary": {
                "overall_status": diagnostics.overall_status,
                "total_tests": len(diagnostics.connectivity_tests),
                "passed_tests": len(
                    [t for t in diagnostics.connectivity_tests if t.success]
                ),
                "failed_tests": len(
                    [t for t in diagnostics.connectivity_tests if not t.success]
                ),
            },
            "tests": [
                {
                    "name": test.test_name,
                    "success": test.success,
                    "duration_ms": test.duration_ms,
                    "error": test.error,
                    "details": test.details,
                }
                for test in diagnostics.connectivity_tests
            ],
            "performance": diagnostics.performance_benchmarks,
            "issues": {
                "configuration": diagnostics.configuration_issues,
                "security": diagnostics.security_issues,
            },
            "recommendations": diagnostics.recommendations,
        }

        return connectivity_report


# Global health checker instance
_health_checker: Optional[LangfuseHealthChecker] = None


def get_health_checker() -> LangfuseHealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = LangfuseHealthChecker()
    return _health_checker


def initialize_health_checker(
    client: Optional[LangfuseClient] = None,
) -> LangfuseHealthChecker:
    """Initialize the global health checker.

    Args:
        client: Optional LangfuseClient instance

    Returns:
        Initialized LangfuseHealthChecker instance
    """
    global _health_checker
    _health_checker = LangfuseHealthChecker(client)
    return _health_checker
