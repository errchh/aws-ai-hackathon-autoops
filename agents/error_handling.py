"""
Error handling and system resilience for the retail optimization multi-agent system.

This module provides comprehensive error handling, retry mechanisms, circuit breakers,
and graceful degradation capabilities to ensure system resilience.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from functools import wraps
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(Enum):
    """Agent operational status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class ErrorContext:
    """Context information for error tracking and resolution."""
    error_id: str
    timestamp: datetime
    component: str
    error_type: str
    severity: ErrorSeverity
    message: str
    stack_trace: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    resolved: bool = False


@dataclass
class CircuitBreakerState:
    """State tracking for circuit breaker pattern."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half_open -> closed transition


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class SystemResilienceManager:
    """Central manager for system resilience and error handling."""
    
    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.agent_statuses: Dict[str, AgentStatus] = {}
        self.fallback_strategies: Dict[str, Callable] = {}
        self.monitoring_callbacks: List[Callable] = []
        
    def register_agent(self, agent_id: str) -> None:
        """Register an agent for monitoring."""
        self.agent_statuses[agent_id] = AgentStatus.HEALTHY
        self.circuit_breakers[agent_id] = CircuitBreakerState()
        logger.info(f"Registered agent {agent_id} for monitoring")
    
    def register_fallback_strategy(self, component: str, strategy: Callable) -> None:
        """Register a fallback strategy for a component."""
        self.fallback_strategies[component] = strategy
        logger.info(f"Registered fallback strategy for {component}")
    
    def add_monitoring_callback(self, callback: Callable) -> None:
        """Add a callback for system monitoring events."""
        self.monitoring_callbacks.append(callback)
    
    def log_error(self, error_context: ErrorContext) -> None:
        """Log an error and update system state."""
        self.error_history.append(error_context)
        
        # Update circuit breaker state
        if error_context.component in self.circuit_breakers:
            cb = self.circuit_breakers[error_context.component]
            cb.failure_count += 1
            cb.last_failure_time = datetime.now()
            
            if cb.failure_count >= cb.failure_threshold:
                cb.state = "open"
                logger.warning(f"Circuit breaker opened for {error_context.component}")
        
        # Update agent status
        if error_context.component in self.agent_statuses:
            if error_context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                self.agent_statuses[error_context.component] = AgentStatus.FAILED
            else:
                self.agent_statuses[error_context.component] = AgentStatus.DEGRADED
        
        # Notify monitoring callbacks
        for callback in self.monitoring_callbacks:
            try:
                callback(error_context)
            except Exception as e:
                logger.error(f"Error in monitoring callback: {e}")
        
        logger.error(f"Error logged: {error_context.component} - {error_context.message}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        healthy_agents = sum(1 for status in self.agent_statuses.values() 
                           if status == AgentStatus.HEALTHY)
        total_agents = len(self.agent_statuses)
        
        recent_errors = [e for e in self.error_history 
                        if e.timestamp > datetime.now() - timedelta(hours=1)]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "healthy_agents": healthy_agents,
            "total_agents": total_agents,
            "health_percentage": (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
            "recent_errors": len(recent_errors),
            "critical_errors": len([e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]),
            "circuit_breakers_open": len([cb for cb in self.circuit_breakers.values() if cb.state == "open"]),
            "agent_statuses": {k: v.value for k, v in self.agent_statuses.items()}
        }
    
    def can_execute(self, component: str) -> bool:
        """Check if a component can execute based on circuit breaker state."""
        if component not in self.circuit_breakers:
            return True
        
        cb = self.circuit_breakers[component]
        
        if cb.state == "closed":
            return True
        elif cb.state == "open":
            # Check if recovery timeout has passed
            if (cb.last_failure_time and 
                datetime.now() - cb.last_failure_time > timedelta(seconds=cb.recovery_timeout)):
                cb.state = "half_open"
                cb.failure_count = 0
                logger.info(f"Circuit breaker for {component} moved to half_open")
                return True
            return False
        elif cb.state == "half_open":
            return True
        
        return False
    
    def record_success(self, component: str) -> None:
        """Record a successful operation for circuit breaker management."""
        if component in self.circuit_breakers:
            cb = self.circuit_breakers[component]
            if cb.state == "half_open":
                cb.failure_count = 0
                if cb.failure_count <= -cb.success_threshold:  # Track successes
                    cb.state = "closed"
                    logger.info(f"Circuit breaker for {component} closed after recovery")
        
        # Update agent status to healthy if it was degraded
        if component in self.agent_statuses and self.agent_statuses[component] != AgentStatus.FAILED:
            self.agent_statuses[component] = AgentStatus.HEALTHY


# Global resilience manager instance
resilience_manager = SystemResilienceManager()


# Global instances for conflict resolution and rollback
class DataConflictResolver:
    """Handles data conflicts between agents and system components."""
    
    def __init__(self):
        self.resolution_strategies = {
            "pricing_conflict": self._resolve_pricing_conflict,
            "inventory_conflict": self._resolve_inventory_conflict,
            "promotion_conflict": self._resolve_promotion_conflict,
            "timestamp_conflict": self._resolve_timestamp_conflict
        }
    
    def resolve_conflict(self, conflict_type: str, conflicting_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve data conflicts using appropriate strategy."""
        if conflict_type not in self.resolution_strategies:
            logger.warning(f"No resolution strategy for conflict type: {conflict_type}")
            return self._default_resolution(conflicting_data)
        
        try:
            resolution = self.resolution_strategies[conflict_type](conflicting_data)
            logger.info(f"Resolved {conflict_type} conflict with {len(conflicting_data)} conflicting entries")
            return resolution
        except Exception as e:
            logger.error(f"Error resolving {conflict_type} conflict: {e}")
            return self._default_resolution(conflicting_data)
    
    def _resolve_pricing_conflict(self, conflicting_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve pricing conflicts by choosing the most recent decision with highest confidence."""
        if not conflicting_data:
            return {}
        
        # Sort by timestamp (most recent first) and confidence score
        sorted_data = sorted(
            conflicting_data,
            key=lambda x: (x.get('timestamp', ''), x.get('confidence_score', 0)),
            reverse=True
        )
        
        return sorted_data[0]
    
    def _resolve_inventory_conflict(self, conflicting_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve inventory conflicts by taking the most conservative (lowest) stock level."""
        if not conflicting_data:
            return {}
        
        # Choose the entry with the lowest inventory level (most conservative)
        return min(conflicting_data, key=lambda x: x.get('inventory_level', float('inf')))
    
    def _resolve_promotion_conflict(self, conflicting_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve promotion conflicts by prioritizing based on business rules."""
        if not conflicting_data:
            return {}
        
        # Priority order: flash_sale > bundle > regular_promotion
        priority_order = {'flash_sale': 3, 'bundle': 2, 'regular_promotion': 1}
        
        return max(
            conflicting_data,
            key=lambda x: priority_order.get(x.get('promotion_type', 'regular_promotion'), 0)
        )
    
    def _resolve_timestamp_conflict(self, conflicting_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve timestamp conflicts by choosing the most recent entry."""
        if not conflicting_data:
            return {}
        
        return max(conflicting_data, key=lambda x: x.get('timestamp', ''))
    
    def _default_resolution(self, conflicting_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default resolution strategy - choose the first entry."""
        return conflicting_data[0] if conflicting_data else {}


class RollbackManager:
    """Manages rollback operations for critical errors."""
    
    def __init__(self):
        self.snapshots: Dict[str, Dict[str, Any]] = {}
        self.rollback_strategies: Dict[str, Callable] = {}
    
    def create_snapshot(self, component: str, state: Dict[str, Any]) -> str:
        """Create a state snapshot for potential rollback."""
        snapshot_id = f"{component}_{int(time.time())}"
        self.snapshots[snapshot_id] = {
            "component": component,
            "timestamp": datetime.now().isoformat(),
            "state": state.copy()
        }
        logger.info(f"Created snapshot {snapshot_id} for {component}")
        return snapshot_id
    
    def register_rollback_strategy(self, component: str, strategy: Callable) -> None:
        """Register a rollback strategy for a component."""
        self.rollback_strategies[component] = strategy
        logger.info(f"Registered rollback strategy for {component}")
    
    def rollback(self, snapshot_id: str) -> bool:
        """Perform rollback to a previous state."""
        if snapshot_id not in self.snapshots:
            logger.error(f"Snapshot {snapshot_id} not found")
            return False
        
        snapshot = self.snapshots[snapshot_id]
        component = snapshot["component"]
        
        if component not in self.rollback_strategies:
            logger.error(f"No rollback strategy for component {component}")
            return False
        
        try:
            self.rollback_strategies[component](snapshot["state"])
            logger.info(f"Successfully rolled back {component} to snapshot {snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed for {component}: {e}")
            return False
    
    def cleanup_old_snapshots(self, max_age_hours: int = 24) -> None:
        """Clean up old snapshots to prevent memory bloat."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for snapshot_id, snapshot in self.snapshots.items():
            snapshot_time = datetime.fromisoformat(snapshot["timestamp"])
            if snapshot_time < cutoff_time:
                to_remove.append(snapshot_id)
        
        for snapshot_id in to_remove:
            del self.snapshots[snapshot_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old snapshots")


# Global instances
conflict_resolver = DataConflictResolver()
rollback_manager = RollbackManager()


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator for automatic retry with exponential backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)
                        
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                except Exception as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay
                        )
                        
                        if config.jitter:
                            import random
                            delay *= (0.5 + random.random() * 0.5)
                        
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay:.2f}s")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {config.max_attempts} attempts failed for {func.__name__}")
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_circuit_breaker(component_name: str):
    """Decorator for circuit breaker pattern."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not resilience_manager.can_execute(component_name):
                # Try fallback strategy
                if component_name in resilience_manager.fallback_strategies:
                    logger.info(f"Using fallback strategy for {component_name}")
                    return await resilience_manager.fallback_strategies[component_name](*args, **kwargs)
                else:
                    raise Exception(f"Circuit breaker open for {component_name} and no fallback available")
            
            try:
                result = await func(*args, **kwargs)
                resilience_manager.record_success(component_name)
                return result
            except Exception as e:
                error_context = ErrorContext(
                    error_id=f"{component_name}_{int(time.time())}",
                    timestamp=datetime.now(),
                    component=component_name,
                    error_type=type(e).__name__,
                    severity=ErrorSeverity.MEDIUM,
                    message=str(e),
                    stack_trace=str(e.__traceback__) if e.__traceback__ else None
                )
                resilience_manager.log_error(error_context)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not resilience_manager.can_execute(component_name):
                # Try fallback strategy
                if component_name in resilience_manager.fallback_strategies:
                    logger.info(f"Using fallback strategy for {component_name}")
                    return resilience_manager.fallback_strategies[component_name](*args, **kwargs)
                else:
                    raise Exception(f"Circuit breaker open for {component_name} and no fallback available")
            
            try:
                result = func(*args, **kwargs)
                resilience_manager.record_success(component_name)
                return result
            except Exception as e:
                error_context = ErrorContext(
                    error_id=f"{component_name}_{int(time.time())}",
                    timestamp=datetime.now(),
                    component=component_name,
                    error_type=type(e).__name__,
                    severity=ErrorSeverity.MEDIUM,
                    message=str(e),
                    stack_trace=str(e.__traceback__) if e.__traceback__ else None
                )
                resilience_manager.log_error(error_context)
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def handle_critical_error(component: str, error: Exception, context: Dict[str, Any] = None) -> None:
    """Handle critical errors with comprehensive logging and recovery attempts."""
    error_context = ErrorContext(
        error_id=f"critical_{component}_{int(time.time())}",
        timestamp=datetime.now(),
        component=component,
        error_type=type(error).__name__,
        severity=ErrorSeverity.CRITICAL,
        message=str(error),
        stack_trace=str(error.__traceback__) if error.__traceback__ else None,
        metadata=context or {}
    )
    
    resilience_manager.log_error(error_context)
    
    # Attempt recovery strategies
    logger.critical(f"Critical error in {component}: {error}")
    
    # Try to activate fallback mechanisms
    if component in resilience_manager.fallback_strategies:
        try:
            logger.info(f"Attempting fallback strategy for {component}")
            resilience_manager.fallback_strategies[component]()
        except Exception as fallback_error:
            logger.error(f"Fallback strategy failed for {component}: {fallback_error}")


def get_system_status() -> Dict[str, Any]:
    """Get comprehensive system status for monitoring."""
    return {
        "system_health": resilience_manager.get_system_health(),
        "recent_errors": [
            {
                "component": e.component,
                "severity": e.severity.value,
                "message": e.message,
                "timestamp": e.timestamp.isoformat()
            }
            for e in resilience_manager.error_history[-10:]  # Last 10 errors
        ],
        "circuit_breakers": {
            name: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            for name, cb in resilience_manager.circuit_breakers.items()
        }
    }