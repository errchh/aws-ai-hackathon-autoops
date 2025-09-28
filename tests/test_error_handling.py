"""
Tests for error handling and system resilience functionality.
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agents.error_handling import (
    SystemResilienceManager,
    ErrorContext,
    ErrorSeverity,
    AgentStatus,
    CircuitBreakerState,
    RetryConfig,
    DataConflictResolver,
    RollbackManager,
    with_retry,
    with_circuit_breaker,
    handle_critical_error,
    get_system_status,
    resilience_manager,
    conflict_resolver,
    rollback_manager
)


class TestSystemResilienceManager:
    """Test cases for SystemResilienceManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SystemResilienceManager()
    
    def test_register_agent(self):
        """Test agent registration."""
        agent_id = "test_agent"
        self.manager.register_agent(agent_id)
        
        assert agent_id in self.manager.agent_statuses
        assert self.manager.agent_statuses[agent_id] == AgentStatus.HEALTHY
        assert agent_id in self.manager.circuit_breakers
        assert isinstance(self.manager.circuit_breakers[agent_id], CircuitBreakerState)
    
    def test_register_fallback_strategy(self):
        """Test fallback strategy registration."""
        component = "test_component"
        strategy = Mock()
        
        self.manager.register_fallback_strategy(component, strategy)
        
        assert component in self.manager.fallback_strategies
        assert self.manager.fallback_strategies[component] == strategy
    
    def test_log_error_updates_circuit_breaker(self):
        """Test that logging errors updates circuit breaker state."""
        component = "test_component"
        self.manager.register_agent(component)
        
        error_context = ErrorContext(
            error_id="test_error",
            timestamp=datetime.now(),
            component=component,
            error_type="TestError",
            severity=ErrorSeverity.MEDIUM,
            message="Test error message"
        )
        
        # Log multiple errors to trigger circuit breaker
        for i in range(6):  # Exceeds default threshold of 5
            self.manager.log_error(error_context)
        
        cb = self.manager.circuit_breakers[component]
        assert cb.state == "open"
        assert cb.failure_count >= 5
    
    def test_log_error_updates_agent_status(self):
        """Test that logging errors updates agent status."""
        component = "test_agent"
        self.manager.register_agent(component)
        
        # Test medium severity error
        error_context = ErrorContext(
            error_id="test_error",
            timestamp=datetime.now(),
            component=component,
            error_type="TestError",
            severity=ErrorSeverity.MEDIUM,
            message="Test error message"
        )
        
        self.manager.log_error(error_context)
        assert self.manager.agent_statuses[component] == AgentStatus.DEGRADED
        
        # Test critical severity error
        error_context.severity = ErrorSeverity.CRITICAL
        self.manager.log_error(error_context)
        assert self.manager.agent_statuses[component] == AgentStatus.FAILED
    
    def test_get_system_health(self):
        """Test system health reporting."""
        # Register some agents
        self.manager.register_agent("agent1")
        self.manager.register_agent("agent2")
        self.manager.register_agent("agent3")
        
        # Create some errors
        error_context = ErrorContext(
            error_id="test_error",
            timestamp=datetime.now(),
            component="agent1",
            error_type="TestError",
            severity=ErrorSeverity.CRITICAL,
            message="Test error"
        )
        self.manager.log_error(error_context)
        
        health = self.manager.get_system_health()
        
        assert "timestamp" in health
        assert health["total_agents"] == 3
        assert health["healthy_agents"] == 2  # agent1 failed, others healthy
        assert health["health_percentage"] == pytest.approx(66.67, rel=1e-2)
        assert health["recent_errors"] == 1
        assert health["critical_errors"] == 1
    
    def test_can_execute_circuit_breaker_logic(self):
        """Test circuit breaker execution logic."""
        component = "test_component"
        self.manager.register_agent(component)
        
        # Initially should be able to execute
        assert self.manager.can_execute(component) is True
        
        # Trigger circuit breaker
        cb = self.manager.circuit_breakers[component]
        cb.state = "open"
        cb.last_failure_time = datetime.now()
        
        # Should not be able to execute when open
        assert self.manager.can_execute(component) is False
        
        # Simulate recovery timeout
        cb.last_failure_time = datetime.now() - timedelta(seconds=61)
        
        # Should move to half_open and allow execution
        assert self.manager.can_execute(component) is True
        assert cb.state == "half_open"
    
    def test_record_success(self):
        """Test recording successful operations."""
        component = "test_component"
        self.manager.register_agent(component)
        
        # Set agent to degraded
        self.manager.agent_statuses[component] = AgentStatus.DEGRADED
        
        # Record success
        self.manager.record_success(component)
        
        # Should return to healthy
        assert self.manager.agent_statuses[component] == AgentStatus.HEALTHY


class TestRetryDecorator:
    """Test cases for retry decorator."""
    
    @pytest.mark.asyncio
    async def test_async_retry_success_on_first_attempt(self):
        """Test async retry succeeds on first attempt."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3))
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_retry_success_after_failures(self):
        """Test async retry succeeds after initial failures."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await test_func()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_exhausts_attempts(self):
        """Test async retry exhausts all attempts."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")
        
        with pytest.raises(Exception, match="Persistent failure"):
            await test_func()
        
        assert call_count == 3
    
    def test_sync_retry_success_on_first_attempt(self):
        """Test sync retry succeeds on first attempt."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3))
        def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 1
    
    def test_sync_retry_success_after_failures(self):
        """Test sync retry succeeds after initial failures."""
        call_count = 0
        
        @with_retry(RetryConfig(max_attempts=3, base_delay=0.1))
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = test_func()
        assert result == "success"
        assert call_count == 3


class TestCircuitBreakerDecorator:
    """Test cases for circuit breaker decorator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Reset global resilience manager
        resilience_manager.circuit_breakers.clear()
        resilience_manager.agent_statuses.clear()
        resilience_manager.fallback_strategies.clear()
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker_allows_execution(self):
        """Test async circuit breaker allows execution when closed."""
        component = "test_component"
        resilience_manager.register_agent(component)
        
        @with_circuit_breaker(component)
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker_blocks_execution(self):
        """Test async circuit breaker blocks execution when open."""
        component = "test_component"
        resilience_manager.register_agent(component)
        
        # Force circuit breaker open
        cb = resilience_manager.circuit_breakers[component]
        cb.state = "open"
        cb.last_failure_time = datetime.now()
        
        @with_circuit_breaker(component)
        async def test_func():
            return "success"
        
        with pytest.raises(Exception, match="Circuit breaker open"):
            await test_func()
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker_uses_fallback(self):
        """Test async circuit breaker uses fallback strategy."""
        component = "test_component"
        resilience_manager.register_agent(component)
        
        # Register fallback strategy
        async def fallback_strategy(*args, **kwargs):
            return "fallback_result"
        
        resilience_manager.register_fallback_strategy(component, fallback_strategy)
        
        # Force circuit breaker open
        cb = resilience_manager.circuit_breakers[component]
        cb.state = "open"
        cb.last_failure_time = datetime.now()
        
        @with_circuit_breaker(component)
        async def test_func():
            return "success"
        
        result = await test_func()
        assert result == "fallback_result"


class TestDataConflictResolver:
    """Test cases for DataConflictResolver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.resolver = DataConflictResolver()
    
    def test_resolve_pricing_conflict(self):
        """Test pricing conflict resolution."""
        conflicting_data = [
            {"timestamp": "2024-01-01T10:00:00", "confidence_score": 0.8, "price": 10.0},
            {"timestamp": "2024-01-01T11:00:00", "confidence_score": 0.9, "price": 12.0},
            {"timestamp": "2024-01-01T09:00:00", "confidence_score": 0.7, "price": 9.0}
        ]
        
        result = self.resolver.resolve_conflict("pricing_conflict", conflicting_data)
        
        # Should choose the most recent with highest confidence
        assert result["price"] == 12.0
        assert result["timestamp"] == "2024-01-01T11:00:00"
    
    def test_resolve_inventory_conflict(self):
        """Test inventory conflict resolution."""
        conflicting_data = [
            {"inventory_level": 100, "source": "agent1"},
            {"inventory_level": 80, "source": "agent2"},
            {"inventory_level": 120, "source": "agent3"}
        ]
        
        result = self.resolver.resolve_conflict("inventory_conflict", conflicting_data)
        
        # Should choose the most conservative (lowest) inventory level
        assert result["inventory_level"] == 80
        assert result["source"] == "agent2"
    
    def test_resolve_promotion_conflict(self):
        """Test promotion conflict resolution."""
        conflicting_data = [
            {"promotion_type": "regular_promotion", "discount": 0.1},
            {"promotion_type": "flash_sale", "discount": 0.3},
            {"promotion_type": "bundle", "discount": 0.2}
        ]
        
        result = self.resolver.resolve_conflict("promotion_conflict", conflicting_data)
        
        # Should prioritize flash_sale over others
        assert result["promotion_type"] == "flash_sale"
        assert result["discount"] == 0.3
    
    def test_resolve_unknown_conflict_type(self):
        """Test handling of unknown conflict types."""
        conflicting_data = [
            {"data": "value1"},
            {"data": "value2"}
        ]
        
        result = self.resolver.resolve_conflict("unknown_conflict", conflicting_data)
        
        # Should use default resolution (first entry)
        assert result["data"] == "value1"


class TestRollbackManager:
    """Test cases for RollbackManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = RollbackManager()
    
    def test_create_snapshot(self):
        """Test snapshot creation."""
        component = "test_component"
        state = {"key": "value", "number": 42}
        
        snapshot_id = self.manager.create_snapshot(component, state)
        
        assert snapshot_id in self.manager.snapshots
        snapshot = self.manager.snapshots[snapshot_id]
        assert snapshot["component"] == component
        assert snapshot["state"] == state
        assert "timestamp" in snapshot
    
    def test_register_rollback_strategy(self):
        """Test rollback strategy registration."""
        component = "test_component"
        strategy = Mock()
        
        self.manager.register_rollback_strategy(component, strategy)
        
        assert component in self.manager.rollback_strategies
        assert self.manager.rollback_strategies[component] == strategy
    
    def test_successful_rollback(self):
        """Test successful rollback operation."""
        component = "test_component"
        state = {"key": "value"}
        
        # Create snapshot
        snapshot_id = self.manager.create_snapshot(component, state)
        
        # Register rollback strategy
        rollback_strategy = Mock()
        self.manager.register_rollback_strategy(component, rollback_strategy)
        
        # Perform rollback
        result = self.manager.rollback(snapshot_id)
        
        assert result is True
        rollback_strategy.assert_called_once_with(state)
    
    def test_rollback_with_missing_snapshot(self):
        """Test rollback with missing snapshot."""
        result = self.manager.rollback("nonexistent_snapshot")
        assert result is False
    
    def test_rollback_with_missing_strategy(self):
        """Test rollback with missing strategy."""
        component = "test_component"
        state = {"key": "value"}
        
        # Create snapshot but no strategy
        snapshot_id = self.manager.create_snapshot(component, state)
        
        result = self.manager.rollback(snapshot_id)
        assert result is False
    
    def test_cleanup_old_snapshots(self):
        """Test cleanup of old snapshots."""
        # Create old snapshot
        old_snapshot_id = self.manager.create_snapshot("old_component", {"old": "data"})
        old_snapshot = self.manager.snapshots[old_snapshot_id]
        old_snapshot["timestamp"] = (datetime.now() - timedelta(hours=25)).isoformat()
        
        # Create recent snapshot
        recent_snapshot_id = self.manager.create_snapshot("recent_component", {"recent": "data"})
        
        # Cleanup with 24 hour threshold
        self.manager.cleanup_old_snapshots(max_age_hours=24)
        
        # Old snapshot should be removed, recent should remain
        assert old_snapshot_id not in self.manager.snapshots
        assert recent_snapshot_id in self.manager.snapshots


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_handle_critical_error(self):
        """Test critical error handling."""
        component = "test_component"
        error = Exception("Critical test error")
        context = {"additional": "info"}
        
        # Mock the resilience manager
        with patch.object(resilience_manager, 'log_error') as mock_log_error:
            handle_critical_error(component, error, context)
            
            # Verify error was logged
            mock_log_error.assert_called_once()
            error_context = mock_log_error.call_args[0][0]
            assert error_context.component == component
            assert error_context.severity == ErrorSeverity.CRITICAL
            assert error_context.message == str(error)
            assert error_context.metadata == context
    
    def test_get_system_status(self):
        """Test system status retrieval."""
        # Add some test data to resilience manager
        resilience_manager.register_agent("test_agent")
        
        error_context = ErrorContext(
            error_id="test_error",
            timestamp=datetime.now(),
            component="test_agent",
            error_type="TestError",
            severity=ErrorSeverity.MEDIUM,
            message="Test error"
        )
        resilience_manager.log_error(error_context)
        
        status = get_system_status()
        
        assert "system_health" in status
        assert "recent_errors" in status
        assert "circuit_breakers" in status
        assert len(status["recent_errors"]) > 0
        assert "test_agent" in status["circuit_breakers"]


if __name__ == "__main__":
    pytest.main([__file__])