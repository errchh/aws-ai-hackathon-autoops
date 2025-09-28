"""
Examples of integrating error handling and resilience into the retail optimization system.

This module demonstrates how to use the error handling components with the existing
agents and API endpoints.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List

from agents.error_handling import (
    resilience_manager,
    conflict_resolver,
    rollback_manager,
    with_retry,
    with_circuit_breaker,
    handle_critical_error,
    RetryConfig,
    ErrorSeverity,
    ErrorContext
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResilientPricingAgent:
    """Example of integrating error handling into the Pricing Agent."""
    
    def __init__(self):
        self.agent_id = "pricing_agent"
        self.current_prices = {}
        
        # Register with resilience manager
        resilience_manager.register_agent(self.agent_id)
        
        # Register fallback strategies
        resilience_manager.register_fallback_strategy(
            self.agent_id, 
            self._fallback_pricing_strategy
        )
        
        # Register rollback strategy
        rollback_manager.register_rollback_strategy(
            self.agent_id,
            self._rollback_pricing_state
        )
    
    @with_circuit_breaker("pricing_agent")
    @with_retry(RetryConfig(max_attempts=3, base_delay=1.0))
    async def update_price(self, product_id: str, new_price: float, reason: str) -> Dict[str, Any]:
        """Update product price with error handling and resilience."""
        try:
            # Create snapshot before making changes
            snapshot_id = rollback_manager.create_snapshot(
                self.agent_id,
                {"prices": self.current_prices.copy()}
            )
            
            # Simulate potential failure points
            if new_price < 0:
                raise ValueError("Price cannot be negative")
            
            if product_id == "FAIL_PRODUCT":
                raise Exception("Simulated pricing service failure")
            
            # Update price
            old_price = self.current_prices.get(product_id, 0.0)
            self.current_prices[product_id] = new_price
            
            logger.info(f"Updated price for {product_id}: ${old_price} -> ${new_price}")
            
            return {
                "product_id": product_id,
                "old_price": old_price,
                "new_price": new_price,
                "reason": reason,
                "timestamp": datetime.now().isoformat(),
                "snapshot_id": snapshot_id
            }
            
        except Exception as e:
            # Handle critical errors
            if isinstance(e, ValueError):
                handle_critical_error(self.agent_id, e, {
                    "product_id": product_id,
                    "attempted_price": new_price,
                    "operation": "update_price"
                })
            raise
    
    async def _fallback_pricing_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback strategy when main pricing logic fails."""
        logger.warning("Using fallback pricing strategy")
        
        # Return a safe default response
        return {
            "product_id": kwargs.get("product_id", "unknown"),
            "old_price": 0.0,
            "new_price": 0.0,
            "reason": "fallback_strategy_applied",
            "timestamp": datetime.now().isoformat(),
            "fallback": True
        }
    
    def _rollback_pricing_state(self, state: Dict[str, Any]) -> None:
        """Rollback pricing state to a previous snapshot."""
        logger.info("Rolling back pricing state")
        self.current_prices = state.get("prices", {})


class ResilientInventoryAgent:
    """Example of integrating error handling into the Inventory Agent."""
    
    def __init__(self):
        self.agent_id = "inventory_agent"
        self.inventory_levels = {}
        
        # Register with resilience manager
        resilience_manager.register_agent(self.agent_id)
        
        # Register fallback strategy
        resilience_manager.register_fallback_strategy(
            self.agent_id,
            self._fallback_inventory_strategy
        )
    
    @with_circuit_breaker("inventory_agent")
    @with_retry(RetryConfig(max_attempts=2, base_delay=0.5))
    async def update_inventory(self, product_id: str, quantity: int, operation: str) -> Dict[str, Any]:
        """Update inventory levels with error handling."""
        try:
            # Validate inputs
            if quantity < 0 and operation == "add":
                raise ValueError("Cannot add negative quantity")
            
            # Simulate database connection failure
            if product_id == "DB_FAIL":
                raise ConnectionError("Database connection failed")
            
            # Update inventory
            current_level = self.inventory_levels.get(product_id, 0)
            
            if operation == "add":
                new_level = current_level + quantity
            elif operation == "remove":
                new_level = max(0, current_level - quantity)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            self.inventory_levels[product_id] = new_level
            
            logger.info(f"Updated inventory for {product_id}: {current_level} -> {new_level}")
            
            return {
                "product_id": product_id,
                "old_level": current_level,
                "new_level": new_level,
                "operation": operation,
                "quantity": quantity,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Log error context
            error_context = ErrorContext(
                error_id=f"inventory_{product_id}_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                component=self.agent_id,
                error_type=type(e).__name__,
                severity=ErrorSeverity.HIGH if isinstance(e, ConnectionError) else ErrorSeverity.MEDIUM,
                message=str(e),
                metadata={
                    "product_id": product_id,
                    "quantity": quantity,
                    "operation": operation
                }
            )
            resilience_manager.log_error(error_context)
            raise
    
    async def _fallback_inventory_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback strategy for inventory operations."""
        logger.warning("Using fallback inventory strategy")
        
        return {
            "product_id": kwargs.get("product_id", "unknown"),
            "old_level": 0,
            "new_level": 0,
            "operation": "fallback",
            "quantity": 0,
            "timestamp": datetime.now().isoformat(),
            "fallback": True
        }


class DataConflictExample:
    """Example of handling data conflicts between agents."""
    
    def __init__(self):
        self.pricing_agent = ResilientPricingAgent()
        self.inventory_agent = ResilientInventoryAgent()
    
    async def handle_conflicting_pricing_decisions(self, product_id: str) -> Dict[str, Any]:
        """Demonstrate handling conflicting pricing decisions."""
        
        # Simulate conflicting pricing decisions from different sources
        conflicting_prices = [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "confidence_score": 0.8,
                "price": 25.99,
                "source": "demand_analysis",
                "rationale": "High demand detected"
            },
            {
                "timestamp": "2024-01-15T10:05:00Z",
                "confidence_score": 0.9,
                "price": 22.99,
                "source": "competitor_analysis",
                "rationale": "Competitor price matching"
            },
            {
                "timestamp": "2024-01-15T09:55:00Z",
                "confidence_score": 0.7,
                "price": 28.99,
                "source": "inventory_clearance",
                "rationale": "Slow moving inventory"
            }
        ]
        
        # Resolve conflict using the conflict resolver
        resolved_decision = conflict_resolver.resolve_conflict(
            "pricing_conflict",
            conflicting_prices
        )
        
        logger.info(f"Resolved pricing conflict for {product_id}: {resolved_decision}")
        
        # Apply the resolved decision
        try:
            result = await self.pricing_agent.update_price(
                product_id,
                resolved_decision["price"],
                f"Conflict resolution: {resolved_decision['rationale']}"
            )
            return result
        except Exception as e:
            logger.error(f"Failed to apply resolved pricing decision: {e}")
            raise
    
    async def handle_inventory_conflicts(self, product_id: str) -> Dict[str, Any]:
        """Demonstrate handling inventory level conflicts."""
        
        # Simulate conflicting inventory reports
        conflicting_inventory = [
            {
                "inventory_level": 150,
                "source": "warehouse_system",
                "timestamp": "2024-01-15T10:00:00Z"
            },
            {
                "inventory_level": 142,
                "source": "pos_system",
                "timestamp": "2024-01-15T10:01:00Z"
            },
            {
                "inventory_level": 148,
                "source": "manual_count",
                "timestamp": "2024-01-15T09:58:00Z"
            }
        ]
        
        # Resolve conflict (should choose most conservative level)
        resolved_inventory = conflict_resolver.resolve_conflict(
            "inventory_conflict",
            conflicting_inventory
        )
        
        logger.info(f"Resolved inventory conflict for {product_id}: {resolved_inventory}")
        
        # Update inventory to resolved level
        current_level = self.inventory_agent.inventory_levels.get(product_id, 0)
        adjustment = resolved_inventory["inventory_level"] - current_level
        
        if adjustment != 0:
            operation = "add" if adjustment > 0 else "remove"
            quantity = abs(adjustment)
            
            result = await self.inventory_agent.update_inventory(
                product_id, quantity, operation
            )
            return result
        
        return {"message": "No adjustment needed", "level": current_level}


class SystemHealthMonitor:
    """Example of monitoring system health and responding to issues."""
    
    def __init__(self):
        # Register monitoring callback
        resilience_manager.add_monitoring_callback(self._handle_error_event)
    
    def _handle_error_event(self, error_context: ErrorContext) -> None:
        """Handle error events from the resilience manager."""
        logger.info(f"Monitoring: Error detected in {error_context.component}")
        
        # Take action based on error severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error_context)
        elif error_context.severity == ErrorSeverity.HIGH:
            self._handle_high_severity_error(error_context)
    
    def _handle_critical_error(self, error_context: ErrorContext) -> None:
        """Handle critical errors."""
        logger.critical(f"CRITICAL ERROR: {error_context.component} - {error_context.message}")
        
        # Could trigger alerts, notifications, or emergency procedures
        # For demo purposes, just log
        logger.critical("Triggering emergency response procedures")
    
    def _handle_high_severity_error(self, error_context: ErrorContext) -> None:
        """Handle high severity errors."""
        logger.warning(f"HIGH SEVERITY: {error_context.component} - {error_context.message}")
        
        # Could trigger automated recovery procedures
        logger.warning("Initiating automated recovery procedures")
    
    def get_system_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive system status report."""
        health = resilience_manager.get_system_health()
        
        # Add additional monitoring metrics
        report = {
            "system_health": health,
            "monitoring_active": True,
            "last_check": datetime.now().isoformat(),
            "recommendations": []
        }
        
        # Generate recommendations based on system state
        if health["health_percentage"] < 50:
            report["recommendations"].append("URGENT: System health below 50% - investigate immediately")
        
        if health["critical_errors"] > 0:
            report["recommendations"].append("Critical errors detected - review error logs")
        
        if health["circuit_breakers_open"] > 0:
            report["recommendations"].append("Circuit breakers open - check component health")
        
        return report


async def demonstrate_error_handling():
    """Demonstrate the error handling and resilience features."""
    logger.info("Starting error handling demonstration")
    
    # Initialize components
    pricing_agent = ResilientPricingAgent()
    inventory_agent = ResilientInventoryAgent()
    conflict_handler = DataConflictExample()
    health_monitor = SystemHealthMonitor()
    
    try:
        # Demonstrate successful operations
        logger.info("=== Demonstrating successful operations ===")
        
        price_result = await pricing_agent.update_price("PRODUCT_001", 29.99, "Market analysis")
        logger.info(f"Price update result: {price_result}")
        
        inventory_result = await inventory_agent.update_inventory("PRODUCT_001", 50, "add")
        logger.info(f"Inventory update result: {inventory_result}")
        
        # Demonstrate error handling
        logger.info("=== Demonstrating error handling ===")
        
        try:
            # This should trigger validation error
            await pricing_agent.update_price("PRODUCT_002", -10.0, "Invalid price test")
        except Exception as e:
            logger.info(f"Caught expected error: {e}")
        
        try:
            # This should trigger circuit breaker after retries
            await pricing_agent.update_price("FAIL_PRODUCT", 25.0, "Circuit breaker test")
        except Exception as e:
            logger.info(f"Caught expected error: {e}")
        
        # Demonstrate conflict resolution
        logger.info("=== Demonstrating conflict resolution ===")
        
        conflict_result = await conflict_handler.handle_conflicting_pricing_decisions("PRODUCT_003")
        logger.info(f"Conflict resolution result: {conflict_result}")
        
        # Show system health
        logger.info("=== System health report ===")
        health_report = health_monitor.get_system_status_report()
        logger.info(f"System health: {health_report}")
        
    except Exception as e:
        logger.error(f"Demonstration error: {e}")
    
    logger.info("Error handling demonstration completed")


if __name__ == "__main__":
    asyncio.run(demonstrate_error_handling())