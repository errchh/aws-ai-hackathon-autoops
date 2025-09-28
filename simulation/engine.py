"""
Main Simulation Engine for Healthcare and Wellness Retail Data Generation

This engine orchestrates all simulation components to generate realistic
retail scenarios for demonstrating the autoops system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from models.core import Product, MarketEvent, AgentDecision
from .products import ProductCatalog
from .demand import DemandGenerator
from .competitors import CompetitorLandscape
from .social import SocialSentimentSimulator
from .triggers import TriggerEngine
from .iot import IoTSensorSimulator
from config.simulation_event_capture import get_simulation_event_capture, TriggerEventType

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation operating modes"""

    DEMO = "demo"  # Fast-paced for demonstrations
    REALISTIC = "realistic"  # Slower, more detailed simulation
    STRESS_TEST = "stress_test"  # High-frequency events for testing


@dataclass
class SimulationState:
    """Current state of the simulation"""

    current_time: datetime = field(default_factory=datetime.now)
    active_triggers: List[Dict[str, Any]] = field(default_factory=list)
    market_events: List[MarketEvent] = field(default_factory=list)
    agent_decisions: List[AgentDecision] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    is_running: bool = False


class SimulationEngine:
    """
    Main simulation engine that coordinates all simulation components
    to generate realistic healthcare and wellness retail scenarios.
    """

    def __init__(self, mode: SimulationMode = SimulationMode.DEMO):
        self.mode = mode
        self.state = SimulationState()

        # Initialize simulation components
        self.product_catalog = ProductCatalog()
        self.demand_generator = DemandGenerator()
        self.competitor_landscape = CompetitorLandscape()
        self.social_sentiment = SocialSentimentSimulator()
        self.trigger_engine = TriggerEngine()
        self.iot_sensors = IoTSensorSimulator()

        # Initialize event capture for Langfuse tracing
        self.event_capture = get_simulation_event_capture()

        # Simulation timing
        self.time_multiplier = {
            SimulationMode.DEMO: 60,  # 1 minute = 1 hour
            SimulationMode.REALISTIC: 1,  # Real-time
            SimulationMode.STRESS_TEST: 3600,  # 1 second = 1 hour
        }[mode]

        logger.info(f"Initialized SimulationEngine in {mode.value} mode")

    async def initialize(self) -> None:
        """Initialize all simulation components"""
        logger.info("Initializing simulation components...")

        # Initialize product catalog
        await self.product_catalog.initialize()

        # Initialize demand patterns
        await self.demand_generator.initialize(self.product_catalog)

        # Initialize competitors
        await self.competitor_landscape.initialize()

        # Initialize social sentiment
        await self.social_sentiment.initialize()

        # Initialize trigger engine
        await self.trigger_engine.initialize(self.product_catalog)

        # Initialize IoT sensors
        await self.iot_sensors.initialize(self.product_catalog)

        logger.info("Simulation components initialized successfully")

    async def start_simulation(self) -> None:
        """Start the simulation engine"""
        if self.state.is_running:
            logger.warning("Simulation is already running")
            return

        self.state.is_running = True
        self.state.current_time = datetime.now()

        logger.info(f"Starting simulation in {self.mode.value} mode")

        # Capture simulation start event for tracing
        start_event_data = {
            "type": TriggerEventType.SIMULATION_START.value,
            "source": "simulation_engine",
            "mode": self.mode.value,
            "timestamp": self.state.current_time.isoformat(),
            "components": [
                "product_catalog", "demand_generator", "competitor_landscape",
                "social_sentiment", "trigger_engine", "iot_sensors"
            ],
            "metadata": {
                "simulation_mode": self.mode.value,
                "start_time": self.state.current_time.isoformat(),
                "system_version": "1.0.0"
            }
        }
        self.event_capture.capture_trigger_event(start_event_data)

        # Start background simulation tasks
        asyncio.create_task(self._simulation_loop())
        asyncio.create_task(self._trigger_scheduler())

    async def stop_simulation(self) -> None:
        """Stop the simulation engine"""
        stop_time = datetime.now()
        duration = (stop_time - self.state.current_time).total_seconds()
        
        self.state.is_running = False
        
        # Capture simulation stop event for tracing
        stop_event_data = {
            "type": TriggerEventType.SIMULATION_STOP.value,
            "source": "simulation_engine",
            "duration": duration,
            "final_metrics": self.state.performance_metrics.copy(),
            "stop_reason": "manual_stop",
            "metadata": {
                "stop_time": stop_time.isoformat(),
                "total_events": len(self.state.market_events),
                "performance_summary": {
                    "total_triggers": len(self.state.active_triggers),
                    "total_decisions": len(self.state.agent_decisions),
                    "uptime_seconds": duration
                }
            }
        }
        self.event_capture.capture_trigger_event(stop_event_data)
        
        logger.info("Simulation stopped")

    async def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state for dashboard display"""
        return {
            "current_time": self.state.current_time.isoformat(),
            "active_triggers": len(self.state.active_triggers),
            "market_events_count": len(self.state.market_events),
            "agent_decisions_count": len(self.state.agent_decisions),
            "performance_metrics": self.state.performance_metrics,
            "is_running": self.state.is_running,
            "mode": self.mode.value,
        }

    async def get_products_data(self) -> List[Dict[str, Any]]:
        """Get current product data for dashboard"""
        return await self.product_catalog.get_all_products_data()

    async def get_market_data(self) -> Dict[str, Any]:
        """Get current market conditions"""
        return {
            "demand_patterns": await self.demand_generator.get_current_demand(),
            "competitor_prices": await self.competitor_landscape.get_current_prices(),
            "social_sentiment": await self.social_sentiment.get_current_sentiment(),
            "iot_readings": await self.iot_sensors.get_current_readings(),
        }

    async def trigger_scenario(
        self, scenario_name: str, intensity: str = "medium"
    ) -> bool:
        """Trigger a specific simulation scenario"""
        try:
            # Get scenario details before triggering
            available_triggers = await self.trigger_engine.get_available_triggers()
            scenario_data = None
            for trigger in available_triggers:
                if trigger["name"] == scenario_name:
                    scenario_data = trigger
                    break
            
            # Trigger the scenario
            success = await self.trigger_engine.trigger_scenario(scenario_name, intensity)
            
            if success and scenario_data:
                # Capture scenario trigger event for tracing
                trigger_event_data = {
                    "type": TriggerEventType.TRIGGER_SCENARIO.value,
                    "source": "simulation_engine",
                    "scenario_id": scenario_data["id"],
                    "scenario_name": scenario_name,
                    "intensity": intensity,
                    "conditions": {},  # Would be populated from scenario details
                    "affected_agents": [scenario_data["agent_type"]],
                    "metadata": {
                        "agent_type": scenario_data["agent_type"],
                        "trigger_type": scenario_data.get("trigger_type", "manual"),
                        "expected_effects": {},  # Would be populated from scenario
                        "cooldown_minutes": scenario_data.get("cooldown_minutes", 30)
                    }
                }
                self.event_capture.capture_trigger_event(trigger_event_data)
            
            logger.info(
                f"Triggered scenario: {scenario_name} with {intensity} intensity"
            )
            return success
        except Exception as e:
            logger.error(f"Failed to trigger scenario {scenario_name}: {e}")
            return False

    async def get_available_triggers(self) -> List[Dict[str, Any]]:
        """Get list of available trigger scenarios"""
        return await self.trigger_engine.get_available_triggers()

    async def _simulation_loop(self) -> None:
        """Main simulation loop that advances time and generates events"""
        while self.state.is_running:
            try:
                # Advance simulation time
                time_advance = timedelta(seconds=self.time_multiplier)
                self.state.current_time += time_advance

                # Update all simulation components
                await self._update_components()

                # Generate random events based on mode
                await self._generate_random_events()

                # Update performance metrics
                await self._update_metrics()

                # Sleep based on mode
                sleep_time = {
                    SimulationMode.DEMO: 1.0,  # Update every second
                    SimulationMode.REALISTIC: 60.0,  # Update every minute
                    SimulationMode.STRESS_TEST: 0.1,  # Update 10x per second
                }[self.mode]

                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _trigger_scheduler(self) -> None:
        """Background task to schedule and execute triggers"""
        while self.state.is_running:
            try:
                # Process scheduled triggers
                await self.trigger_engine.process_scheduled_triggers()

                # Clean up expired triggers
                await self.trigger_engine.cleanup_expired_triggers()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in trigger scheduler: {e}")
                await asyncio.sleep(10)

    async def _update_components(self) -> None:
        """Update all simulation components for current time"""
        # Update demand patterns
        await self.demand_generator.update_for_time(self.state.current_time)

        # Update competitor actions
        await self.competitor_landscape.update_competitors(self.state.current_time)

        # Update social sentiment
        await self.social_sentiment.update_sentiment(self.state.current_time)

        # Update IoT sensors
        await self.iot_sensors.update_sensors(self.state.current_time)

    async def _generate_random_events(self) -> None:
        """Generate random market events based on current conditions"""
        # This will be implemented based on probability distributions
        # For now, it's a placeholder for random event generation
        pass

    async def _update_metrics(self) -> None:
        """Update simulation performance metrics"""
        # Calculate various KPIs for the simulation
        self.state.performance_metrics.update(
            {
                "uptime_seconds": (
                    datetime.now() - self.state.current_time
                ).total_seconds(),
                "events_per_hour": len(self.state.market_events)
                / max(
                    1, (datetime.now() - self.state.current_time).total_seconds() / 3600
                ),
                "decisions_per_hour": len(self.state.agent_decisions)
                / max(
                    1, (datetime.now() - self.state.current_time).total_seconds() / 3600
                ),
            }
        )

    async def reset_simulation(self) -> None:
        """Reset simulation to initial state"""
        logger.info("Resetting simulation...")
        await self.stop_simulation()

        # Reset all components
        self.state = SimulationState()
        await self.product_catalog.reset()
        await self.demand_generator.reset()
        await self.competitor_landscape.reset()
        await self.social_sentiment.reset()
        await self.trigger_engine.reset()
        await self.iot_sensors.reset()

        logger.info("Simulation reset complete")
