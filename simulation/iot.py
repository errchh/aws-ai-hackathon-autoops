"""
IoT Sensor Simulator for Healthcare and Wellness Products

This module simulates IoT sensors that monitor product conditions
like temperature, humidity, and expiration tracking.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import random

from .products import ProductCatalog

logger = logging.getLogger(__name__)


class IoTSensorSimulator:
    """
    Simulates IoT sensors for monitoring healthcare and wellness products,
    including temperature control, humidity monitoring, and expiration alerts.
    """

    def __init__(self):
        self.product_catalog: Optional[ProductCatalog] = None
        self.sensor_readings: Dict[
            str, Dict[str, Any]
        ] = {}  # product_id -> sensor data
        self.alerts: List[Dict[str, Any]] = []
        self._initialized = False

    async def initialize(self, product_catalog: ProductCatalog) -> None:
        """Initialize IoT sensors for products that need monitoring"""
        if self._initialized:
            return

        self.product_catalog = product_catalog
        logger.info("Initializing IoT sensor simulator...")

        # Initialize sensors for products that need monitoring
        for product_id, product_data in self.product_catalog.product_data.items():
            if self._needs_monitoring(product_data):
                self.sensor_readings[product_id] = {
                    "temperature_celsius": self._get_baseline_temp(product_data),
                    "humidity_percent": self._get_baseline_humidity(product_data),
                    "last_reading": datetime.now(),
                    "alerts_active": [],
                    "sensor_health": "online",
                }

        self._initialized = True
        logger.info(f"Initialized IoT sensors for {len(self.sensor_readings)} products")

    def _needs_monitoring(self, product_data) -> bool:
        """Determine if a product needs IoT monitoring"""
        # Products that need temperature/humidity monitoring
        monitored_categories = [
            "supplements",
            "essential_oils",
            "vitamins",
            "organic_products",
            "first_aid",
        ]

        # Products with expiration dates
        has_expiration = product_data.shelf_life_days is not None

        return (
            product_data.subcategory in monitored_categories
            or product_data.category == "healthcare_essential"
            or has_expiration
        )

    def _get_baseline_temp(self, product_data) -> float:
        """Get baseline temperature for product storage"""
        temp_requirements = {
            "supplements": 20.0,  # Room temperature for supplements
            "essential_oils": 18.0,  # Cool room temperature
            "vitamins": 22.0,  # Standard room temperature
            "organic_products": 15.0,  # Refrigerated storage
            "first_aid": 25.0,  # Warm storage for creams/gels
        }

        return temp_requirements.get(product_data.subcategory, 20.0)

    def _get_baseline_humidity(self, product_data) -> float:
        """Get baseline humidity for product storage"""
        humidity_requirements = {
            "supplements": 45.0,  # Moderate humidity
            "essential_oils": 40.0,  # Low humidity to prevent degradation
            "vitamins": 50.0,  # Standard humidity
            "organic_products": 55.0,  # Slightly higher for natural products
            "first_aid": 60.0,  # Higher humidity tolerance
        }

        return humidity_requirements.get(product_data.subcategory, 50.0)

    async def update_sensors(self, current_time: datetime) -> None:
        """Update all sensor readings"""
        if not self._initialized:
            return

        for product_id, sensor_data in self.sensor_readings.items():
            # Update readings with small random variations
            product_data = self.product_catalog.product_data[product_id]

            # Temperature variation (±2°C)
            temp_variation = random.uniform(-2.0, 2.0)
            sensor_data["temperature_celsius"] = max(
                5.0, min(35.0, sensor_data["temperature_celsius"] + temp_variation)
            )

            # Humidity variation (±5%)
            humidity_variation = random.uniform(-5.0, 5.0)
            sensor_data["humidity_percent"] = max(
                20.0, min(80.0, sensor_data["humidity_percent"] + humidity_variation)
            )

            sensor_data["last_reading"] = current_time

            # Check for alerts
            await self._check_sensor_alerts(product_id, sensor_data, product_data)

        # Random sensor failures (small chance)
        if random.random() < 0.02:  # 2% chance per update
            failed_sensor = random.choice(list(self.sensor_readings.keys()))
            self.sensor_readings[failed_sensor]["sensor_health"] = "offline"
            logger.warning(f"IoT sensor for {failed_sensor} went offline")

    async def _check_sensor_alerts(
        self, product_id: str, sensor_data: Dict[str, Any], product_data
    ) -> None:
        """Check sensor readings for alert conditions"""
        alerts = []

        # Temperature alerts
        baseline_temp = self._get_baseline_temp(product_data)
        temp_deviation = abs(sensor_data["temperature_celsius"] - baseline_temp)

        if temp_deviation > 5.0:  # More than 5°C deviation
            severity = "critical" if temp_deviation > 10.0 else "warning"
            alerts.append(
                {
                    "type": "temperature",
                    "severity": severity,
                    "message": f"Temperature {sensor_data['temperature_celsius']:.1f}°C "
                    f"(baseline: {baseline_temp:.1f}°C)",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Humidity alerts
        baseline_humidity = self._get_baseline_humidity(product_data)
        humidity_deviation = abs(sensor_data["humidity_percent"] - baseline_humidity)

        if humidity_deviation > 10.0:  # More than 10% deviation
            severity = "critical" if humidity_deviation > 20.0 else "warning"
            alerts.append(
                {
                    "type": "humidity",
                    "severity": severity,
                    "message": f"Humidity {sensor_data['humidity_percent']:.1f}% "
                    f"(baseline: {baseline_humidity:.1f}%)",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Expiration alerts (if product has shelf life)
        if product_data.shelf_life_days:
            # Simplified: assume products are at 80% of shelf life
            days_remaining = int(
                product_data.shelf_life_days * 0.2
            )  # 20% of shelf life left

            if days_remaining <= 30:  # Less than 30 days remaining
                alerts.append(
                    {
                        "type": "expiration",
                        "severity": "warning" if days_remaining > 7 else "critical",
                        "message": f"Product expires in {days_remaining} days",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Update active alerts
        sensor_data["alerts_active"] = alerts

        # Add to global alerts if any new alerts
        for alert in alerts:
            if not any(
                a["type"] == alert["type"] and a.get("timestamp") == alert["timestamp"]
                for a in self.alerts[-10:]
            ):  # Check last 10 alerts
                self.alerts.append(
                    {
                        "product_id": product_id,
                        "product_name": self.product_catalog.products[product_id].name,
                        **alert,
                    }
                )

    async def get_current_readings(self) -> Dict[str, Any]:
        """Get current sensor readings for all monitored products"""
        return {
            "sensor_readings": self.sensor_readings.copy(),
            "total_sensors": len(self.sensor_readings),
            "online_sensors": len(
                [
                    s
                    for s in self.sensor_readings.values()
                    if s["sensor_health"] == "online"
                ]
            ),
            "active_alerts": len(
                [s for s in self.sensor_readings.values() if s["alerts_active"]]
            ),
        }

    async def get_sensor_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent sensor alerts"""
        return self.alerts[-limit:] if self.alerts else []

    async def get_product_sensor_data(
        self, product_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get sensor data for a specific product"""
        return self.sensor_readings.get(product_id)

    async def simulate_sensor_failure(self, product_id: str) -> bool:
        """Simulate a sensor failure for testing"""
        if product_id in self.sensor_readings:
            self.sensor_readings[product_id]["sensor_health"] = "offline"
            logger.info(f"Simulated sensor failure for {product_id}")
            return True
        return False

    async def repair_sensor(self, product_id: str) -> bool:
        """Repair a failed sensor"""
        if product_id in self.sensor_readings:
            self.sensor_readings[product_id]["sensor_health"] = "online"
            logger.info(f"Repaired sensor for {product_id}")
            return True
        return False

    async def get_sensor_health_summary(self) -> Dict[str, Any]:
        """Get summary of sensor health across all products"""
        total_sensors = len(self.sensor_readings)
        online_sensors = len(
            [s for s in self.sensor_readings.values() if s["sensor_health"] == "online"]
        )
        offline_sensors = total_sensors - online_sensors

        alerts_by_type = {}
        for sensor_data in self.sensor_readings.values():
            for alert in sensor_data["alerts_active"]:
                alert_type = alert["type"]
                alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1

        return {
            "total_sensors": total_sensors,
            "online_sensors": online_sensors,
            "offline_sensors": offline_sensors,
            "uptime_percentage": (online_sensors / total_sensors * 100)
            if total_sensors > 0
            else 0,
            "alerts_by_type": alerts_by_type,
            "total_active_alerts": sum(alerts_by_type.values()),
        }

    async def reset(self) -> None:
        """Reset IoT sensors to initial state"""
        self.alerts = []

        # Reset all sensors to baseline values
        for product_id, sensor_data in self.sensor_readings.items():
            product_data = self.product_catalog.product_data[product_id]
            sensor_data["temperature_celsius"] = self._get_baseline_temp(product_data)
            sensor_data["humidity_percent"] = self._get_baseline_humidity(product_data)
            sensor_data["last_reading"] = datetime.now()
            sensor_data["alerts_active"] = []
            sensor_data["sensor_health"] = "online"

        logger.info("IoT sensor simulator reset to initial state")
