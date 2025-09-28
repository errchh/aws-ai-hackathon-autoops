"""
Inventory API endpoints for the autoops retail optimization system.

This module provides REST API endpoints for inventory-related operations
that the Inventory Agent can execute.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from langfuse import observe

from models.core import Product, AgentDecision, ActionType
from config.langfuse_integration import get_langfuse_integration


router = APIRouter()


# Request/Response Models
class UpdateStockRequest(BaseModel):
    """Request model for updating product stock levels."""

    product_id: str = Field(..., description="Product ID to update")
    new_stock_level: int = Field(..., ge=0, description="New stock level")
    reason: str = Field(
        ..., min_length=1, max_length=500, description="Reason for stock update"
    )
    agent_id: str = Field(..., description="ID of the agent making the change")
    source: str = Field(
        default="manual", description="Source of stock update (manual, iot, delivery)"
    )


class UpdateStockResponse(BaseModel):
    """Response model for stock update operation."""

    success: bool = Field(..., description="Whether the operation succeeded")
    product_id: str = Field(..., description="Product ID that was updated")
    previous_stock: int = Field(..., description="Previous stock level")
    new_stock: int = Field(..., description="New stock level")
    decision_id: UUID = Field(..., description="Decision tracking ID")
    timestamp: datetime = Field(..., description="When the change was made")


class CreateRestockAlertRequest(BaseModel):
    """Request model for creating restock alerts."""

    product_id: str = Field(..., description="Product ID needing restock")
    current_stock: int = Field(..., ge=0, description="Current stock level")
    recommended_quantity: int = Field(
        ..., gt=0, description="Recommended restock quantity"
    )
    urgency: str = Field(..., description="Alert urgency (low/medium/high/critical)")
    reason: str = Field(
        ..., min_length=1, max_length=500, description="Reason for restock alert"
    )
    agent_id: str = Field(..., description="ID of the agent creating alert")


class CreateRestockAlertResponse(BaseModel):
    """Response model for restock alert creation."""

    success: bool = Field(..., description="Whether the operation succeeded")
    alert_id: UUID = Field(..., description="Generated alert ID")
    product_id: str = Field(..., description="Product ID for the alert")
    recommended_quantity: int = Field(..., description="Recommended restock quantity")
    estimated_cost: float = Field(..., description="Estimated restock cost")
    expected_delivery: datetime = Field(..., description="Expected delivery date")
    decision_id: UUID = Field(..., description="Decision tracking ID")


class DemandForecastRequest(BaseModel):
    """Request model for demand forecasting."""

    product_id: str = Field(..., description="Product ID to forecast")
    forecast_days: int = Field(
        ..., ge=1, le=365, description="Number of days to forecast"
    )
    include_seasonality: bool = Field(
        default=True, description="Include seasonal patterns"
    )
    include_trends: bool = Field(default=True, description="Include trend analysis")


class DemandForecastPoint(BaseModel):
    """Individual forecast data point."""

    date: datetime = Field(..., description="Forecast date")
    expected_demand: float = Field(..., ge=0, description="Expected demand quantity")
    confidence_interval_lower: float = Field(..., description="Lower confidence bound")
    confidence_interval_upper: float = Field(..., description="Upper confidence bound")
    confidence_level: float = Field(
        ..., ge=0, le=1, description="Confidence level (0-1)"
    )


class DemandForecastResponse(BaseModel):
    """Response model for demand forecasting."""

    product_id: str = Field(..., description="Product ID forecasted")
    forecast_period_days: int = Field(..., description="Forecast period in days")
    total_expected_demand: float = Field(
        ..., description="Total expected demand over period"
    )
    average_daily_demand: float = Field(..., description="Average daily demand")
    demand_variance: float = Field(..., description="Demand variance measure")
    seasonality_detected: bool = Field(
        ..., description="Whether seasonality was detected"
    )
    trend_direction: str = Field(
        ..., description="Trend direction (increasing/stable/decreasing)"
    )
    forecast_points: List[DemandForecastPoint] = Field(
        ..., description="Detailed forecast data"
    )
    model_accuracy: float = Field(
        ..., ge=0, le=1, description="Historical model accuracy"
    )


class InventoryAnalysis(BaseModel):
    """Model for inventory analysis results."""

    product_id: str = Field(..., description="Product ID analyzed")
    current_stock: int = Field(..., description="Current stock level")
    reorder_point: int = Field(..., description="Recommended reorder point")
    safety_stock: int = Field(..., description="Recommended safety stock")
    optimal_order_quantity: int = Field(..., description="Optimal order quantity")
    days_of_supply: float = Field(..., description="Days of supply at current demand")
    stockout_risk: float = Field(..., ge=0, le=1, description="Risk of stockout (0-1)")
    carrying_cost_daily: float = Field(..., description="Daily carrying cost")
    recommendations: List[str] = Field(..., description="Inventory recommendations")


# Inventory Endpoints
@observe(name="inventory_agent_update_stock")
@router.post("/update-stock", response_model=UpdateStockResponse)
async def update_stock(request: UpdateStockRequest) -> UpdateStockResponse:
    """
    Update the stock level of a specific product.

    This endpoint allows the Inventory Agent to update stock levels based on
    deliveries, sales, IoT sensor data, or manual counts.
    """
    try:
        # Get Langfuse integration service
        langfuse_service = get_langfuse_integration()

        # Generate decision ID for tracking
        decision_id = uuid4()
        timestamp = datetime.now(timezone.utc)

        # Start span for stock validation
        span_id = langfuse_service.start_agent_span(
            agent_id=request.agent_id,
            operation="stock_validation",
            input_data={
                "product_id": request.product_id,
                "requested_stock_level": request.new_stock_level,
                "source": request.source,
            },
        )

        # Simulate retrieving current stock data
        # In real implementation: product = await get_product(request.product_id)
        current_stock = 45  # Simulated current stock

        # Validate stock update is reasonable
        stock_change = abs(request.new_stock_level - current_stock)
        if stock_change > 1000:  # Arbitrary large change threshold
            # End span with error if span_id exists
            if span_id:
                langfuse_service.end_agent_span(
                    span_id,
                    error=ValueError(
                        f"Stock change of {stock_change} units seems unusually large"
                    ),
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Stock change of {stock_change} units seems unusually large. Please verify.",
            )

        # End validation span successfully
        langfuse_service.end_agent_span(
            span_id,
            outcome={
                "current_stock": current_stock,
                "stock_change": stock_change,
                "validation_passed": True,
            },
        )

        # Log the decision for agent memory
        decision = AgentDecision(
            id=decision_id,
            agent_id=request.agent_id,
            action_type=ActionType.STOCK_ALERT,
            parameters={
                "product_id": request.product_id,
                "new_stock_level": request.new_stock_level,
                "previous_stock": current_stock,
                "source": request.source,
                "reason": request.reason,
            },
            rationale=request.reason,
            confidence_score=0.95 if request.source == "iot" else 0.85,
            expected_outcome={
                "inventory_accuracy": "improved",
                "stock_visibility": "updated",
                "reorder_trigger_check": "pending",
            },
        )

        # In real implementation: await store_decision(decision)
        # In real implementation: await update_product_stock(request.product_id, request.new_stock_level)

        return UpdateStockResponse(
            success=True,
            product_id=request.product_id,
            previous_stock=current_stock,
            new_stock=request.new_stock_level,
            decision_id=decision_id,
            timestamp=timestamp,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update stock: {str(e)}",
        )


@observe(name="inventory_agent_create_restock_alert")
@router.post("/create-restock-alert", response_model=CreateRestockAlertResponse)
async def create_restock_alert(
    request: CreateRestockAlertRequest,
) -> CreateRestockAlertResponse:
    """
    Create a restock alert for a product that needs replenishment.

    This endpoint allows the Inventory Agent to generate restock alerts
    with recommended quantities and timing.
    """
    try:
        # Get Langfuse integration service
        langfuse_service = get_langfuse_integration()

        # Generate IDs for tracking
        alert_id = uuid4()
        decision_id = uuid4()
        timestamp = datetime.now(timezone.utc)

        # Start span for cost calculation
        span_id = langfuse_service.start_agent_span(
            agent_id=request.agent_id,
            operation="cost_calculation",
            input_data={
                "product_id": request.product_id,
                "recommended_quantity": request.recommended_quantity,
                "urgency": request.urgency,
            },
        )

        # Simulate product cost and lead time data
        product_cost = 12.50  # Simulated product cost
        supplier_lead_time = 7  # Simulated lead time in days

        # Calculate estimated cost and delivery date
        estimated_cost = request.recommended_quantity * product_cost
        expected_delivery = timestamp + timedelta(days=supplier_lead_time)

        # End cost calculation span
        if span_id:
            langfuse_service.end_agent_span(
                span_id,
                outcome={
                    "estimated_cost": estimated_cost,
                    "expected_delivery": expected_delivery.isoformat(),
                    "product_cost": product_cost,
                    "supplier_lead_time": supplier_lead_time,
                },
            )

        # Validate restock quantity is reasonable
        if request.recommended_quantity > 10000:  # Arbitrary large quantity threshold
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Recommended quantity {request.recommended_quantity} seems unusually large",
            )

        # Log the decision for agent memory
        decision = AgentDecision(
            id=decision_id,
            agent_id=request.agent_id,
            action_type=ActionType.INVENTORY_RESTOCK,
            parameters={
                "product_id": request.product_id,
                "current_stock": request.current_stock,
                "recommended_quantity": request.recommended_quantity,
                "urgency": request.urgency,
                "estimated_cost": estimated_cost,
                "expected_delivery": expected_delivery.isoformat(),
            },
            rationale=request.reason,
            confidence_score=0.88,
            expected_outcome={
                "stockout_prevention": "high_probability",
                "inventory_optimization": "improved",
                "cost_impact": estimated_cost,
            },
        )

        # In real implementation: await store_decision(decision)
        # In real implementation: await create_purchase_order(request.product_id, request.recommended_quantity)

        return CreateRestockAlertResponse(
            success=True,
            alert_id=alert_id,
            product_id=request.product_id,
            recommended_quantity=request.recommended_quantity,
            estimated_cost=estimated_cost,
            expected_delivery=expected_delivery,
            decision_id=decision_id,
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create restock alert: {str(e)}",
        )


@observe(name="inventory_agent_generate_demand_forecast")
@router.post("/demand-forecast", response_model=DemandForecastResponse)
async def generate_demand_forecast(
    request: DemandForecastRequest,
) -> DemandForecastResponse:
    """
    Generate demand forecast for a specific product.

    This endpoint provides the Inventory Agent with demand predictions
    to inform inventory planning and restock decisions.
    """
    try:
        # Get Langfuse integration service
        langfuse_service = get_langfuse_integration()

        # Start span for demand forecasting
        forecast_span_id = langfuse_service.start_agent_span(
            agent_id="inventory_agent",
            operation="demand_forecasting",
            input_data={
                "product_id": request.product_id,
                "forecast_days": request.forecast_days,
                "include_seasonality": request.include_seasonality,
                "include_trends": request.include_trends,
            },
        )

        # Simulate demand forecasting (in real implementation, this would use ML models)
        base_daily_demand = 8.5  # Simulated base daily demand
        demand_variance = 2.1  # Simulated demand variance

        # Start span for seasonal analysis
        seasonal_span_id = langfuse_service.start_agent_span(
            agent_id="inventory_agent",
            operation="seasonal_analysis",
            input_data={
                "include_seasonality": request.include_seasonality,
                "forecast_days": request.forecast_days,
            },
        )

        # Generate forecast points
        forecast_points = []
        total_demand = 0

        for day in range(request.forecast_days):
            forecast_date = datetime.now(timezone.utc) + timedelta(days=day + 1)

            # Simulate demand with some randomness and seasonality
            seasonal_factor = 1.0
            if request.include_seasonality:
                # Simple seasonal pattern (higher on weekends)
                if forecast_date.weekday() >= 5:  # Weekend
                    seasonal_factor = 1.3
                else:
                    seasonal_factor = 0.9

            # End seasonal analysis span
            if seasonal_span_id:
                langfuse_service.end_agent_span(
                    seasonal_span_id,
                    outcome={
                        "seasonal_factor": seasonal_factor,
                        "is_weekend": forecast_date.weekday() >= 5,
                    },
                )

            # Start span for trend analysis
            trend_span_id = langfuse_service.start_agent_span(
                agent_id="inventory_agent",
                operation="trend_analysis",
                input_data={"day": day, "include_trends": request.include_trends},
            )

            trend_factor = 1.0
            if request.include_trends:
                # Simple upward trend
                trend_factor = 1.0 + (day * 0.002)  # 0.2% daily growth

            expected_demand = base_daily_demand * seasonal_factor * trend_factor
            total_demand += expected_demand

            # End trend analysis span
            if trend_span_id:
                langfuse_service.end_agent_span(
                    trend_span_id,
                    outcome={
                        "trend_factor": trend_factor,
                        "expected_demand": expected_demand,
                    },
                )

            # Calculate confidence intervals (simplified)
            confidence_lower = max(0, expected_demand - demand_variance)
            confidence_upper = expected_demand + demand_variance

            forecast_points.append(
                DemandForecastPoint(
                    date=forecast_date,
                    expected_demand=round(expected_demand, 2),
                    confidence_interval_lower=round(confidence_lower, 2),
                    confidence_interval_upper=round(confidence_upper, 2),
                    confidence_level=0.85,
                )
            )

        # Calculate summary statistics
        average_daily_demand = total_demand / request.forecast_days

        # Determine trend direction
        first_week_avg = sum(fp.expected_demand for fp in forecast_points[:7]) / 7
        last_week_avg = sum(fp.expected_demand for fp in forecast_points[-7:]) / 7

        if last_week_avg > first_week_avg * 1.05:
            trend_direction = "increasing"
        elif last_week_avg < first_week_avg * 0.95:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        # End demand forecasting span
        if forecast_span_id:
            langfuse_service.end_agent_span(
                forecast_span_id,
                outcome={
                    "total_expected_demand": round(total_demand, 2),
                    "average_daily_demand": round(average_daily_demand, 2),
                    "trend_direction": trend_direction,
                    "forecast_points_count": len(forecast_points),
                    "model_accuracy": 0.82,
                },
            )

        return DemandForecastResponse(
            product_id=request.product_id,
            forecast_period_days=request.forecast_days,
            total_expected_demand=round(total_demand, 2),
            average_daily_demand=round(average_daily_demand, 2),
            demand_variance=demand_variance,
            seasonality_detected=request.include_seasonality,
            trend_direction=trend_direction,
            forecast_points=forecast_points,
            model_accuracy=0.82,  # Simulated historical accuracy
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate demand forecast: {str(e)}",
        )


@observe(name="inventory_agent_get_inventory_analysis")
@router.get("/analysis/{product_id}", response_model=InventoryAnalysis)
async def get_inventory_analysis(product_id: str) -> InventoryAnalysis:
    """
    Get comprehensive inventory analysis for a specific product.

    This endpoint provides the Inventory Agent with detailed analysis
    including reorder points, safety stock, and optimization recommendations.
    """
    try:
        # Get Langfuse integration service
        langfuse_service = get_langfuse_integration()

        # Start span for inventory analysis
        analysis_span_id = langfuse_service.start_agent_span(
            agent_id="inventory_agent",
            operation="inventory_analysis",
            input_data={"product_id": product_id},
        )

        # Simulate inventory analysis calculations
        # In real implementation: product_data = await get_product_with_history(product_id)

        current_stock = 45
        daily_demand = 8.5
        demand_variance = 2.1
        lead_time_days = 7
        carrying_cost_per_unit_per_day = 0.05

        # Start span for safety stock calculation
        safety_span_id = langfuse_service.start_agent_span(
            agent_id="inventory_agent",
            operation="safety_stock_calculation",
            input_data={
                "daily_demand": daily_demand,
                "demand_variance": demand_variance,
                "lead_time_days": lead_time_days,
                "z_score": 1.65,
            },
        )

        # Calculate safety stock (simplified formula)
        # Safety stock = Z-score * sqrt(lead_time) * demand_std_dev
        z_score = 1.65  # 95% service level
        demand_std_dev = demand_variance**0.5
        safety_stock = int(z_score * (lead_time_days**0.5) * demand_std_dev)

        # End safety stock span
        if safety_span_id:
            langfuse_service.end_agent_span(
                safety_span_id,
                outcome={
                    "safety_stock": safety_stock,
                    "demand_std_dev": demand_std_dev,
                    "z_score": z_score,
                },
            )

        # Start span for reorder point calculation
        reorder_span_id = langfuse_service.start_agent_span(
            agent_id="inventory_agent",
            operation="reorder_point_calculation",
            input_data={
                "daily_demand": daily_demand,
                "lead_time_days": lead_time_days,
                "safety_stock": safety_stock,
            },
        )

        # Calculate reorder point
        reorder_point = int((daily_demand * lead_time_days) + safety_stock)

        # End reorder point span
        if reorder_span_id:
            langfuse_service.end_agent_span(
                reorder_span_id, outcome={"reorder_point": reorder_point}
            )

        # Start span for EOQ calculation
        eoq_span_id = langfuse_service.start_agent_span(
            agent_id="inventory_agent",
            operation="eoq_calculation",
            input_data={
                "daily_demand": daily_demand,
                "order_cost": 50,
                "carrying_cost_per_unit_per_day": carrying_cost_per_unit_per_day,
            },
        )

        # Calculate optimal order quantity (simplified EOQ)
        # EOQ = sqrt(2 * annual_demand * order_cost / carrying_cost_per_unit_per_year)
        annual_demand = daily_demand * 365
        order_cost = 50  # Simulated fixed order cost
        carrying_cost_per_unit_per_year = carrying_cost_per_unit_per_day * 365
        optimal_order_quantity = int(
            (2 * annual_demand * order_cost / carrying_cost_per_unit_per_year) ** 0.5
        )

        # End EOQ span
        if eoq_span_id:
            langfuse_service.end_agent_span(
                eoq_span_id,
                outcome={
                    "optimal_order_quantity": optimal_order_quantity,
                    "annual_demand": annual_demand,
                    "carrying_cost_per_unit_per_year": carrying_cost_per_unit_per_year,
                },
            )

        # Calculate days of supply
        days_of_supply = (
            current_stock / daily_demand if daily_demand > 0 else float("inf")
        )

        # Calculate stockout risk
        if current_stock <= reorder_point:
            stockout_risk = min(1.0, (reorder_point - current_stock) / reorder_point)
        else:
            stockout_risk = 0.1  # Base risk level

        # Calculate daily carrying cost
        carrying_cost_daily = current_stock * carrying_cost_per_unit_per_day

        # Generate recommendations
        recommendations = []
        if current_stock <= reorder_point:
            recommendations.append(
                f"URGENT: Stock below reorder point. Recommend ordering {optimal_order_quantity} units."
            )
        if stockout_risk > 0.3:
            recommendations.append("HIGH RISK: Significant stockout risk detected.")
        if days_of_supply < 3:
            recommendations.append("LOW STOCK: Less than 3 days of supply remaining.")
        if current_stock > optimal_order_quantity * 2:
            recommendations.append(
                "OVERSTOCK: Consider promotional pricing to reduce excess inventory."
            )
        if not recommendations:
            recommendations.append(
                "OPTIMAL: Inventory levels are within acceptable ranges."
            )

        # End inventory analysis span
        if analysis_span_id:
            langfuse_service.end_agent_span(
                analysis_span_id,
                outcome={
                    "recommendations_count": len(recommendations),
                    "stockout_risk": round(stockout_risk, 3),
                    "days_of_supply": round(days_of_supply, 1),
                    "recommendations": recommendations,
                },
            )

        return InventoryAnalysis(
            product_id=product_id,
            current_stock=current_stock,
            reorder_point=reorder_point,
            safety_stock=safety_stock,
            optimal_order_quantity=optimal_order_quantity,
            days_of_supply=round(days_of_supply, 1),
            stockout_risk=round(stockout_risk, 3),
            carrying_cost_daily=round(carrying_cost_daily, 2),
            recommendations=recommendations,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze inventory: {str(e)}",
        )
