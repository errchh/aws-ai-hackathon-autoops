"""
Decision logging API endpoints for the autoops retail optimization system.

This module provides REST API endpoints for logging and retrieving agent decisions
for memory storage and learning purposes.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from models.core import AgentDecision, ActionType


router = APIRouter()


# Request/Response Models
class LogDecisionRequest(BaseModel):
    """Request model for logging agent decisions."""
    agent_id: str = Field(..., min_length=1, description="ID of the agent making the decision")
    action_type: ActionType = Field(..., description="Type of action being taken")
    parameters: Dict[str, Any] = Field(..., description="Action-specific parameters")
    rationale: str = Field(..., min_length=10, max_length=1000, description="Decision reasoning")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in decision")
    expected_outcome: Dict[str, Any] = Field(..., description="Expected results")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


class LogDecisionResponse(BaseModel):
    """Response model for decision logging."""
    success: bool = Field(..., description="Whether the operation succeeded")
    decision_id: UUID = Field(..., description="Generated decision ID")
    timestamp: datetime = Field(..., description="When the decision was logged")
    storage_location: str = Field(..., description="Where the decision was stored")


class DecisionQueryResponse(BaseModel):
    """Response model for decision queries."""
    decisions: List[AgentDecision] = Field(..., description="Matching decisions")
    total_count: int = Field(..., description="Total number of matching decisions")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")


class DecisionSummary(BaseModel):
    """Summary statistics for agent decisions."""
    agent_id: str = Field(..., description="Agent ID")
    total_decisions: int = Field(..., description="Total number of decisions")
    average_confidence: float = Field(..., description="Average confidence score")
    action_type_distribution: Dict[str, int] = Field(..., description="Distribution of action types")
    recent_activity: int = Field(..., description="Decisions in last 24 hours")
    success_rate: float = Field(..., description="Estimated success rate")


class SimilarDecisionRequest(BaseModel):
    """Request model for finding similar decisions."""
    current_context: Dict[str, Any] = Field(..., description="Current decision context")
    agent_id: Optional[str] = Field(None, description="Filter by specific agent")
    action_type: Optional[ActionType] = Field(None, description="Filter by action type")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")


class SimilarDecisionResult(BaseModel):
    """Result for similar decision search."""
    decision: AgentDecision = Field(..., description="Similar decision found")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    outcome_effectiveness: Optional[float] = Field(None, description="Historical effectiveness score")


class SimilarDecisionResponse(BaseModel):
    """Response model for similar decision search."""
    similar_decisions: List[SimilarDecisionResult] = Field(..., description="Similar decisions found")
    search_context: Dict[str, Any] = Field(..., description="Original search context")
    total_matches: int = Field(..., description="Total number of matches found")


# Decision Logging Endpoints
@router.post("/log", response_model=LogDecisionResponse)
async def log_decision(request: LogDecisionRequest) -> LogDecisionResponse:
    """
    Log an agent decision for memory storage and learning.
    
    This endpoint allows agents to store their decisions with full context
    for future reference and learning from past experiences.
    """
    try:
        # Create decision object
        decision = AgentDecision(
            agent_id=request.agent_id,
            action_type=request.action_type,
            parameters=request.parameters,
            rationale=request.rationale,
            confidence_score=request.confidence_score,
            expected_outcome=request.expected_outcome,
            context=request.context or {}
        )
        
        # In real implementation: await store_decision_in_vector_db(decision)
        # For now, simulate storage
        storage_location = f"chromadb://decisions/{decision.id}"
        
        return LogDecisionResponse(
            success=True,
            decision_id=decision.id,
            timestamp=decision.timestamp,
            storage_location=storage_location
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log decision: {str(e)}"
        )


@router.get("/query", response_model=DecisionQueryResponse)
async def query_decisions(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    action_type: Optional[ActionType] = Query(None, description="Filter by action type"),
    start_date: Optional[datetime] = Query(None, description="Start date for time range"),
    end_date: Optional[datetime] = Query(None, description="End date for time range"),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results to return")
) -> DecisionQueryResponse:
    """
    Query historical agent decisions with filtering options.
    
    This endpoint allows agents to search through historical decisions
    to learn from past experiences and outcomes.
    """
    try:
        import time
        start_time = time.time()
        
        # Simulate decision retrieval (in real implementation, this would query ChromaDB)
        simulated_decisions = [
            AgentDecision(
                id=uuid4(),
                agent_id="pricing_agent",
                action_type=ActionType.PRICE_ADJUSTMENT,
                parameters={
                    "product_id": "SKU123456",
                    "new_price": 19.99,
                    "previous_price": 22.99
                },
                rationale="Reducing price by 13% to increase demand and clear slow-moving inventory",
                confidence_score=0.85,
                expected_outcome={
                    "demand_increase_percentage": 25,
                    "inventory_turnover_days": 14,
                    "profit_impact": -150.00
                },
                context={
                    "inventory_level": 150,
                    "days_since_last_sale": 7,
                    "competitor_prices": [18.99, 21.49, 20.99]
                }
            ),
            AgentDecision(
                id=uuid4(),
                agent_id="inventory_agent",
                action_type=ActionType.INVENTORY_RESTOCK,
                parameters={
                    "product_id": "SKU789012",
                    "restock_quantity": 100,
                    "urgency": "high"
                },
                rationale="Stock level below safety threshold, high demand forecast for next week",
                confidence_score=0.92,
                expected_outcome={
                    "stockout_prevention": "high_probability",
                    "service_level_improvement": 0.15
                },
                context={
                    "current_stock": 8,
                    "reorder_point": 15,
                    "forecast_demand": 45
                }
            )
        ]
        
        # Apply filters
        filtered_decisions = simulated_decisions
        
        if agent_id:
            filtered_decisions = [d for d in filtered_decisions if d.agent_id == agent_id]
        
        if action_type:
            filtered_decisions = [d for d in filtered_decisions if d.action_type == action_type]
        
        if start_date:
            filtered_decisions = [d for d in filtered_decisions if d.timestamp >= start_date]
        
        if end_date:
            filtered_decisions = [d for d in filtered_decisions if d.timestamp <= end_date]
        
        if min_confidence is not None:
            filtered_decisions = [d for d in filtered_decisions if d.confidence_score >= min_confidence]
        
        # Apply limit
        limited_decisions = filtered_decisions[:limit]
        
        query_time_ms = (time.time() - start_time) * 1000
        
        return DecisionQueryResponse(
            decisions=limited_decisions,
            total_count=len(filtered_decisions),
            query_time_ms=round(query_time_ms, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query decisions: {str(e)}"
        )


@router.post("/similar", response_model=SimilarDecisionResponse)
async def find_similar_decisions(request: SimilarDecisionRequest) -> SimilarDecisionResponse:
    """
    Find similar historical decisions based on current context.
    
    This endpoint uses vector similarity search to find past decisions
    that were made in similar circumstances, helping agents learn from experience.
    """
    try:
        # Simulate vector similarity search (in real implementation, this would use ChromaDB embeddings)
        # For demonstration, we'll return some mock similar decisions
        
        similar_decisions = [
            SimilarDecisionResult(
                decision=AgentDecision(
                    id=uuid4(),
                    agent_id="pricing_agent",
                    action_type=ActionType.PRICE_ADJUSTMENT,
                    parameters={
                        "product_id": "SKU123456",
                        "new_price": 20.99,
                        "previous_price": 24.99
                    },
                    rationale="Similar market conditions, reducing price to stimulate demand",
                    confidence_score=0.88,
                    expected_outcome={
                        "demand_increase_percentage": 20,
                        "revenue_impact": 500.00
                    },
                    context={
                        "inventory_level": 140,
                        "competitor_average": 21.50,
                        "seasonal_factor": 0.95
                    }
                ),
                similarity_score=0.89,
                outcome_effectiveness=0.82
            ),
            SimilarDecisionResult(
                decision=AgentDecision(
                    id=uuid4(),
                    agent_id="pricing_agent",
                    action_type=ActionType.MARKDOWN_APPLICATION,
                    parameters={
                        "product_id": "SKU345678",
                        "discount_percentage": 15,
                        "duration_hours": 72
                    },
                    rationale="Applying temporary markdown to move slow inventory",
                    confidence_score=0.75,
                    expected_outcome={
                        "inventory_reduction": 60,
                        "margin_impact": -200.00
                    },
                    context={
                        "inventory_age_days": 45,
                        "demand_trend": "declining",
                        "storage_cost": 2.50
                    }
                ),
                similarity_score=0.73,
                outcome_effectiveness=0.91
            )
        ]
        
        # Filter by agent_id if specified
        if request.agent_id:
            similar_decisions = [
                sd for sd in similar_decisions 
                if sd.decision.agent_id == request.agent_id
            ]
        
        # Filter by action_type if specified
        if request.action_type:
            similar_decisions = [
                sd for sd in similar_decisions 
                if sd.decision.action_type == request.action_type
            ]
        
        # Filter by similarity threshold
        similar_decisions = [
            sd for sd in similar_decisions 
            if sd.similarity_score >= request.similarity_threshold
        ]
        
        # Apply limit
        limited_results = similar_decisions[:request.max_results]
        
        return SimilarDecisionResponse(
            similar_decisions=limited_results,
            search_context=request.current_context,
            total_matches=len(similar_decisions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar decisions: {str(e)}"
        )


@router.get("/summary/{agent_id}", response_model=DecisionSummary)
async def get_decision_summary(agent_id: str) -> DecisionSummary:
    """
    Get summary statistics for an agent's decision history.
    
    This endpoint provides agents with insights into their decision patterns,
    confidence levels, and performance metrics.
    """
    try:
        # Simulate decision summary calculation
        # In real implementation: summary = await calculate_agent_summary(agent_id)
        
        # Generate simulated statistics
        total_decisions = hash(agent_id + "total") % 500 + 100
        average_confidence = (hash(agent_id + "confidence") % 40 + 60) / 100  # 0.6 to 1.0
        recent_activity = hash(agent_id + "recent") % 20 + 5
        success_rate = (hash(agent_id + "success") % 30 + 70) / 100  # 0.7 to 1.0
        
        # Simulate action type distribution
        action_type_distribution = {
            "price_adjustment": hash(agent_id + "price") % 50 + 10,
            "markdown_application": hash(agent_id + "markdown") % 30 + 5,
            "inventory_restock": hash(agent_id + "restock") % 40 + 8,
            "promotion_creation": hash(agent_id + "promo") % 25 + 3,
            "bundle_recommendation": hash(agent_id + "bundle") % 15 + 2
        }
        
        return DecisionSummary(
            agent_id=agent_id,
            total_decisions=total_decisions,
            average_confidence=round(average_confidence, 2),
            action_type_distribution=action_type_distribution,
            recent_activity=recent_activity,
            success_rate=round(success_rate, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get decision summary: {str(e)}"
        )


@router.delete("/{decision_id}")
async def delete_decision(decision_id: UUID) -> Dict[str, str]:
    """
    Delete a specific decision from storage.
    
    This endpoint allows for cleanup of incorrect or outdated decisions
    from the agent memory system.
    """
    try:
        # In real implementation: await delete_decision_from_storage(decision_id)
        # For now, simulate deletion
        
        return {
            "message": f"Decision {decision_id} deleted successfully",
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete decision: {str(e)}"
        )