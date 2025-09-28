#!/usr/bin/env python3
"""
Focused demo to trigger specific agent actions and observe them in Langfuse.
"""

import asyncio
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, '.')

from config.langfuse_integration import get_langfuse_integration
from simulation.engine import SimulationEngine, SimulationMode

def generate_realistic_decision(agent: str, scenario: str, intensity: str):
    """Generate realistic agent-specific decisions based on scenario."""
    
    if agent == "inventory_agent":
        if scenario == "inventory_low":
            return {
                "decision_type": "restock_recommendation",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "current_stock": 25,
                    "safety_stock": 50,
                    "lead_time_days": 3
                },
                "outputs": {
                    "action": "emergency_restock",
                    "quantity": 200,
                    "priority": "high",
                    "supplier": "primary_supplier",
                    "expected_delivery": "2025-10-01"
                },
                "confidence": 0.95,
                "reasoning": "Stock level (25 units) is below safety threshold (50 units). High intensity requires immediate action to prevent stockouts.",
                "summary": "Emergency restock: 200 units, delivery Oct 1st"
            }
        elif scenario == "demand_spike":
            return {
                "decision_type": "demand_forecast_adjustment",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "current_demand": 150,
                    "forecasted_spike": 280
                },
                "outputs": {
                    "action": "increase_stock_target",
                    "new_target": 350,
                    "buffer_increase": 70,
                    "reorder_point": 120
                },
                "confidence": 0.88,
                "reasoning": "Demand spike detected. Adjusting stock targets to handle 280% increase in demand with safety buffer.",
                "summary": "Increased stock target to 350 units with 70-unit buffer"
            }
        elif scenario == "competitor_price_change":
            return {
                "decision_type": "inventory_positioning",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "competitor_price_drop": 0.15
                },
                "outputs": {
                    "action": "maintain_stock_levels",
                    "rationale": "price_war_preparation",
                    "stock_adjustment": 0
                },
                "confidence": 0.82,
                "reasoning": "Competitor 15% price drop may increase demand. Maintaining current levels to avoid overstock risk.",
                "summary": "Maintaining stock levels, monitoring demand response"
            }
        else:  # social_trend
            return {
                "decision_type": "trend_inventory_adjustment",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "trend_products": ["protein_powder", "fitness_gear"]
                },
                "outputs": {
                    "action": "selective_stock_increase",
                    "affected_products": ["protein_powder", "fitness_gear"],
                    "increase_percentage": 0.25
                },
                "confidence": 0.78,
                "reasoning": "Social trend affecting fitness products. Increasing stock by 25% for trending items.",
                "summary": "Increased fitness product stock by 25% due to social trend"
            }
    
    elif agent == "pricing_agent":
        if scenario == "inventory_low":
            return {
                "decision_type": "scarcity_pricing",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "current_price": 29.99,
                    "stock_level": 25
                },
                "outputs": {
                    "action": "increase_price",
                    "new_price": 34.99,
                    "increase_percentage": 0.167,
                    "duration": "until_restock"
                },
                "confidence": 0.91,
                "reasoning": "Low inventory allows premium pricing to manage demand and maximize revenue per unit.",
                "summary": "Increased price to $34.99 (16.7% up) until restock"
            }
        elif scenario == "demand_spike":
            return {
                "decision_type": "dynamic_pricing",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "demand_multiplier": 2.8
                },
                "outputs": {
                    "action": "moderate_price_increase",
                    "new_price": 32.99,
                    "increase_percentage": 0.10,
                    "elasticity_consideration": True
                },
                "confidence": 0.86,
                "reasoning": "High demand allows 10% price increase while maintaining volume. Balancing revenue and market share.",
                "summary": "Moderate price increase to $32.99 (10% up) for demand spike"
            }
        elif scenario == "competitor_price_change":
            return {
                "decision_type": "competitive_pricing",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "competitor_price": 25.49,
                    "our_price": 29.99
                },
                "outputs": {
                    "action": "strategic_price_match",
                    "new_price": 26.99,
                    "competitive_gap": 1.50,
                    "margin_impact": -0.08
                },
                "confidence": 0.84,
                "reasoning": "Competitor dropped to $25.49. Matching within $1.50 to stay competitive while preserving some margin.",
                "summary": "Price match to $26.99, staying within $1.50 of competitor"
            }
        else:  # social_trend
            return {
                "decision_type": "trend_pricing",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "trend_boost": 0.35
                },
                "outputs": {
                    "action": "premium_positioning",
                    "new_price": 33.99,
                    "premium_percentage": 0.133,
                    "positioning": "trend_leader"
                },
                "confidence": 0.79,
                "reasoning": "Social trend creates premium opportunity. Positioning as trend leader with 13.3% premium.",
                "summary": "Premium pricing at $33.99 (13.3% up) for trend positioning"
            }
    
    else:  # promotion_agent
        if scenario == "inventory_low":
            return {
                "decision_type": "demand_management",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "stock_level": 25
                },
                "outputs": {
                    "action": "pause_promotions",
                    "affected_campaigns": ["flash_sale", "bulk_discount"],
                    "reason": "inventory_conservation"
                },
                "confidence": 0.93,
                "reasoning": "Low inventory requires demand reduction. Pausing promotions to conserve stock for regular customers.",
                "summary": "Paused promotions to conserve low inventory"
            }
        elif scenario == "demand_spike":
            return {
                "decision_type": "momentum_promotion",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "spike_magnitude": 2.8
                },
                "outputs": {
                    "action": "amplify_promotion",
                    "campaign_type": "viral_boost",
                    "discount": 0.05,
                    "duration_hours": 6
                },
                "confidence": 0.87,
                "reasoning": "Demand spike creates viral opportunity. Small 5% discount for 6 hours to maximize momentum.",
                "summary": "Viral boost campaign: 5% off for 6 hours to amplify spike"
            }
        elif scenario == "competitor_price_change":
            return {
                "decision_type": "competitive_promotion",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "competitor_advantage": 0.15
                },
                "outputs": {
                    "action": "value_add_promotion",
                    "offer_type": "bundle_deal",
                    "additional_value": "free_shipping",
                    "effective_discount": 0.12
                },
                "confidence": 0.81,
                "reasoning": "Instead of price matching, offering bundle + free shipping creates 12% effective discount with higher perceived value.",
                "summary": "Bundle deal + free shipping (12% effective discount)"
            }
        else:  # social_trend
            return {
                "decision_type": "trend_amplification",
                "inputs": {
                    "scenario": scenario,
                    "intensity": intensity,
                    "trend_engagement": 0.65
                },
                "outputs": {
                    "action": "social_campaign",
                    "campaign_type": "user_generated_content",
                    "incentive": "contest_entry",
                    "hashtag": "#FitnessGoals2025"
                },
                "confidence": 0.76,
                "reasoning": "Social trend opportunity. UGC contest with #FitnessGoals2025 to amplify organic engagement.",
                "summary": "UGC contest campaign with #FitnessGoals2025 hashtag"
            }
    
    # Fallback for unknown combinations
    return {
        "decision_type": "default_response",
        "inputs": {"scenario": scenario, "intensity": intensity},
        "outputs": {"action": "monitor_and_assess"},
        "confidence": 0.70,
        "reasoning": f"{agent} is monitoring {scenario} situation and will respond as needed.",
        "summary": f"Monitoring {scenario} situation"
    }

async def run_agent_scenario_demo():
    """Run a focused demo showing agent actions in Langfuse."""
    print("🚀 Starting Agent Action Demo for Langfuse Observation")
    print("=" * 60)
    
    # Get Langfuse integration
    langfuse = get_langfuse_integration()
    
    # Initialize simulation engine
    print("📦 Initializing simulation engine...")
    engine = SimulationEngine(mode=SimulationMode.DEMO)
    await engine.initialize()
    
    # Start simulation
    print("▶️  Starting simulation...")
    await engine.start_simulation()
    
    # Wait a moment for initial setup
    await asyncio.sleep(2)
    
    # Trigger specific scenarios to generate agent actions
    scenarios = [
        ("demand_spike", "high"),
        ("competitor_price_change", "medium"),
        ("inventory_low", "high"),
        ("social_trend", "medium")
    ]
    
    print("\n🎯 Triggering Agent Scenarios:")
    for scenario, intensity in scenarios:
        print(f"   Triggering: {scenario} with {intensity} intensity")
        
        # Create a trace for this scenario
        trace_data = {
            "event_type": scenario,
            "intensity": intensity,
            "timestamp": datetime.now().isoformat(),
            "trigger_source": "manual_demo"
        }
        
        trace_id = langfuse.create_simulation_trace(trace_data)
        print(f"   📊 Created trace: {trace_id}")
        
        # Simulate agent responses
        agents = ["pricing_agent", "inventory_agent", "promotion_agent"]
        
        for agent in agents:
            # Start agent span
            span_id = langfuse.start_agent_span(
                agent_id=agent,
                operation=f"respond_to_{scenario}",
                parent_trace_id=trace_id,
                input_data={"scenario": scenario, "intensity": intensity}
            )
            
            # Generate realistic agent-specific decisions
            decision = generate_realistic_decision(agent, scenario, intensity)
            
            langfuse.log_agent_decision(agent, decision)
            
            # End span with results
            langfuse.end_agent_span(
                span_id,
                outcome=decision["outputs"]
            )
            
            print(f"     🤖 {agent}: {decision['summary']}")
        
        # Finalize trace
        langfuse.finalize_trace(trace_id, {
            "status": "completed",
            "agents_involved": len(agents),
            "scenario_resolved": True
        })
        
        # Brief pause between scenarios
        await asyncio.sleep(1)
    
    # Get current simulation state
    print("\n📊 Current Simulation State:")
    state = await engine.get_current_state()
    for key, value in state.items():
        print(f"   {key}: {value}")
    
    # Stop simulation
    print("\n⏹️  Stopping simulation...")
    await engine.stop_simulation()
    
    print("\n✅ Demo completed! Check your Langfuse dashboard at:")
    print("   https://us.cloud.langfuse.com")
    print("\nYou should see:")
    print("   • Multiple simulation traces")
    print("   • Agent spans for each scenario")
    print("   • Agent decisions and outcomes")
    print("   • Collaboration patterns between agents")

if __name__ == "__main__":
    asyncio.run(run_agent_scenario_demo())