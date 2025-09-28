#!/usr/bin/env python3
"""
WebSocket client example for the AutoOps Retail Optimization Dashboard.

This script demonstrates how to connect to the dashboard WebSocket endpoints
and receive real-time updates from the system.
"""

import asyncio
import json
import websockets
from datetime import datetime


async def connect_to_live_updates():
    """Connect to the live updates WebSocket endpoint."""
    uri = "ws://localhost:8000/api/dashboard/ws/live-updates"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to live updates at {datetime.now()}")
            
            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                print(f"[LIVE UPDATE] {data['type']}: {data.get('message', 'Data received')}")
                
                if data['type'] == 'metrics_update':
                    metrics = data['data']['system_metrics']
                    print(f"  Revenue: ${metrics['total_revenue']:,.2f}")
                    print(f"  Profit: ${metrics['total_profit']:,.2f}")
                    print(f"  Inventory Turnover: {metrics['inventory_turnover']}")
                
    except websockets.exceptions.ConnectionClosed:
        print("Connection to live updates closed")
    except Exception as e:
        print(f"Error connecting to live updates: {e}")


async def connect_to_agent_status(agent_id: str):
    """Connect to agent-specific status updates."""
    uri = f"ws://localhost:8000/api/dashboard/ws/agents/{agent_id}/status"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {agent_id} status at {datetime.now()}")
            
            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                print(f"[{agent_id.upper()}] {data['type']}")
                
                if 'data' in data:
                    agent_data = data['data']
                    print(f"  Status: {agent_data['status']}")
                    print(f"  Activity: {agent_data.get('current_activity', 'None')}")
                    print(f"  Decisions: {agent_data['decisions_count']}")
                    print(f"  Success Rate: {agent_data['success_rate']:.1f}%")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"Connection to {agent_id} status closed")
    except Exception as e:
        print(f"Error connecting to {agent_id} status: {e}")


async def connect_to_metrics_stream():
    """Connect to real-time metrics stream."""
    uri = "ws://localhost:8000/api/dashboard/ws/metrics/real-time"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to metrics stream at {datetime.now()}")
            
            # Listen for messages
            async for message in websocket:
                data = json.loads(message)
                print(f"[METRICS] {data['type']}")
                
                if 'data' in data:
                    metrics = data['data']
                    print(f"  Revenue: ${metrics['total_revenue']:,.2f}")
                    print(f"  Profit: ${metrics['total_profit']:,.2f}")
                    print(f"  Optimization Score: {metrics['price_optimization_score']:.2f}")
                    print(f"  Collaboration Score: {metrics['agent_collaboration_score']:.2f}")
                
    except websockets.exceptions.ConnectionClosed:
        print("Connection to metrics stream closed")
    except Exception as e:
        print(f"Error connecting to metrics stream: {e}")


async def test_rest_endpoints():
    """Test the REST API endpoints."""
    import httpx
    
    base_url = "http://localhost:8000/api/dashboard"
    
    async with httpx.AsyncClient() as client:
        try:
            # Test agent status endpoint
            response = await client.get(f"{base_url}/agents/status")
            if response.status_code == 200:
                agents = response.json()
                print(f"\n[REST API] Found {len(agents)} agents:")
                for agent in agents:
                    print(f"  - {agent['name']}: {agent['status']}")
            
            # Test current metrics endpoint
            response = await client.get(f"{base_url}/metrics/current")
            if response.status_code == 200:
                metrics = response.json()
                print(f"\n[REST API] Current metrics:")
                print(f"  - Active alerts: {metrics['active_alerts_count']}")
                print(f"  - Recent decisions: {metrics['recent_decisions_count']}")
            
            # Test recent decisions endpoint
            response = await client.get(f"{base_url}/decisions/recent?limit=3")
            if response.status_code == 200:
                decisions = response.json()
                print(f"\n[REST API] Recent decisions ({len(decisions)}):")
                for decision in decisions:
                    print(f"  - {decision['agent_id']}: {decision['action_type']}")
            
            # Test active alerts endpoint
            response = await client.get(f"{base_url}/alerts/active")
            if response.status_code == 200:
                alerts = response.json()
                print(f"\n[REST API] Active alerts ({len(alerts)}):")
                for alert in alerts:
                    print(f"  - {alert['severity']}: {alert['title']}")
            
        except Exception as e:
            print(f"Error testing REST endpoints: {e}")


async def main():
    """Main function to demonstrate WebSocket connections."""
    print("AutoOps Retail Optimization Dashboard WebSocket Client Example")
    print("=" * 60)
    
    # Test REST endpoints first
    print("\n1. Testing REST API endpoints...")
    await test_rest_endpoints()
    
    print("\n2. Starting WebSocket connections...")
    print("Press Ctrl+C to stop\n")
    
    # Create tasks for different WebSocket connections
    tasks = [
        asyncio.create_task(connect_to_live_updates()),
        asyncio.create_task(connect_to_agent_status("pricing_agent")),
        asyncio.create_task(connect_to_metrics_stream()),
    ]
    
    try:
        # Run all WebSocket connections concurrently
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\nShutting down WebSocket connections...")
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    # Install required dependencies:
    # pip install websockets httpx
    
    print("Starting WebSocket client example...")
    print("Make sure the FastAPI server is running on localhost:8000")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Example stopped by user")
    except Exception as e:
        print(f"Error: {e}")