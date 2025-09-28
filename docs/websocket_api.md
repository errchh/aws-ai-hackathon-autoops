# WebSocket API Documentation

## Overview

The AutoOps Retail Optimization system provides real-time WebSocket endpoints for the React.js dashboard to receive live updates about agent activities, system metrics, and alerts. This document describes the available WebSocket endpoints and their usage.

## WebSocket Endpoints

### 1. Live Updates Stream
**Endpoint:** `ws://localhost:8000/api/dashboard/ws/live-updates`

Provides general system updates including metrics changes, agent status updates, and system events.

**Message Types:**
- `connection_established`: Sent when connection is first established
- `metrics_update`: Periodic system metrics updates (every 5 seconds)
- `agent_intervention`: Broadcast when manual intervention occurs
- `system_event`: General system events and notifications

**Example Messages:**
```json
{
  "type": "connection_established",
  "timestamp": "2024-01-15T10:30:00Z",
  "message": "Connected to live updates stream"
}

{
  "type": "metrics_update",
  "timestamp": "2024-01-15T10:30:05Z",
  "data": {
    "timestamp": "2024-01-15T10:30:05Z",
    "system_metrics": {
      "total_revenue": 125000.50,
      "total_profit": 45000.25,
      "inventory_turnover": 8.5,
      "stockout_incidents": 3,
      "waste_reduction_percentage": 15.2,
      "price_optimization_score": 0.87,
      "promotion_effectiveness": 0.72,
      "agent_collaboration_score": 0.91,
      "decision_count": 247,
      "response_time_avg": 1.35
    },
    "agent_statuses": [...],
    "active_alerts_count": 2,
    "recent_decisions_count": 15
  }
}
```

### 2. Agent Status Updates
**Endpoint:** `ws://localhost:8000/api/dashboard/ws/agents/{agent_id}/status`

Provides real-time updates for a specific agent's status and activities.

**Valid Agent IDs:**
- `pricing_agent`
- `inventory_agent`
- `promotion_agent`

**Message Types:**
- `agent_status`: Initial agent status when connection is established
- `agent_status_update`: Periodic agent status updates (every 3 seconds)

**Example Messages:**
```json
{
  "type": "agent_status",
  "agent_id": "pricing_agent",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "agent_id": "pricing_agent",
    "name": "Pricing Agent",
    "status": "active",
    "current_activity": "Analyzing demand elasticity for SKU123456",
    "last_decision_time": "2024-01-15T10:29:45Z",
    "decisions_count": 45,
    "success_rate": 92.3,
    "avg_response_time": 1.2
  }
}
```

### 3. Real-Time Metrics Stream
**Endpoint:** `ws://localhost:8000/api/dashboard/ws/metrics/real-time`

Provides high-frequency updates of key performance indicators for real-time dashboard charts.

**Message Types:**
- `metrics_stream`: Initial metrics when connection is established
- `metrics_update`: High-frequency metrics updates (every 2 seconds)

**Example Messages:**
```json
{
  "type": "metrics_stream",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "total_revenue": 125000.50,
    "total_profit": 45000.25,
    "inventory_turnover": 8.5,
    "stockout_incidents": 3,
    "waste_reduction_percentage": 15.2,
    "price_optimization_score": 0.87,
    "promotion_effectiveness": 0.72,
    "agent_collaboration_score": 0.91,
    "decision_count": 247,
    "response_time_avg": 1.35
  }
}
```

## REST API Endpoints

### Dashboard-Specific Endpoints

#### Get Agent Status
**GET** `/api/dashboard/agents/status`

Returns current status of all agents.

**Response:**
```json
[
  {
    "agent_id": "pricing_agent",
    "name": "Pricing Agent",
    "status": "active",
    "current_activity": "Analyzing demand elasticity for SKU123456",
    "last_decision_time": "2024-01-15T10:30:00Z",
    "decisions_count": 45,
    "success_rate": 92.3,
    "avg_response_time": 1.2
  }
]
```

#### Get Current Dashboard Metrics
**GET** `/api/dashboard/metrics/current`

Returns comprehensive dashboard metrics including system performance and agent status.

#### Get Recent Decisions
**GET** `/api/dashboard/decisions/recent?limit=20&agent_id=pricing_agent`

Returns recent agent decisions with optional filtering.

**Query Parameters:**
- `limit`: Maximum number of decisions to return (default: 20)
- `agent_id`: Filter by specific agent (optional)

#### Get Active Alerts
**GET** `/api/dashboard/alerts/active`

Returns current system alerts and notifications.

#### Manual Intervention
**POST** `/api/dashboard/agents/{agent_id}/intervention`

Perform manual intervention on a specific agent.

**Request Body:**
```json
{
  "agent_id": "pricing_agent",
  "action": "pause",
  "parameters": {},
  "reason": "Manual testing intervention"
}
```

**Valid Actions:**
- `pause`: Temporarily pause agent operations
- `resume`: Resume paused agent operations
- `reset`: Reset agent state
- `override`: Override current agent decision

## Client Implementation Examples

### JavaScript/TypeScript (React)
```typescript
// WebSocket connection for live updates
const ws = new WebSocket('ws://localhost:8000/api/dashboard/ws/live-updates');

ws.onopen = () => {
  console.log('Connected to live updates');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'connection_established':
      console.log('Connection established:', data.message);
      break;
    case 'metrics_update':
      updateDashboardMetrics(data.data);
      break;
    case 'agent_intervention':
      showInterventionNotification(data);
      break;
  }
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('WebSocket connection closed');
  // Implement reconnection logic
};
```

### Python Client
```python
import asyncio
import json
import websockets

async def connect_to_dashboard():
    uri = "ws://localhost:8000/api/dashboard/ws/live-updates"
    
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data['type']}")
            
            if data['type'] == 'metrics_update':
                metrics = data['data']['system_metrics']
                print(f"Revenue: ${metrics['total_revenue']:,.2f}")

asyncio.run(connect_to_dashboard())
```

## Error Handling

### WebSocket Error Codes
- `4004`: Invalid agent ID for agent-specific endpoints
- `1000`: Normal closure
- `1001`: Going away (server shutdown)
- `1006`: Abnormal closure (connection lost)

### Connection Management
The WebSocket server automatically handles:
- Connection cleanup on client disconnect
- Broadcasting to multiple clients
- Agent-specific subscription management
- Graceful error handling and logging

### Reconnection Strategy
Clients should implement exponential backoff for reconnection:
1. Initial reconnect after 1 second
2. Double the delay on each failed attempt
3. Maximum delay of 30 seconds
4. Reset delay on successful connection

## Security Considerations

### CORS Configuration
The API is configured to allow connections from:
- `http://localhost:3000` (React development server)
- `http://localhost:8501` (Streamlit dashboard)

### Rate Limiting
WebSocket connections are limited to:
- Maximum 100 concurrent connections per client IP
- Message rate limit of 10 messages per second per connection

### Authentication
Currently, the WebSocket endpoints do not require authentication for development purposes. In production, implement:
- JWT token validation
- Session-based authentication
- Role-based access control

## Monitoring and Debugging

### Connection Monitoring
Monitor WebSocket connections using:
```bash
# Check active connections
curl http://localhost:8000/api/dashboard/agents/status

# Monitor server logs
tail -f logs/websocket.log
```

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export LOG_LEVEL=DEBUG
```

This will provide detailed WebSocket connection and message logs for troubleshooting.