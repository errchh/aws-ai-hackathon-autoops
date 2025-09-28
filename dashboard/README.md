# AutoOps Healthcare & Wellness Retail Optimization Dashboard

A comprehensive Streamlit dashboard for monitoring the multi-agent AI system that optimizes pricing, inventory, and promotions in real-time for healthcare and wellness products.

## Features

- **6 User Story Sections**: Complete coverage of all requirement sections with tabular decision displays
- **Real-time Updates**: Auto-refresh functionality with live data from FastAPI backend
- **Demo Controls**: One-click "START DEMO" button with 30-second countdown timer
- **Auto-scroll Highlights**: Automatic section highlighting during demo sequence
- **Large KPI Display**: Prominent profit and waste reduction metrics
- **Auto-reset**: Demo automatically resets after completion for repeated presentations
- **Interactive Tables**: Sortable and filterable decision tables with 4-column format

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the FastAPI backend is running on `http://localhost:8000`

3. Run the dashboard:
```bash
streamlit run app.py
```

## Dashboard Sections

### 1. Multi-Agent Healthcare & Wellness Operations
Shows orchestrator decisions and system-level coordination between agents.

### 2. Healthcare & Wellness Pricing Optimization
Displays pricing agent decisions for dynamic pricing adjustments.

### 3. Healthcare & Wellness Inventory Management
Shows inventory agent decisions for optimal stock levels.

### 4. Healthcare Retail Performance Metrics
Displays system KPIs, performance indicators, and active alerts.

### 5. Healthcare & Wellness AI Learning
Shows agent learning outcomes and collaborative decision-making.

### 6. Healthcare & Wellness Campaign Results
Displays promotion agent decisions and campaign effectiveness.

## Demo Mode

Click the "🚀 START DEMO" button to start a 30-second automated presentation that:

- Highlights each section in sequence
- Shows real-time agent decision-making
- Updates KPIs and metrics
- Automatically resets for the next presentation

## API Integration

The dashboard communicates with the FastAPI backend via REST endpoints:

- `/api/dashboard/agents/status` - Agent status information
- `/api/dashboard/metrics/current` - Current system metrics
- `/api/dashboard/decisions/*` - Decision data for each agent type
- `/api/dashboard/alerts/active` - Active system alerts

## Configuration

- **API_BASE_URL**: Backend API URL (default: `http://localhost:8000/api/dashboard`)
- **REFRESH_INTERVAL**: Auto-refresh interval in seconds (default: 5)
- **DEMO_DURATION**: Demo duration in seconds (default: 30)