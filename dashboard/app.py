"""
AutoOps Healthcare & Wellness Retail Optimization Dashboard

A comprehensive Streamlit dashboard for monitoring the multi-agent AI system
that optimizes pricing, inventory, and promotions in real-time.

Features:
- 6 user story sections with tabular decision displays
- Real-time updates and auto-refresh
- Demo controls with START DEMO button and countdown timer
- Auto-scroll highlights and large KPI display
- HTTP client integration with FastAPI backend
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import pandas as pd
import requests
import streamlit as st

# Configuration
API_BASE_URL = "http://localhost:8000/api/dashboard"
REFRESH_INTERVAL = 5  # seconds
DEMO_DURATION = 30  # seconds

# Page configuration
st.set_page_config(
    page_title="AutoOps Healthcare & Wellness Retail Optimization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }

    .kpi-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .kpi-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }

    .kpi-label {
        font-size: 1rem;
        opacity: 0.9;
    }

    .demo-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 1rem 2rem;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
        transition: all 0.3s ease;
    }

    .demo-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
    }

    .countdown-timer {
        font-size: 3rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    .decision-table {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .highlight-section {
        animation: highlight 2s ease-in-out;
    }

    @keyframes highlight {
        0% { background-color: rgba(52, 152, 219, 0.1); }
        50% { background-color: rgba(52, 152, 219, 0.3); }
        100% { background-color: rgba(52, 152, 219, 0.1); }
    }

    .status-active {
        color: #27ae60;
        font-weight: bold;
    }

    .status-idle {
        color: #f39c12;
        font-weight: bold;
    }

    .status-error {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


class DashboardClient:
    """HTTP client for communicating with the FastAPI backend."""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def get_agents_status(self) -> List[Dict]:
        """Get current status of all agents."""
        try:
            response = self.session.get(f"{self.base_url}/agents/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch agent status: {e}")
            return []

    def get_current_metrics(self) -> Dict:
        """Get current dashboard metrics."""
        try:
            response = self.session.get(f"{self.base_url}/metrics/current", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch metrics: {e}")
            return {}

    def get_recent_decisions(self, limit: int = 20) -> List[Dict]:
        """Get recent agent decisions."""
        try:
            response = self.session.get(
                f"{self.base_url}/decisions/recent", params={"limit": limit}, timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch recent decisions: {e}")
            return []

    def get_active_alerts(self) -> List[Dict]:
        """Get active system alerts."""
        try:
            response = self.session.get(f"{self.base_url}/alerts/active", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch alerts: {e}")
            return []

    def get_pricing_decisions(self, limit: int = 10) -> List[Dict]:
        """Get pricing agent decisions."""
        try:
            response = self.session.get(
                f"{self.base_url}/decisions/pricing", params={"limit": limit}, timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch pricing decisions: {e}")
            return []

    def get_inventory_decisions(self, limit: int = 10) -> List[Dict]:
        """Get inventory agent decisions."""
        try:
            response = self.session.get(
                f"{self.base_url}/decisions/inventory",
                params={"limit": limit},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch inventory decisions: {e}")
            return []

    def get_promotion_decisions(self, limit: int = 10) -> List[Dict]:
        """Get promotion agent decisions."""
        try:
            response = self.session.get(
                f"{self.base_url}/decisions/promotion",
                params={"limit": limit},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch promotion decisions: {e}")
            return []

    def get_orchestrator_decisions(self, limit: int = 10) -> List[Dict]:
        """Get orchestrator decisions."""
        try:
            response = self.session.get(
                f"{self.base_url}/decisions/orchestrator",
                params={"limit": limit},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch orchestrator decisions: {e}")
            return []

    def get_collaboration_decisions(self, limit: int = 10) -> List[Dict]:
        """Get collaboration decisions."""
        try:
            response = self.session.get(
                f"{self.base_url}/decisions/collaboration",
                params={"limit": limit},
                timeout=5,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch collaboration decisions: {e}")
            return []

    def get_coverage_metrics(self) -> Dict:
        """Get function coverage metrics."""
        try:
            response = self.session.get(f"{self.base_url}/coverage/metrics", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"Failed to fetch coverage metrics: {e}")
            return {}


class DemoController:
    """Controls the demo flow and timing."""

    def __init__(self):
        self.demo_active = False
        self.demo_start_time = None
        self.demo_end_time = None
        self.current_section = 0
        self.section_order = [1, 2, 3, 4, 5, 6]  # Order to highlight sections

    def start_demo(self):
        """Start the demo sequence."""
        self.demo_active = True
        self.demo_start_time = datetime.now()
        self.demo_end_time = self.demo_start_time + timedelta(seconds=DEMO_DURATION)
        self.current_section = 0

    def stop_demo(self):
        """Stop the demo sequence."""
        self.demo_active = False
        self.demo_start_time = None
        self.demo_end_time = None
        self.current_section = 0

    def get_remaining_time(self) -> int:
        """Get remaining demo time in seconds."""
        if not self.demo_active or self.demo_end_time is None:
            return 0
        remaining = max(0, int((self.demo_end_time - datetime.now()).total_seconds()))
        if remaining == 0:
            self.stop_demo()
        return remaining

    def should_highlight_section(self, section_number: int) -> bool:
        """Check if a section should be highlighted."""
        if not self.demo_active or self.demo_start_time is None:
            return False

        # Type checker help: we've already checked demo_start_time is not None
        start_time = self.demo_start_time  # type: datetime
        elapsed = (datetime.now() - start_time).total_seconds()
        section_duration = DEMO_DURATION / len(self.section_order)

        target_section_index = int(elapsed / section_duration)
        if target_section_index >= len(self.section_order):
            target_section_index = len(self.section_order) - 1

        return section_number == self.section_order[target_section_index]


def format_decision_table(decisions: List[Dict]) -> pd.DataFrame:
    """Format decisions into a pandas DataFrame for display."""
    if not decisions:
        # Create empty DataFrame with proper column structure
        empty_data = {
            "Trigger Detected": [],
            "Agent Decision & Action": [],
            "Value Before": [],
            "Value After": [],
        }
        return pd.DataFrame(empty_data)

    df_data = []
    for decision in decisions:
        df_data.append(
            {
                "Trigger Detected": decision.get("trigger_detected", "N/A"),
                "Agent Decision & Action": decision.get("agent_decision_action", "N/A"),
                "Value Before": decision.get("value_before", "N/A"),
                "Value After": decision.get("value_after", "N/A"),
            }
        )

    return pd.DataFrame(df_data)


def display_kpi_metrics(metrics: Dict):
    """Display key performance indicators."""
    if not metrics:
        st.warning("No metrics available")
        return

    system_metrics = metrics.get("system_metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        profit = system_metrics.get("total_profit", 0)
        st.markdown(
            f"""
        <div class="kpi-container">
            <div class="kpi-label">Total Profit</div>
            <div class="kpi-value">${profit:,.0f}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        waste_reduction = system_metrics.get("waste_reduction_percentage", 0)
        st.markdown(
            f"""
        <div class="kpi-container">
            <div class="kpi-label">Waste Reduction</div>
            <div class="kpi-value">{waste_reduction:.1f}%</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        turnover = system_metrics.get("inventory_turnover", 0)
        st.markdown(
            f"""
        <div class="kpi-container">
            <div class="kpi-label">Inventory Turnover</div>
            <div class="kpi-value">{turnover:.1f}x</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        decisions = system_metrics.get("decision_count", 0)
        st.markdown(
            f"""
        <div class="kpi-container">
            <div class="kpi-label">AI Decisions</div>
            <div class="kpi-value">{decisions}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def display_agent_status(agents: List[Dict]):
    """Display agent status information."""
    if not agents:
        st.warning("No agent status available")
        return

    cols = st.columns(len(agents))
    for i, agent in enumerate(agents):
        with cols[i]:
            status_class = f"status-{agent.get('status', 'unknown').lower()}"
            st.markdown(
                f"""
            <div style="text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; margin: 0.5rem 0;">
                <h4>{agent.get("name", "Unknown Agent")}</h4>
                <p class="{status_class}">{agent.get("status", "Unknown").title()}</p>
                <p style="font-size: 0.8rem; color: #666;">
                    {agent.get("decisions_count", 0)} decisions<br>
                    {agent.get("success_rate", 0):.1f}% success rate
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )


def main():
    """Main dashboard application."""
    # Initialize session state
    if "client" not in st.session_state:
        st.session_state.client = DashboardClient()

    if "demo_controller" not in st.session_state:
        st.session_state.demo_controller = DemoController()

    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    # Main header
    st.markdown(
        '<h1 class="main-header">AutoOps Healthcare & Wellness Retail Optimization</h1>',
        unsafe_allow_html=True,
    )

    # Demo controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button(
            "🚀 START DEMO", key="start_demo", help="Start 30-second demo sequence"
        ):
            st.session_state.demo_controller.start_demo()
            st.rerun()

    with col2:
        remaining_time = st.session_state.demo_controller.get_remaining_time()
        if remaining_time > 0:
            st.markdown(
                f'<div class="countdown-timer">⏱️ {remaining_time}s</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align: center; font-size: 1.2rem; color: #666;">Ready for Demo</div>',
                unsafe_allow_html=True,
            )

    with col3:
        if st.button("🔄 Refresh Data", key="refresh_data"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    # Fetch data
    current_time = datetime.now()
    if (
        current_time - st.session_state.last_refresh
    ).total_seconds() > REFRESH_INTERVAL:
        st.session_state.last_refresh = current_time

        # Fetch all data sequentially (no threading to avoid session state issues)
        with st.spinner("Fetching latest data..."):
            st.session_state.agents_status = st.session_state.client.get_agents_status()
            st.session_state.current_metrics = st.session_state.client.get_current_metrics()
            st.session_state.active_alerts = st.session_state.client.get_active_alerts()
            st.session_state.pricing_decisions = st.session_state.client.get_pricing_decisions()
            st.session_state.inventory_decisions = st.session_state.client.get_inventory_decisions()
            st.session_state.promotion_decisions = st.session_state.client.get_promotion_decisions()
            st.session_state.orchestrator_decisions = st.session_state.client.get_orchestrator_decisions()
            st.session_state.collaboration_decisions = st.session_state.client.get_collaboration_decisions()

    # Display KPI metrics
    if "current_metrics" in st.session_state:
        display_kpi_metrics(st.session_state.current_metrics)

    # Display agent status
    if "agents_status" in st.session_state:
        display_agent_status(st.session_state.agents_status)

    # Section 1: Multi-Agent System Operations
    section_highlight = (
        "highlight-section"
        if st.session_state.demo_controller.should_highlight_section(1)
        else ""
    )
    st.markdown(f'<div class="{section_highlight}">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-header">1. Multi-Agent Healthcare & Wellness Operations</h2>',
        unsafe_allow_html=True,
    )

    if "orchestrator_decisions" in st.session_state:
        df = format_decision_table(st.session_state.orchestrator_decisions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Loading orchestrator decisions...")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 2: Dynamic Pricing
    section_highlight = (
        "highlight-section"
        if st.session_state.demo_controller.should_highlight_section(2)
        else ""
    )
    st.markdown(f'<div class="{section_highlight}">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-header">2. Healthcare & Wellness Pricing Optimization</h2>',
        unsafe_allow_html=True,
    )

    if "pricing_decisions" in st.session_state:
        df = format_decision_table(st.session_state.pricing_decisions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Loading pricing decisions...")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 3: Inventory Management
    section_highlight = (
        "highlight-section"
        if st.session_state.demo_controller.should_highlight_section(3)
        else ""
    )
    st.markdown(f'<div class="{section_highlight}">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-header">3. Healthcare & Wellness Inventory Management</h2>',
        unsafe_allow_html=True,
    )

    if "inventory_decisions" in st.session_state:
        df = format_decision_table(st.session_state.inventory_decisions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Loading inventory decisions...")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 4: System Performance Metrics
    section_highlight = (
        "highlight-section"
        if st.session_state.demo_controller.should_highlight_section(4)
        else ""
    )
    st.markdown(f'<div class="{section_highlight}">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-header">4. Healthcare Retail Performance Metrics</h2>',
        unsafe_allow_html=True,
    )

    if "current_metrics" in st.session_state:
        metrics = st.session_state.current_metrics.get("system_metrics", {})
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Revenue", f"${metrics.get('total_revenue', 0):,.0f}")
            st.metric("Stockouts", metrics.get("stockout_incidents", 0))

        with col2:
            st.metric(
                "Price Optimization",
                f"{metrics.get('price_optimization_score', 0):.2f}",
            )
            st.metric(
                "Promotion Effectiveness",
                f"{metrics.get('promotion_effectiveness', 0):.2f}",
            )

        # Display alerts if any
        if "active_alerts" in st.session_state and st.session_state.active_alerts:
            st.subheader("Active Alerts")
            for alert in st.session_state.active_alerts:
                if alert.get("severity") == "critical":
                    st.error(f"🚨 {alert.get('title')}: {alert.get('message')}")
                elif alert.get("severity") == "warning":
                    st.warning(f"⚠️ {alert.get('title')}: {alert.get('message')}")
                else:
                    st.info(f"ℹ️ {alert.get('title')}: {alert.get('message')}")
    else:
        st.info("Loading system metrics...")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 5: Agent Learning & Memory
    section_highlight = (
        "highlight-section"
        if st.session_state.demo_controller.should_highlight_section(5)
        else ""
    )
    st.markdown(f'<div class="{section_highlight}">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-header">5. Healthcare & Wellness AI Learning</h2>',
        unsafe_allow_html=True,
    )

    if "collaboration_decisions" in st.session_state:
        df = format_decision_table(st.session_state.collaboration_decisions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Loading learning outcomes...")
    st.markdown("</div>", unsafe_allow_html=True)

    # Section 6: Simulation Results
    section_highlight = (
        "highlight-section"
        if st.session_state.demo_controller.should_highlight_section(6)
        else ""
    )
    st.markdown(f'<div class="{section_highlight}">', unsafe_allow_html=True)
    st.markdown(
        '<h2 class="section-header">6. Healthcare & Wellness Campaign Results</h2>',
        unsafe_allow_html=True,
    )

    if "promotion_decisions" in st.session_state:
        df = format_decision_table(st.session_state.promotion_decisions)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("Loading simulation results...")
    st.markdown("</div>", unsafe_allow_html=True)

    # Auto-refresh and demo control
    if st.session_state.demo_controller.demo_active:
        remaining = st.session_state.demo_controller.get_remaining_time()
        if remaining > 0:
            # Use st.empty() and sleep for smoother countdown
            placeholder = st.empty()
            placeholder.info(f"Demo running... {remaining}s remaining")
            time.sleep(1)
            st.rerun()
        else:
            st.session_state.demo_controller.stop_demo()
            st.success("🎉 Demo completed! Click START DEMO to run again.")
    
    # Add auto-refresh button instead of automatic refresh
    if st.button("🔄 Auto Refresh", key="auto_refresh"):
        st.rerun()


if __name__ == "__main__":
    main()
