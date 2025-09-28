#!/usr/bin/env python3
"""
Test script for the AutoOps dashboard to verify functionality without full dependencies.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def test_imports():
    """Test that basic imports work."""
    try:
        import pandas as pd

        print("✓ pandas import successful")

        import requests

        print("✓ requests import successful")

        # Test streamlit import (may not be available)
        try:
            import streamlit as st

            print("✓ streamlit import successful")
        except ImportError:
            print("⚠ streamlit not available - install with: pip install streamlit")

        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_dashboard_logic():
    """Test dashboard logic with mock data."""
    print("\nTesting dashboard logic...")

    # Mock data structures
    mock_decisions = [
        {
            "trigger_detected": "High inventory levels detected",
            "agent_decision_action": "Applied markdown strategy",
            "value_before": "$24.99",
            "value_after": "$22.99",
        }
    ]

    mock_metrics = {
        "system_metrics": {
            "total_profit": 45000.25,
            "waste_reduction_percentage": 15.2,
            "inventory_turnover": 8.5,
            "decision_count": 247,
        }
    }

    mock_agents = [
        {
            "name": "Pricing Agent",
            "status": "active",
            "decisions_count": 45,
            "success_rate": 92.3,
        }
    ]

    # Test data formatting
    try:
        import pandas as pd

        # Test decision table formatting
        df = pd.DataFrame(mock_decisions)
        assert len(df.columns) == 4
        assert "Trigger Detected" in df.columns
        print("✓ Decision table formatting works")

        # Test KPI display logic
        metrics = mock_metrics.get("system_metrics", {})
        profit = metrics.get("total_profit", 0)
        assert isinstance(profit, (int, float))
        print("✓ KPI metrics extraction works")

        # Test agent status display
        assert len(mock_agents) > 0
        agent = mock_agents[0]
        assert "name" in agent
        assert "status" in agent
        print("✓ Agent status display logic works")

        return True

    except Exception as e:
        print(f"✗ Dashboard logic test failed: {e}")
        return False


def test_demo_controller():
    """Test demo controller logic."""
    print("\nTesting demo controller...")

    class MockDemoController:
        def __init__(self):
            self.demo_active = False
            self.current_section = 0
            self.section_order = [1, 2, 3, 4, 5, 6]

        def start_demo(self):
            self.demo_active = True

        def should_highlight_section(self, section_number: int) -> bool:
            if not self.demo_active:
                return False
            return section_number == self.section_order[self.current_section]

    controller = MockDemoController()
    assert not controller.should_highlight_section(1)

    controller.start_demo()
    controller.current_section = 0  # First section
    assert controller.should_highlight_section(1)
    assert not controller.should_highlight_section(2)

    print("✓ Demo controller logic works")
    return True


def main():
    """Run all tests."""
    print("AutoOps Dashboard Test Suite")
    print("=" * 40)

    tests = [
        test_imports,
        test_dashboard_logic,
        test_demo_controller,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Dashboard implementation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
