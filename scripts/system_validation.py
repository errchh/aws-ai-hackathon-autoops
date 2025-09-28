#!/usr/bin/env python3
"""
System validation script for the autoops retail optimization system.

This script performs basic validation of system components and integration
points without requiring complex dependencies.
"""

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Any


def validate_project_structure():
    """Validate that the project has the required structure."""
    print("🔍 Validating Project Structure")
    print("-" * 40)

    required_dirs = [
        "agents",
        "api",
        "models",
        "config",
        "scenarios",
        "tests",
        "scripts",
    ]

    required_files = [
        "pyproject.toml",
        "README.md",
        "api/main.py",
        "agents/orchestrator.py",
        "models/core.py",
        "config/settings.py",
    ]

    missing_dirs = []
    missing_files = []

    for dir_name in required_dirs:
        if not os.path.isdir(dir_name):
            missing_dirs.append(dir_name)

    for file_name in required_files:
        if not os.path.isfile(file_name):
            missing_files.append(file_name)

    if missing_dirs:
        print(f"❌ Missing directories: {', '.join(missing_dirs)}")
        return False

    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False

    print("✅ Project structure is valid")
    return True


def validate_configuration():
    """Validate configuration files."""
    print("\n🔧 Validating Configuration")
    print("-" * 40)

    try:
        # Check if pyproject.toml exists and has basic structure
        with open("pyproject.toml", "r") as f:
            content = f.read()
            if "[project]" not in content:
                print("❌ pyproject.toml missing [project] section")
                return False
            if "[tool.hatch" not in content:
                print("❌ pyproject.toml missing tool configuration")
                return False

        print("✅ Configuration files are valid")
        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def validate_code_syntax():
    """Validate Python code syntax."""
    print("\n🐍 Validating Code Syntax")
    print("-" * 40)

    python_files = [
        "api/main.py",
        "agents/orchestrator.py",
        "models/core.py",
        "config/settings.py",
        "scenarios/data_generator.py",
    ]

    syntax_errors = []

    for file_path in python_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    code = f.read()
                compile(code, file_path, "exec")
            except SyntaxError as e:
                syntax_errors.append(f"{file_path}: {e}")
            except Exception as e:
                syntax_errors.append(f"{file_path}: {e}")

    if syntax_errors:
        print("❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        return False

    print("✅ Code syntax is valid")
    return True


def validate_test_structure():
    """Validate test file structure."""
    print("\n🧪 Validating Test Structure")
    print("-" * 40)

    test_files = [
        "tests/__init__.py",
        "tests/test_integration_e2e.py",
        "tests/test_api_endpoints.py",
        "tests/test_models.py",
    ]

    missing_tests = []
    for test_file in test_files:
        if not os.path.exists(test_file):
            missing_tests.append(test_file)

    if missing_tests:
        print(f"❌ Missing test files: {', '.join(missing_tests)}")
        return False

    print("✅ Test structure is valid")
    return True


def validate_scenario_data():
    """Validate scenario data generation."""
    print("\n📊 Validating Scenario Data")
    print("-" * 40)

    try:
        # Try to import and test basic functionality
        sys.path.insert(0, os.getcwd())

        # Mock chromadb to avoid import errors
        import types

        sys.modules["chromadb"] = types.ModuleType("chromadb")

        from scenarios.data_generator import ScenarioDataGenerator

        generator = ScenarioDataGenerator(seed=42)

        # Test basic data generation
        products = generator.generate_product_catalog(3)
        events = generator.generate_market_events(2)
        decisions = generator.generate_agent_decisions(2)
        metrics = generator.generate_performance_metrics(5)

        if len(products) != 3:
            print(f"❌ Product generation failed: expected 3, got {len(products)}")
            return False

        if len(events) != 2:
            print(f"❌ Event generation failed: expected 2, got {len(events)}")
            return False

        if len(decisions) != 2:
            print(f"❌ Decision generation failed: expected 2, got {len(decisions)}")
            return False

        if len(metrics) != 5:
            print(f"❌ Metrics generation failed: expected 5, got {len(metrics)}")
            return False

        print("✅ Scenario data generation is working")
        return True

    except Exception as e:
        print(f"❌ Scenario data validation failed: {e}")
        return False


def validate_agent_structure():
    """Validate agent implementation structure."""
    print("\n🤖 Validating Agent Structure")
    print("-" * 40)

    agent_files = [
        "agents/__init__.py",
        "agents/orchestrator.py",
        "agents/pricing_agent.py",
        "agents/inventory_agent.py",
        "agents/promotion_agent.py",
        "agents/collaboration.py",
    ]

    missing_agents = []
    for agent_file in agent_files:
        if not os.path.exists(agent_file):
            missing_agents.append(agent_file)

    if missing_agents:
        print(f"❌ Missing agent files: {', '.join(missing_agents)}")
        return False

    # Check for basic agent class structure
    try:
        with open("agents/orchestrator.py", "r") as f:
            content = f.read()
            if "class RetailOptimizationOrchestrator" not in content:
                print("❌ Orchestrator class not found")
                return False

        with open("agents/pricing_agent.py", "r") as f:
            content = f.read()
            if "pricing_agent" not in content:
                print("❌ Pricing agent instance not found")
                return False

        print("✅ Agent structure is valid")
        return True

    except Exception as e:
        print(f"❌ Agent structure validation failed: {e}")
        return False


def validate_api_structure():
    """Validate API implementation structure."""
    print("\n🌐 Validating API Structure")
    print("-" * 40)

    api_files = [
        "api/__init__.py",
        "api/main.py",
        "api/routers/__init__.py",
        "api/routers/pricing.py",
        "api/routers/inventory.py",
        "api/routers/promotions.py",
    ]

    missing_api = []
    for api_file in api_files:
        if not os.path.exists(api_file):
            missing_api.append(api_file)

    if missing_api:
        print(f"❌ Missing API files: {', '.join(missing_api)}")
        return False

    # Check for FastAPI app structure
    try:
        with open("api/main.py", "r") as f:
            content = f.read()
            if "create_app()" not in content:
                print("❌ FastAPI app creation function not found")
                return False
            if "FastAPI" not in content:
                print("❌ FastAPI import not found")
                return False

        print("✅ API structure is valid")
        return True

    except Exception as e:
        print(f"❌ API structure validation failed: {e}")
        return False


def validate_memory_system():
    """Validate memory system structure."""
    print("\n🧠 Validating Memory System")
    print("-" * 40)

    if not os.path.exists("agents/memory.py"):
        print("❌ Memory system file not found")
        return False

    try:
        with open("agents/memory.py", "r") as f:
            content = f.read()
            if "class AgentMemory" not in content:
                print("❌ AgentMemory class not found")
                return False
            if "agent_memory" not in content:
                print("❌ Global agent_memory instance not found")
                return False

        print("✅ Memory system structure is valid")
        return True

    except Exception as e:
        print(f"❌ Memory system validation failed: {e}")
        return False


def generate_validation_report(results: Dict[str, bool]):
    """Generate a comprehensive validation report."""
    print("\n" + "=" * 80)
    print("📋 SYSTEM VALIDATION REPORT")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)
    success_rate = passed / total if total > 0 else 0

    print("\n🔍 Validation Results:")
    print(f"  Passed: {passed}/{total} ({success_rate:.1%})")

    print("\n📊 Component Status:")
    for component, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {component.replace('_', ' ').title()}")

    print("\n🎯 Overall Assessment:")
    if success_rate == 1.0:
        print("  🟢 EXCELLENT: All components validated successfully")
        print("  The system is ready for integration testing")
    elif success_rate >= 0.8:
        print("  🟡 GOOD: Most components are valid")
        print("  Minor issues need to be addressed before full testing")
    elif success_rate >= 0.6:
        print("  🟠 FAIR: Several components need attention")
        print("  Significant work required before integration testing")
    else:
        print("  🔴 CRITICAL: Major components are missing or broken")
        print("  System requires substantial rework")

    # Save report to file
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_results": results,
        "summary": {
            "passed": passed,
            "total": total,
            "success_rate": success_rate,
            "assessment": "excellent"
            if success_rate == 1.0
            else "good"
            if success_rate >= 0.8
            else "fair"
            if success_rate >= 0.6
            else "critical",
        },
    }

    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n💾 Report saved to validation_report.json")
    return success_rate == 1.0


def main():
    """Run complete system validation."""
    print("🚀 AutoOps Retail Optimization - System Validation")
    print("=" * 80)

    validation_results = {}

    # Run all validations
    validations = [
        ("project_structure", validate_project_structure),
        ("configuration", validate_configuration),
        ("code_syntax", validate_code_syntax),
        ("test_structure", validate_test_structure),
        ("scenario_data", validate_scenario_data),
        ("agent_structure", validate_agent_structure),
        ("api_structure", validate_api_structure),
        ("memory_system", validate_memory_system),
    ]

    for name, validator_func in validations:
        try:
            result = validator_func()
            validation_results[name] = result
        except Exception as e:
            print(f"❌ Validation '{name}' crashed: {e}")
            validation_results[name] = False

    # Generate final report
    success = generate_validation_report(validation_results)

    if success:
        print("\n🎉 System validation completed successfully!")
        print("   Ready to proceed with integration testing.")
    else:
        print("\n⚠️  System validation found issues that need to be addressed.")
        print("   Please fix the reported problems before proceeding.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
