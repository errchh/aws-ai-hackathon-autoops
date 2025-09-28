#!/usr/bin/env python3
"""
Dependency validation script for the AutoOps Retail Optimization system.

This script checks that all required dependencies are available and properly configured.
"""

import sys
from typing import List, Tuple


def check_dependency(module_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a dependency is available."""
    try:
        if import_name:
            __import__(import_name)
        else:
            __import__(module_name)
        return True, f"✓ {module_name} - Available"
    except ImportError as e:
        return False, f"✗ {module_name} - Missing: {e}"


def main():
    """Main validation function."""
    print("AutoOps Retail Optimization - Dependency Validation")
    print("=" * 60)
    
    # Core dependencies
    core_deps = [
        ("strands", "strands"),
        ("boto3", "boto3"),
        ("botocore", "botocore"),
        ("chromadb", "chromadb"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("pydantic-settings", "pydantic_settings"),
        ("numpy", "numpy"),
        ("httpx", "httpx"),
        ("python-dotenv", "dotenv"),
        ("structlog", "structlog"),
        ("python-multipart", "multipart"),
    ]
    
    # Development dependencies
    dev_deps = [
        ("pytest", "pytest"),
        ("pytest-asyncio", "pytest_asyncio"),
        ("pytest-mock", "pytest_mock"),
        ("black", "black"),
        ("isort", "isort"),
        ("mypy", "mypy"),
        ("ruff", "ruff"),
    ]
    
    print("\nCore Dependencies:")
    print("-" * 30)
    core_results = []
    for dep_name, import_name in core_deps:
        success, message = check_dependency(dep_name, import_name)
        core_results.append(success)
        print(message)
    
    print("\nDevelopment Dependencies:")
    print("-" * 30)
    dev_results = []
    for dep_name, import_name in dev_deps:
        success, message = check_dependency(dep_name, import_name)
        dev_results.append(success)
        print(message)
    
    # Summary
    print("\nSummary:")
    print("-" * 30)
    core_available = sum(core_results)
    core_total = len(core_results)
    dev_available = sum(dev_results)
    dev_total = len(dev_results)
    
    print(f"Core dependencies: {core_available}/{core_total} available")
    print(f"Dev dependencies: {dev_available}/{dev_total} available")
    
    if core_available == core_total:
        print("✓ All core dependencies are available - system ready!")
        return 0
    else:
        print("✗ Some core dependencies are missing - please install them")
        print("\nTo install missing dependencies:")
        print("pip install -e .")
        print("# or for development:")
        print("pip install -e .[dev]")
        return 1


if __name__ == "__main__":
    sys.exit(main())