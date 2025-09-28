#!/usr/bin/env python3
"""Validation script to check that all dependencies are properly installed."""

import sys
from pathlib import Path

def check_imports():
    """Check that all required packages can be imported."""
    required_packages = [
        ("strands", "AWS Strands Agents SDK"),
        ("chromadb", "ChromaDB"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("boto3", "AWS SDK"),
        ("pydantic", "Pydantic"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("structlog", "Structured Logging"),
    ]
    
    print("🔍 Checking package imports...")
    failed_imports = []
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {description} ({package})")
        except ImportError as e:
            print(f"❌ {description} ({package}): {e}")
            failed_imports.append(package)
    
    return failed_imports


def check_configuration():
    """Check that configuration can be loaded."""
    print("\n🔍 Checking configuration...")
    try:
        from config.settings import settings
        print(f"✅ Configuration loaded successfully")
        print(f"   - App Name: {settings.app_name}")
        print(f"   - Version: {settings.version}")
        print(f"   - AWS Region: {settings.aws.region}")
        print(f"   - Bedrock Model: {settings.bedrock.model_id}")
        return True
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False


def check_directories():
    """Check that required directories exist."""
    print("\n🔍 Checking project structure...")
    required_dirs = [
        "agents", "tools", "models", "api", "config", 
        "simulation", "dashboard", "tests", "data"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ directory exists")
        else:
            print(f"❌ {dir_name}/ directory missing")
            missing_dirs.append(dir_name)
    
    return missing_dirs


def main():
    """Run all validation checks."""
    print("🚀 AutoOps Retail Optimization - Setup Validation")
    print("=" * 50)
    
    # Check imports
    failed_imports = check_imports()
    
    # Check configuration
    config_ok = check_configuration()
    
    # Check directories
    missing_dirs = check_directories()
    
    # Summary
    print("\n📋 Validation Summary")
    print("=" * 30)
    
    if not failed_imports and config_ok and not missing_dirs:
        print("🎉 All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and configure your AWS credentials")
        print("2. Ensure AWS Bedrock access is enabled for Anthropic Claude")
        print("3. Start implementing the agents!")
        return 0
    else:
        print("⚠️  Some issues found:")
        if failed_imports:
            print(f"   - Failed imports: {', '.join(failed_imports)}")
        if not config_ok:
            print("   - Configuration loading failed")
        if missing_dirs:
            print(f"   - Missing directories: {', '.join(missing_dirs)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())