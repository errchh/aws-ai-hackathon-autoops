#!/usr/bin/env python3
"""
Startup script for the FastAPI execution layer.

This script starts the API server and provides basic health checks.
"""

import subprocess
import sys
import time
import httpx
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi
        import uvicorn
        import pydantic
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False


def start_api_server():
    """Start the FastAPI server."""
    print("🚀 Starting AutoOps Retail Optimization API...")
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    
    try:
        # Start the server
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        print("API will be available at: http://localhost:8000")
        print("API documentation: http://localhost:8000/docs")
        print("Press Ctrl+C to stop the server")
        
        # Run the server
        subprocess.run(cmd, cwd=project_root, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True


def health_check():
    """Perform a health check on the running API."""
    print("🔍 Performing health check...")
    
    try:
        response = httpx.get("http://localhost:8000/health", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API is healthy: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except httpx.ConnectError:
        print("❌ Could not connect to API server")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def main():
    """Main startup function."""
    print("AutoOps Retail Optimization API Startup")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if we should just do a health check
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        if health_check():
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Start the server
    if start_api_server():
        print("✅ Server started successfully")
    else:
        print("❌ Failed to start server")
        sys.exit(1)


if __name__ == "__main__":
    main()