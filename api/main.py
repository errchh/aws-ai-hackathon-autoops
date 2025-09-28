"""
FastAPI execution layer for the autoops retail optimization system.

This module provides the REST API endpoints that agents use to execute actions
and retrieve data. It serves as the execution layer between the AI agents and
the underlying retail system.
"""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import pricing, inventory, promotions, shared, decisions, dashboard
from config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    settings = get_settings()
    print(f"Starting AutoOps Retail Optimization API v{settings.version}")
    print(f"Environment: {settings.environment}")
    
    yield
    
    # Shutdown
    print("Shutting down AutoOps Retail Optimization API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="AutoOps Retail Optimization API",
        description="Multi-agent AI system for retail optimization execution layer",
        version=settings.version,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(pricing.router, prefix="/api/pricing", tags=["pricing"])
    app.include_router(inventory.router, prefix="/api/inventory", tags=["inventory"])
    app.include_router(promotions.router, prefix="/api/promotions", tags=["promotions"])
    app.include_router(shared.router, prefix="/api", tags=["shared"])
    app.include_router(decisions.router, prefix="/api/decisions", tags=["decisions"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint providing API information."""
        return {
            "name": "AutoOps Retail Optimization API",
            "version": settings.version,
            "status": "operational",
            "docs": "/docs" if settings.environment != "production" else "disabled"
        }
    
    @app.get("/health", response_model=Dict[str, str])
    async def health_check():
        """Health check endpoint for monitoring."""
        return {"status": "healthy", "service": "autoops-retail-api"}
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request, exc):
        """Handle ValueError exceptions."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid input", "detail": str(exc)}
        )
    
    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        """Handle 404 errors."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Resource not found", "detail": "The requested resource was not found"}
        )
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )