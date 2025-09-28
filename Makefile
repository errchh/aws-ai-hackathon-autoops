.PHONY: help install test lint format clean validate run-api run-dashboard

help: ## Show this help message
	@echo "AutoOps Retail Optimization - Development Commands"
	@echo "=================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies with uv
	uv venv
	uv pip install -e .

validate: ## Validate the setup
	python scripts/validate_setup.py

test: ## Run tests
	pytest

lint: ## Run linting checks
	ruff check .
	mypy .

format: ## Format code
	black .
	isort .
	ruff check --fix .

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/

run-api: ## Run the FastAPI server
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-dashboard: ## Run the Streamlit dashboard
	streamlit run dashboard/app.py --server.port 8501

dev: ## Run in development mode
	python main.py