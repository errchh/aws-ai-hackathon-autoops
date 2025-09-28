# Dependencies Documentation

This document explains the dependencies used in the AutoOps Retail Optimization system.

## Core Dependencies

### Multi-Agent Framework
- **strands**: AWS Strands Agents framework for multi-agent orchestration and communication
- **boto3/botocore**: AWS SDK for Python, used for Bedrock (Claude) integration

### Vector Database & Memory
- **chromadb**: Vector database for agent memory and decision storage
- **numpy**: Numerical computing support for embeddings and calculations

### Web Framework & API
- **fastapi**: Modern, fast web framework for building APIs
- **uvicorn**: ASGI server for running FastAPI applications
- **python-multipart**: Support for form data and file uploads in FastAPI

### Data Validation & Configuration
- **pydantic**: Data validation and settings management using Python type annotations
- **pydantic-settings**: Settings management with environment variable support
- **python-dotenv**: Load environment variables from .env files

### HTTP & Networking
- **httpx**: Modern HTTP client for Python with async support

### Logging & Monitoring
- **structlog**: Structured logging for better observability

## Development Dependencies

### Testing Framework
- **pytest**: Testing framework
- **pytest-asyncio**: Async support for pytest
- **pytest-mock**: Mock object support for testing
- **anyio**: Async I/O library compatibility

### Code Quality
- **black**: Code formatter
- **isort**: Import sorter
- **mypy**: Static type checker
- **ruff**: Fast Python linter
- **pre-commit**: Git hooks for code quality

## Optional Dependencies

### Dashboard (Python Alternative)
- **streamlit**: Alternative Python-based dashboard (React dashboard is preferred)

### Enhanced Simulation
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting library
- **seaborn**: Statistical data visualization

## Frontend Dependencies (Separate)

The React.js dashboard has its own dependency management in `dashboard/package.json`:

### Core React Dependencies
- **react**: React library
- **react-dom**: React DOM rendering
- **typescript**: TypeScript support

### UI Components
- **@radix-ui/react-slot**: Accessible UI primitives
- **lucide-react**: Icon library
- **tailwindcss**: Utility-first CSS framework
- **class-variance-authority**: Utility for creating variant-based component APIs

### Data Visualization
- **recharts**: React charting library for metrics visualization

### Development Tools
- **vite**: Fast build tool and dev server
- **eslint**: JavaScript/TypeScript linter
- **@playwright/test**: End-to-end testing framework
- **@testing-library/react**: React testing utilities

## Installation

### Python Dependencies
```bash
# Install core dependencies
pip install -e .

# Install with development dependencies
pip install -e .[dev]

# Install with all optional dependencies
pip install -e .[dev,dashboard,simulation]
```

### Frontend Dependencies
```bash
cd dashboard
npm install
```

## Validation

To validate that all dependencies are properly installed:

```bash
python scripts/validate_dependencies.py
```

## Version Requirements

- **Python**: >=3.9
- **Node.js**: >=18.0 (for React dashboard)
- **AWS CLI**: Configured with appropriate permissions for Bedrock

## AWS Services Used

- **Amazon Bedrock**: For Claude model access
- **AWS IAM**: For service permissions
- **AWS CloudWatch**: For logging (optional)

## Environment Variables

Key environment variables that need to be configured:

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Application Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Database Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma
```

See `.env.example` for a complete list of configuration options.