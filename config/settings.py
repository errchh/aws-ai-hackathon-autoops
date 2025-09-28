"""Configuration management for the retail optimization system."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AWSSettings(BaseSettings):
    """AWS configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="AWS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    region: str = Field(default="us-east-1", description="AWS region")
    access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret access key"
    )
    session_token: Optional[str] = Field(default=None, description="AWS session token")


class BedrockSettings(BaseSettings):
    """AWS Bedrock configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="BEDROCK_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Bedrock model ID for Anthropic Claude",
    )
    max_tokens: int = Field(
        default=4096, description="Maximum tokens for model responses"
    )
    temperature: float = Field(
        default=0.1, description="Model temperature for responses"
    )


class ChromaDBSettings(BaseSettings):
    """ChromaDB configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="CHROMADB_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="localhost", description="ChromaDB host")
    port: int = Field(default=8000, description="ChromaDB port")
    collection_name: str = Field(
        default="retail_agent_memory",
        description="ChromaDB collection name for agent memory",
    )
    persist_directory: str = Field(
        default="./data/chromadb", description="Directory for ChromaDB persistence"
    )


class APISettings(BaseSettings):
    """API configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")
    reload: bool = Field(default=False, description="Enable auto-reload")


class DashboardSettings(BaseSettings):
    """Dashboard configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="DASHBOARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    host: str = Field(default="0.0.0.0", description="Dashboard host")
    port: int = Field(default=8501, description="Dashboard port")
    title: str = Field(
        default="AutoOps Retail Optimization Dashboard", description="Dashboard title"
    )


class StrandsSettings(BaseSettings):
    """AWS Strands Agents configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="STRANDS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    timeout_seconds: int = Field(default=300, description="Agent timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    log_level: str = Field(default="INFO", description="Logging level")


class LangfuseSettings(BaseSettings):
    """Langfuse observability configuration settings for v3."""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    host: str = Field(
        default="https://cloud.langfuse.com", description="Langfuse host URL"
    )
    enabled: bool = Field(default=True, description="Enable Langfuse tracing")
    sample_rate: float = Field(
        default=1.0, description="Sampling rate for traces (0.0-1.0)"
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    flush_interval: float = Field(default=5.0, description="Flush interval in seconds")
    flush_at: int = Field(default=15, description="Number of events to trigger flush")
    timeout: int = Field(default=60, description="HTTP timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    max_latency_ms: int = Field(
        default=100, description="Maximum acceptable latency in milliseconds"
    )
    enable_sampling: bool = Field(
        default=True, description="Enable intelligent sampling"
    )
    buffer_size: int = Field(default=1000, description="Buffer size for traces")
    environment: Optional[str] = Field(
        default=None, description="Environment name (e.g., production, staging)"
    )
    release: Optional[str] = Field(default=None, description="Release version")
    tracing_enabled: bool = Field(
        default=True, description="Enable OpenTelemetry tracing"
    )
    media_upload_thread_count: int = Field(
        default=4, description="Number of threads for media uploads"
    )
    blocked_instrumentation_scopes: Optional[str] = Field(
        default=None,
        description="Comma-separated list of blocked instrumentation scopes",
    )
    additional_headers: Optional[str] = Field(
        default=None, description="Additional HTTP headers as JSON string"
    )

    # Security settings
    enable_data_masking: bool = Field(
        default=True, description="Enable data masking for sensitive information"
    )
    pii_filtering_enabled: bool = Field(
        default=True, description="Enable PII filtering in traces"
    )
    data_retention_days: int = Field(
        default=90, description="Default data retention period in days"
    )
    secure_credential_storage: bool = Field(
        default=True, description="Use secure credential storage"
    )
    audit_trace_access: bool = Field(
        default=False, description="Enable audit logging for trace access"
    )
    allowed_trace_types: Optional[str] = Field(
        default=None, description="Comma-separated list of allowed trace types"
    )
    encryption_enabled: bool = Field(
        default=False, description="Enable encryption for sensitive trace data"
    )


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application settings
    app_name: str = Field(
        default="AutoOps Retail Optimization", description="Application name"
    )
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(
        default="development", description="Environment (development/production)"
    )

    # API settings (flattened for convenience)
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    log_level: str = Field(default="INFO", description="Logging level")
    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="CORS allowed origins",
    )

    # Component settings
    aws: AWSSettings = Field(default_factory=AWSSettings)
    bedrock: BedrockSettings = Field(default_factory=BedrockSettings)
    chromadb: ChromaDBSettings = Field(default_factory=ChromaDBSettings)
    api: APISettings = Field(default_factory=APISettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    strands: StrandsSettings = Field(default_factory=StrandsSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
