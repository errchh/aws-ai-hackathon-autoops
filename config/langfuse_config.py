"""Langfuse integration configuration and client management."""

import logging
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

from langfuse import Langfuse, observe
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LangfuseConfig:
    """Langfuse configuration data class for Langfuse v3."""

    public_key: str
    secret_key: str
    host: str = "https://cloud.langfuse.com"
    enabled: bool = True
    sample_rate: float = 1.0
    debug: bool = False
    flush_interval: float = 5.0  # Changed to float for v3 compatibility
    flush_at: int = 15  # New v3 parameter - number of events to trigger flush
    timeout: int = 60  # New v3 parameter - HTTP timeout in seconds
    max_retries: int = 3
    max_latency_ms: int = 100
    enable_sampling: bool = True
    buffer_size: int = 1000
    environment: Optional[str] = None  # New v3 parameter
    release: Optional[str] = None  # New v3 parameter
    tracing_enabled: bool = True  # New v3 parameter
    media_upload_thread_count: int = 4  # New v3 parameter
    blocked_instrumentation_scopes: Optional[List[str]] = None  # New v3 parameter
    additional_headers: Optional[Dict[str, str]] = None  # New v3 parameter

    # Security configuration
    enable_data_masking: bool = True
    pii_filtering_enabled: bool = True
    data_retention_days: int = 90
    secure_credential_storage: bool = True
    audit_trace_access: bool = False
    allowed_trace_types: Optional[List[str]] = None
    encryption_enabled: bool = False


class LangfuseClientError(Exception):
    """Custom exception for Langfuse client errors."""

    pass


class LangfuseClient:
    """Langfuse client wrapper with error handling and configuration management."""

    def __init__(self, config: Optional[LangfuseConfig] = None):
        """Initialize Langfuse client with configuration.

        Args:
            config: Optional Langfuse configuration. If None, loads from settings.
        """
        self._client: Optional[Langfuse] = None
        self._config: Optional[LangfuseConfig] = config
        self._is_initialized = False
        self._fallback_mode = False

        if config is None:
            self._load_config_from_settings()

        # Initialize client if configuration is available
        if self._config and self._config.enabled:
            self._initialize_client()

    def _load_config_from_settings(self) -> None:
        """Load Langfuse configuration from application settings."""
        try:
            settings = get_settings()
            langfuse_settings = settings.langfuse

            # Check if required credentials are available
            if not langfuse_settings.public_key or not langfuse_settings.secret_key:
                logger.warning(
                    "Langfuse credentials not found in settings. "
                    "Tracing will be disabled."
                )
                self._config = LangfuseConfig(
                    public_key="", secret_key="", enabled=False
                )
                return

            # Parse additional headers from JSON string if provided
            additional_headers = None
            if langfuse_settings.additional_headers:
                try:
                    additional_headers = json.loads(
                        langfuse_settings.additional_headers
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse additional_headers JSON: {e}")

            # Parse blocked instrumentation scopes from comma-separated string
            blocked_scopes = None
            if langfuse_settings.blocked_instrumentation_scopes:
                blocked_scopes = [
                    scope.strip()
                    for scope in langfuse_settings.blocked_instrumentation_scopes.split(
                        ","
                    )
                    if scope.strip()
                ]

            self._config = LangfuseConfig(
                public_key=langfuse_settings.public_key,
                secret_key=langfuse_settings.secret_key,
                host=langfuse_settings.host,
                enabled=langfuse_settings.enabled,
                sample_rate=langfuse_settings.sample_rate,
                debug=langfuse_settings.debug,
                flush_interval=langfuse_settings.flush_interval,
                flush_at=langfuse_settings.flush_at,
                timeout=langfuse_settings.timeout,
                max_retries=langfuse_settings.max_retries,
                max_latency_ms=langfuse_settings.max_latency_ms,
                enable_sampling=langfuse_settings.enable_sampling,
                buffer_size=langfuse_settings.buffer_size,
                environment=langfuse_settings.environment,
                release=langfuse_settings.release,
                tracing_enabled=langfuse_settings.tracing_enabled,
                media_upload_thread_count=langfuse_settings.media_upload_thread_count,
                blocked_instrumentation_scopes=blocked_scopes,
                additional_headers=additional_headers,
                enable_data_masking=langfuse_settings.enable_data_masking,
                pii_filtering_enabled=langfuse_settings.pii_filtering_enabled,
                data_retention_days=langfuse_settings.data_retention_days,
                secure_credential_storage=langfuse_settings.secure_credential_storage,
                audit_trace_access=langfuse_settings.audit_trace_access,
                allowed_trace_types=langfuse_settings.allowed_trace_types.split(",")
                if langfuse_settings.allowed_trace_types
                else None,
                encryption_enabled=langfuse_settings.encryption_enabled,
            )

        except Exception as e:
            logger.error(f"Failed to load Langfuse configuration: {e}")
            self._config = LangfuseConfig(public_key="", secret_key="", enabled=False)

    def _initialize_client(self) -> None:
        """Initialize the Langfuse client with error handling."""
        if not self._config or not self._config.enabled:
            logger.info("Langfuse tracing is disabled")
            return

        try:
            # Initialize Langfuse v3 client with all supported parameters
            self._client = Langfuse(
                public_key=self._config.public_key,
                secret_key=self._config.secret_key,
                host=self._config.host,
                debug=self._config.debug,
                flush_interval=self._config.flush_interval,
                flush_at=self._config.flush_at,
                timeout=self._config.timeout,
                sample_rate=self._config.sample_rate,
                tracing_enabled=self._config.tracing_enabled,
                environment=self._config.environment,
                release=self._config.release,
                media_upload_thread_count=self._config.media_upload_thread_count,
                blocked_instrumentation_scopes=self._config.blocked_instrumentation_scopes,
                additional_headers=self._config.additional_headers,
            )

            # Test connection with a simple health check
            self._test_connection()

            self._is_initialized = True
            self._fallback_mode = False

            logger.info(
                f"Langfuse v3 client initialized successfully. "
                f"Host: {self._config.host}, "
                f"Sample rate: {self._config.sample_rate}, "
                f"Environment: {self._config.environment}, "
                f"Release: {self._config.release}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Langfuse client: {e}")
            self._handle_initialization_error(e)

    def _test_connection(self) -> None:
        """Test Langfuse connection by creating a simple span."""
        if not self._client:
            raise LangfuseClientError("Client not initialized")

        try:
            # Create a test span to verify connection
            test_span = self._client.start_span(
                name="connection_test",
                input={"test": True},
                metadata={"timestamp": datetime.now().isoformat()},
            )

            # Immediately flush to test the connection
            self._client.flush()

            logger.debug("Langfuse connection test successful")

        except Exception as e:
            raise LangfuseClientError(f"Connection test failed: {e}")

    def _handle_initialization_error(self, error: Exception) -> None:
        """Handle initialization errors with graceful degradation."""
        self._fallback_mode = True
        self._is_initialized = False

        logger.warning(
            f"Langfuse initialization failed, enabling fallback mode: {error}. "
            "System will continue without tracing."
        )

    @property
    def is_available(self) -> bool:
        """Check if Langfuse client is available and ready to use."""
        return (
            self._is_initialized
            and self._client is not None
            and not self._fallback_mode
            and self._config is not None
            and self._config.enabled
        )

    @property
    def client(self) -> Optional[Langfuse]:
        """Get the Langfuse client instance."""
        return self._client if self.is_available else None

    @property
    def config(self) -> Optional[LangfuseConfig]:
        """Get the current configuration."""
        return self._config

    def enable_fallback_mode(self) -> None:
        """Enable fallback mode to disable tracing."""
        self._fallback_mode = True
        logger.warning("Langfuse fallback mode enabled - tracing disabled")

    def disable_fallback_mode(self) -> None:
        """Disable fallback mode and attempt to re-initialize."""
        self._fallback_mode = False
        if self._config and self._config.enabled:
            self._initialize_client()

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check and return status information."""
        status = {
            "initialized": self._is_initialized,
            "available": self.is_available,
            "fallback_mode": self._fallback_mode,
            "enabled": self._config.enabled if self._config else False,
            "host": self._config.host if self._config else None,
            "sample_rate": self._config.sample_rate if self._config else None,
        }

        if self.is_available:
            try:
                # Test connection
                self._test_connection()
                status["connection_status"] = "healthy"
            except Exception as e:
                status["connection_status"] = f"unhealthy: {e}"
                logger.warning(f"Langfuse health check failed: {e}")
        else:
            status["connection_status"] = "unavailable"

        return status

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self.is_available and self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Failed to flush Langfuse traces: {e}")

    def shutdown(self) -> None:
        """Shutdown the Langfuse client gracefully."""
        if self._client:
            try:
                self._client.flush()
                logger.info("Langfuse client shutdown completed")
            except Exception as e:
                logger.error(f"Error during Langfuse client shutdown: {e}")
            finally:
                self._client = None
                self._is_initialized = False


# Global Langfuse client instance
_langfuse_client: Optional[LangfuseClient] = None


def get_langfuse_client() -> LangfuseClient:
    """Get the global Langfuse client instance."""
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = LangfuseClient()
    return _langfuse_client


def initialize_langfuse(config: Optional[LangfuseConfig] = None) -> LangfuseClient:
    """Initialize Langfuse client with optional configuration.

    Args:
        config: Optional Langfuse configuration

    Returns:
        Initialized LangfuseClient instance
    """
    global _langfuse_client
    _langfuse_client = LangfuseClient(config)
    return _langfuse_client


def shutdown_langfuse() -> None:
    """Shutdown the global Langfuse client."""
    global _langfuse_client
    if _langfuse_client:
        _langfuse_client.shutdown()
        _langfuse_client = None
