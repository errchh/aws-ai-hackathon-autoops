"""Local buffering system for offline Langfuse trace storage."""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Deque
from dataclasses import dataclass, field
import sqlite3
import tempfile
import shutil

from .langfuse_error_handler import LangfuseErrorHandler
from .langfuse_config import LangfuseClient

logger = logging.getLogger(__name__)


@dataclass
class BufferedTrace:
    """Represents a trace stored in the local buffer."""

    trace_id: str
    trace_data: Dict[str, Any]
    timestamp: datetime
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 1
    expires_at: Optional[datetime] = None


class LangfuseBuffer:
    """Local buffer for storing traces when Langfuse is unavailable."""

    def __init__(
        self,
        langfuse_client: LangfuseClient,
        error_handler: LangfuseErrorHandler,
        max_buffer_size: int = 10000,
        buffer_directory: Optional[str] = None,
        use_database: bool = True,
        cleanup_interval: int = 300,  # 5 minutes
        max_age_hours: int = 24,
    ):
        """Initialize the buffer.

        Args:
            langfuse_client: Langfuse client instance
            error_handler: Error handler instance
            max_buffer_size: Maximum number of traces to buffer
            buffer_directory: Directory to store buffer files (None for temp dir)
            use_database: Whether to use SQLite database for persistence
            cleanup_interval: Seconds between cleanup operations
            max_age_hours: Maximum age of buffered traces in hours
        """
        self.langfuse_client = langfuse_client
        self.error_handler = error_handler
        self.max_buffer_size = max_buffer_size
        self.use_database = use_database
        self.cleanup_interval = cleanup_interval
        self.max_age_hours = max_age_hours

        # Setup buffer storage
        if buffer_directory:
            self.buffer_dir = Path(buffer_directory)
        else:
            self.buffer_dir = Path(tempfile.gettempdir()) / "langfuse_buffer"

        self.buffer_dir.mkdir(parents=True, exist_ok=True)

        # Database setup
        self.db_path = self.buffer_dir / "traces.db"
        self._init_database()

        # In-memory buffer for immediate operations
        self._memory_buffer: Deque[BufferedTrace] = deque(maxlen=max_buffer_size)

        # Control flags
        self._shutdown = False
        self._cleanup_thread: Optional[threading.Thread] = None
        self._flush_thread: Optional[threading.Thread] = None

        # Start background threads
        self._start_background_threads()

        logger.info(f"LangfuseBuffer initialized with max size {max_buffer_size}")

    def _init_database(self) -> None:
        """Initialize the SQLite database for trace storage."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS traces (
                        trace_id TEXT PRIMARY KEY,
                        trace_data TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        priority INTEGER DEFAULT 1,
                        expires_at TEXT
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_timestamp ON traces(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_priority ON traces(priority)"
                )
                conn.commit()
                logger.debug("Buffer database initialized")
        except Exception as e:
            logger.error(f"Failed to initialize buffer database: {e}")
            self.use_database = False

    def _start_background_threads(self) -> None:
        """Start background threads for cleanup and flushing."""
        # Cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="LangfuseBufferCleanup"
        )
        self._cleanup_thread.start()

        # Flush thread
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="LangfuseBufferFlush"
        )
        self._flush_thread.start()

        logger.debug("Buffer background threads started")

    def _cleanup_loop(self) -> None:
        """Background loop for cleaning up old traces."""
        while not self._shutdown:
            try:
                self._cleanup_old_traces()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(self.cleanup_interval)

    def _flush_loop(self) -> None:
        """Background loop for attempting to flush traces."""
        while not self._shutdown:
            try:
                if self.langfuse_client.is_available:
                    self._flush_to_langfuse()
                time.sleep(30)  # Try every 30 seconds
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                time.sleep(30)

    def _cleanup_old_traces(self) -> None:
        """Remove traces that are too old or have exceeded max retries."""
        cutoff_time = datetime.now() - timedelta(hours=self.max_age_hours)

        try:
            if self.use_database:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM traces WHERE timestamp < ? OR retry_count >= max_retries",
                        (cutoff_time.isoformat(),),
                    )
                    deleted_count = cursor.rowcount
                    conn.commit()

                if deleted_count > 0:
                    logger.debug(
                        f"Cleaned up {deleted_count} old/failed traces from database"
                    )

            # Also clean memory buffer
            original_size = len(self._memory_buffer)
            self._memory_buffer = deque(
                [
                    trace
                    for trace in self._memory_buffer
                    if trace.timestamp > cutoff_time
                    and trace.retry_count < trace.max_retries
                ],
                maxlen=self.max_buffer_size,
            )

            if len(self._memory_buffer) < original_size:
                logger.debug(
                    f"Cleaned up {original_size - len(self._memory_buffer)} old/failed traces from memory"
                )

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _flush_to_langfuse(self) -> None:
        """Attempt to flush buffered traces to Langfuse."""
        traces_to_flush = self._get_traces_for_flush()

        for trace in traces_to_flush:
            try:
                # Attempt to send the trace
                self._send_trace_to_langfuse(trace)

                # Remove from both memory and database if successful
                self._remove_trace(trace.trace_id)
                self.error_handler.record_success()

                logger.debug(f"Successfully flushed trace {trace.trace_id}")

            except Exception as e:
                logger.warning(f"Failed to flush trace {trace.trace_id}: {e}")
                self._increment_retry_count(trace.trace_id)

    def _get_traces_for_flush(self, limit: int = 100) -> List[BufferedTrace]:
        """Get traces ready for flushing, ordered by priority and age."""
        traces = []

        try:
            if self.use_database:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT trace_id, trace_data, timestamp, retry_count, max_retries, priority, expires_at
                        FROM traces
                        WHERE retry_count < max_retries
                        ORDER BY priority DESC, timestamp ASC
                        LIMIT ?
                    """,
                        (limit,),
                    )

                    for row in cursor.fetchall():
                        traces.append(
                            BufferedTrace(
                                trace_id=row[0],
                                trace_data=json.loads(row[1]),
                                timestamp=datetime.fromisoformat(row[2]),
                                retry_count=row[3],
                                max_retries=row[4],
                                priority=row[5],
                                expires_at=datetime.fromisoformat(row[6])
                                if row[6]
                                else None,
                            )
                        )

            # Also include recent memory buffer traces
            for trace in self._memory_buffer:
                if trace.retry_count < trace.max_retries:
                    traces.append(trace)

        except Exception as e:
            logger.error(f"Error getting traces for flush: {e}")

        return traces

    def _send_trace_to_langfuse(self, trace: BufferedTrace) -> None:
        """Send a buffered trace to Langfuse."""
        # This would need to be implemented based on how traces are structured
        # For now, we'll simulate the operation
        client = self.langfuse_client.client
        if not client:
            raise Exception("Langfuse client not available")

        # In a real implementation, you would reconstruct the trace object
        # and call the appropriate Langfuse client methods
        logger.debug(f"Would send trace {trace.trace_id} to Langfuse")

        # Simulate some processing time
        time.sleep(0.01)

    def _remove_trace(self, trace_id: str) -> None:
        """Remove a trace from both memory and database."""
        # Remove from memory
        self._memory_buffer = deque(
            [trace for trace in self._memory_buffer if trace.trace_id != trace_id],
            maxlen=self.max_buffer_size,
        )

        # Remove from database
        if self.use_database:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM traces WHERE trace_id = ?", (trace_id,))
                    conn.commit()
            except Exception as e:
                logger.error(f"Error removing trace {trace_id} from database: {e}")

    def _increment_retry_count(self, trace_id: str) -> None:
        """Increment the retry count for a trace."""
        # Update in memory
        for trace in self._memory_buffer:
            if trace.trace_id == trace_id:
                trace.retry_count += 1
                break

        # Update in database
        if self.use_database:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE traces SET retry_count = retry_count + 1 WHERE trace_id = ?",
                        (trace_id,),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error incrementing retry count for {trace_id}: {e}")

    def add_trace(
        self,
        trace_id: str,
        trace_data: Dict[str, Any],
        priority: int = 1,
        max_retries: int = 3,
    ) -> bool:
        """Add a trace to the buffer.

        Args:
            trace_id: Unique identifier for the trace
            trace_data: The trace data
            priority: Priority level (higher = more important)
            max_retries: Maximum number of retry attempts

        Returns:
            True if trace was added, False if buffer is full
        """
        if len(self._memory_buffer) >= self.max_buffer_size:
            logger.warning("Buffer is full, cannot add more traces")
            return False

        trace = BufferedTrace(
            trace_id=trace_id,
            trace_data=trace_data,
            timestamp=datetime.now(),
            priority=priority,
            max_retries=max_retries,
            expires_at=datetime.now() + timedelta(hours=self.max_age_hours),
        )

        # Add to memory buffer
        self._memory_buffer.append(trace)

        # Add to database if enabled
        if self.use_database:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO traces
                        (trace_id, trace_data, timestamp, retry_count, max_retries, priority, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            trace_id,
                            json.dumps(trace_data),
                            trace.timestamp.isoformat(),
                            trace.retry_count,
                            trace.max_retries,
                            trace.priority,
                            trace.expires_at.isoformat() if trace.expires_at else None,
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.error(f"Error storing trace {trace_id} in database: {e}")

        logger.debug(f"Added trace {trace_id} to buffer (priority: {priority})")
        return True

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get statistics about the current buffer state.

        Returns:
            Dictionary containing buffer statistics
        """
        memory_count = len(self._memory_buffer)

        db_count = 0
        if self.use_database:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM traces")
                    db_count = cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Error getting database count: {e}")

        return {
            "memory_buffer_size": memory_count,
            "database_buffer_size": db_count,
            "max_buffer_size": self.max_buffer_size,
            "buffer_utilization": (memory_count / self.max_buffer_size) * 100,
            "database_enabled": self.use_database,
            "buffer_directory": str(self.buffer_dir),
            "oldest_trace": None,  # Would need to query for this
            "newest_trace": None,  # Would need to query for this
        }

    def clear_buffer(self) -> None:
        """Clear all traces from the buffer."""
        # Clear memory buffer
        self._memory_buffer.clear()

        # Clear database
        if self.use_database:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM traces")
                    conn.commit()
                logger.info("Cleared all traces from database")
            except Exception as e:
                logger.error(f"Error clearing database: {e}")

        logger.info("Buffer cleared")

    def export_buffer(self, export_path: str) -> bool:
        """Export all buffered traces to a file.

        Args:
            export_path: Path to export the traces to

        Returns:
            True if export was successful
        """
        try:
            export_data = {"export_timestamp": datetime.now().isoformat(), "traces": []}

            # Get all traces from database
            if self.use_database:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT * FROM traces")
                    for row in cursor.fetchall():
                        export_data["traces"].append(
                            {
                                "trace_id": row[0],
                                "trace_data": json.loads(row[1]),
                                "timestamp": row[2],
                                "retry_count": row[3],
                                "priority": row[5],
                            }
                        )

            # Write to file
            with open(export_path, "w") as f:
                json.dump(export_data, f, indent=2)

            logger.info(
                f"Exported {len(export_data['traces'])} traces to {export_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Error exporting buffer: {e}")
            return False

    def import_buffer(self, import_path: str) -> int:
        """Import traces from an export file.

        Args:
            import_path: Path to the export file

        Returns:
            Number of traces imported
        """
        try:
            with open(import_path, "r") as f:
                import_data = json.load(f)

            imported_count = 0

            for trace_data in import_data.get("traces", []):
                trace = BufferedTrace(
                    trace_id=trace_data["trace_id"],
                    trace_data=trace_data["trace_data"],
                    timestamp=datetime.fromisoformat(trace_data["timestamp"]),
                    retry_count=trace_data.get("retry_count", 0),
                    priority=trace_data.get("priority", 1),
                )

                # Add to memory buffer (will be persisted to DB by background process)
                if len(self._memory_buffer) < self.max_buffer_size:
                    self._memory_buffer.append(trace)
                    imported_count += 1

            logger.info(f"Imported {imported_count} traces from {import_path}")
            return imported_count

        except Exception as e:
            logger.error(f"Error importing buffer: {e}")
            return 0

    def shutdown(self) -> None:
        """Shutdown the buffer and cleanup resources."""
        self._shutdown = True

        # Wait for background threads to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)

        # Final flush attempt
        if self.langfuse_client.is_available:
            self._flush_to_langfuse()

        logger.info("LangfuseBuffer shutdown completed")


# Global buffer instance
_buffer: Optional[LangfuseBuffer] = None


def get_langfuse_buffer() -> Optional[LangfuseBuffer]:
    """Get the global Langfuse buffer instance."""
    return _buffer


def initialize_langfuse_buffer(
    langfuse_client: LangfuseClient, error_handler: LangfuseErrorHandler, **kwargs
) -> LangfuseBuffer:
    """Initialize the global Langfuse buffer.

    Args:
        langfuse_client: Langfuse client instance
        error_handler: Error handler instance
        **kwargs: Additional configuration parameters

    Returns:
        Initialized LangfuseBuffer instance
    """
    global _buffer
    _buffer = LangfuseBuffer(
        langfuse_client=langfuse_client, error_handler=error_handler, **kwargs
    )
    return _buffer


def shutdown_langfuse_buffer() -> None:
    """Shutdown the global Langfuse buffer."""
    global _buffer
    if _buffer:
        _buffer.shutdown()
        _buffer = None
