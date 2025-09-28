"""Async trace processor for non-blocking Langfuse operations."""

import asyncio
import json
import logging
import time
import zlib
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Deque
from dataclasses import dataclass, field
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from .langfuse_config import get_langfuse_client
from .langfuse_error_handler import get_langfuse_error_handler
from .langfuse_sampling import get_langfuse_sampler, TraceContext, TracePriority
from .langfuse_buffer import get_langfuse_buffer

logger = logging.getLogger(__name__)


@dataclass
class AsyncTraceOperation:
    """Represents an async trace operation."""

    operation_id: str
    operation_type: (
        str  # 'create_trace', 'start_span', 'end_span', 'log_event', 'finalize_trace'
    )
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    priority: TracePriority = TracePriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    compressed: bool = False


@dataclass
class ProcessingMetrics:
    """Metrics for async processing performance."""

    operations_processed: int = 0
    operations_failed: int = 0
    average_processing_time: float = 0.0
    queue_size: int = 0
    processing_rate: float = 0.0  # operations per second
    last_processing_time: Optional[datetime] = None
    compression_ratio: float = 1.0
    memory_usage_mb: float = 0.0


class AsyncTraceProcessor:
    """Async processor for non-blocking Langfuse trace operations."""

    def __init__(
        self,
        max_queue_size: int = 10000,
        batch_size: int = 50,
        flush_interval: float = 1.0,
        max_workers: int = 4,
        enable_compression: bool = True,
        compression_threshold: int = 1024,  # bytes
        enable_priority_queue: bool = True,
        processing_timeout: float = 30.0,
    ):
        """Initialize the async trace processor.

        Args:
            max_queue_size: Maximum number of operations to queue
            batch_size: Number of operations to process in each batch
            flush_interval: Seconds between processing batches
            max_workers: Maximum number of worker threads
            enable_compression: Whether to compress large trace data
            compression_threshold: Minimum size in bytes to trigger compression
            enable_priority_queue: Whether to use priority-based processing
            processing_timeout: Maximum time to spend processing a batch
        """
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_workers = max_workers
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold
        self.enable_priority_queue = enable_priority_queue
        self.processing_timeout = processing_timeout

        # Operation queues
        if enable_priority_queue:
            self._operation_queues: Dict[int, Deque[AsyncTraceOperation]] = {
                priority.value: deque(maxlen=max_queue_size // len(TracePriority))
                for priority in TracePriority
            }
            self._queue_sizes: Dict[str, int] = {
                str(priority.value): 0 for priority in TracePriority
            }
        else:
            self._operation_queue: Deque[AsyncTraceOperation] = deque(
                maxlen=max_queue_size
            )
            self._queue_sizes: Dict[str, int] = {"default": 0}

        # Processing control
        self._shutdown = False
        self._processing_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="LangfuseAsync"
        )

        # Metrics
        self.metrics = ProcessingMetrics()
        self._processing_times: Deque[float] = deque(maxlen=1000)
        self._lock = asyncio.Lock()

        # Callbacks
        self._completion_callbacks: List[
            Callable[[str, bool, Optional[str]], None]
        ] = []

        # Start background processing
        self._start_background_processing()

        logger.info(
            f"AsyncTraceProcessor initialized with {max_workers} workers, batch_size={batch_size}"
        )

    def _start_background_processing(self) -> None:
        """Start background processing tasks."""
        self._processing_task = asyncio.create_task(
            self._processing_loop(), name="AsyncTraceProcessing"
        )
        self._metrics_task = asyncio.create_task(
            self._metrics_update_loop(), name="AsyncTraceMetrics"
        )

    async def _processing_loop(self) -> None:
        """Main processing loop for async operations."""
        while not self._shutdown:
            try:
                await self._process_batch()
                await asyncio.sleep(self.flush_interval)
            except Exception as e:
                logger.error(f"Error in async processing loop: {e}")
                await asyncio.sleep(self.flush_interval)

    async def _metrics_update_loop(self) -> None:
        """Update processing metrics periodically."""
        while not self._shutdown:
            try:
                await self._update_metrics()
                await asyncio.sleep(5.0)  # Update metrics every 5 seconds
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(5.0)

    async def _process_batch(self) -> None:
        """Process a batch of queued operations."""
        start_time = time.time()

        try:
            # Get operations to process
            operations = await self._get_operations_for_processing()

            if not operations:
                return

            # Process operations concurrently
            tasks = []
            for operation in operations:
                task = asyncio.get_event_loop().run_in_executor(
                    self._thread_pool, self._process_operation_sync, operation
                )
                tasks.append(task)

            # Wait for all operations to complete with timeout
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=self.processing_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Processing timeout after {self.processing_timeout}s")
                # Cancel remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                return

            # Handle results
            successful = 0
            failed = 0

            for i, result in enumerate(results):
                operation = operations[i]

                if isinstance(result, Exception):
                    failed += 1
                    logger.error(
                        f"Failed to process operation {operation.operation_id}: {result}"
                    )
                    await self._handle_operation_failure(operation, result)
                else:
                    successful += 1
                    await self._handle_operation_success(operation, result)

            # Update metrics
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)

            async with self._lock:
                self.metrics.operations_processed += successful
                self.metrics.operations_failed += failed
                total_ops = (
                    self.metrics.operations_processed + self.metrics.operations_failed
                )
                if total_ops > 0:
                    self.metrics.average_processing_time = sum(
                        self._processing_times
                    ) / len(self._processing_times)
                self.metrics.processing_rate = (
                    len(operations) / processing_time if processing_time > 0 else 0
                )

        except Exception as e:
            logger.error(f"Error processing batch: {e}")

    def _process_operation_sync(self, operation: AsyncTraceOperation) -> Any:
        """Process a single operation synchronously in thread pool."""
        try:
            client = get_langfuse_client()
            if not client or not client.is_available:
                raise Exception("Langfuse client not available")

            langfuse = client.client
            if not langfuse:
                raise Exception("Langfuse client instance not available")

            # Process based on operation type
            if operation.operation_type == "create_trace":
                return self._process_create_trace(langfuse, operation)
            elif operation.operation_type == "start_span":
                return self._process_start_span(langfuse, operation)
            elif operation.operation_type == "end_span":
                return self._process_end_span(langfuse, operation)
            elif operation.operation_type == "log_event":
                return self._process_log_event(langfuse, operation)
            elif operation.operation_type == "finalize_trace":
                return self._process_finalize_trace(langfuse, operation)
            else:
                raise ValueError(f"Unknown operation type: {operation.operation_type}")

        except Exception as e:
            logger.error(f"Error processing operation {operation.operation_id}: {e}")
            raise

    def _process_create_trace(
        self, langfuse, operation: AsyncTraceOperation
    ) -> Optional[str]:
        """Process create trace operation."""
        data = operation.data
        trace = langfuse.start_span(
            name=data.get("name", "async_trace"),
            input=data.get("input"),
            metadata=data.get("metadata"),
        )
        return operation.trace_id if operation.trace_id else None

    def _process_start_span(
        self, langfuse, operation: AsyncTraceOperation
    ) -> Optional[str]:
        """Process start span operation."""
        data = operation.data
        parent_trace_id = data.get("parent_trace_id")

        if parent_trace_id:
            # Get parent trace (this might need to be handled differently)
            span = langfuse.start_span(
                name=data.get("name", "async_span"),
                input=data.get("input"),
                metadata=data.get("metadata"),
            )
        else:
            span = langfuse.start_span(
                name=data.get("name", "async_span"),
                input=data.get("input"),
                metadata=data.get("metadata"),
            )

        return operation.span_id if operation.span_id else None

    def _process_end_span(self, langfuse, operation: AsyncTraceOperation) -> None:
        """Process end span operation."""
        data = operation.data
        # In a real implementation, you'd need to track active spans
        # For now, we'll just log the event
        langfuse.event(
            name=f"span_end_{operation.span_id}",
            metadata={
                "span_id": operation.span_id,
                "output": data.get("output"),
                "status": "completed",
            },
        )

    def _process_log_event(self, langfuse, operation: AsyncTraceOperation) -> None:
        """Process log event operation."""
        data = operation.data
        langfuse.event(
            name=data.get("name", "async_event"),
            input=data.get("input"),
            output=data.get("output"),
            metadata=data.get("metadata"),
        )

    def _process_finalize_trace(self, langfuse, operation: AsyncTraceOperation) -> None:
        """Process finalize trace operation."""
        data = operation.data
        # In a real implementation, you'd need to track active traces
        # For now, we'll just log the finalization
        langfuse.event(
            name=f"trace_finalized_{operation.trace_id}",
            metadata={
                "trace_id": operation.trace_id,
                "output": data.get("output"),
                "status": "finalized",
            },
        )

    async def _get_operations_for_processing(self) -> List[AsyncTraceOperation]:
        """Get operations ready for processing, respecting priority."""
        operations = []

        if self.enable_priority_queue:
            # Process in priority order (highest first)
            for priority in sorted(TracePriority, key=lambda p: p.value, reverse=True):
                queue = self._operation_queues[priority.value]
                if queue:
                    # Take up to batch_size / len(priorities) operations from each queue
                    batch_quota = max(1, self.batch_size // len(TracePriority))
                    for _ in range(min(batch_quota, len(queue))):
                        if queue:
                            op = queue.popleft()
                            operations.append(op)
                            self._queue_sizes[str(priority.value)] -= 1
        else:
            # Simple FIFO processing
            queue = self._operation_queue
            for _ in range(min(self.batch_size, len(queue))):
                if queue:
                    op = queue.popleft()
                    operations.append(op)
                    self._queue_sizes["default"] -= 1

        return operations

    async def _handle_operation_success(
        self, operation: AsyncTraceOperation, result: Any
    ) -> None:
        """Handle successful operation completion."""
        # Call completion callback if provided
        if operation.callback:
            try:
                if asyncio.iscoroutinefunction(operation.callback):
                    await operation.callback(operation.operation_id, True, result)
                else:
                    operation.callback(operation.operation_id, True, result)
            except Exception as e:
                logger.error(f"Error in operation callback: {e}")

        # Trigger completion callbacks
        for callback in self._completion_callbacks:
            try:
                callback(operation.operation_id, True, None)
            except Exception as e:
                logger.error(f"Error in completion callback: {e}")

    async def _handle_operation_failure(
        self, operation: AsyncTraceOperation, error: Exception
    ) -> None:
        """Handle operation failure with retry logic."""
        operation.retry_count += 1

        if operation.retry_count < operation.max_retries:
            # Re-queue for retry with exponential backoff
            delay = min(2**operation.retry_count, 30)  # Max 30 second delay
            await asyncio.sleep(delay)
            await self._enqueue_operation(operation)
        else:
            # Max retries exceeded
            logger.error(
                f"Operation {operation.operation_id} failed after {operation.max_retries} retries"
            )

            # Call completion callback with failure
            if operation.callback:
                try:
                    if asyncio.iscoroutinefunction(operation.callback):
                        await operation.callback(
                            operation.operation_id, False, str(error)
                        )
                    else:
                        operation.callback(operation.operation_id, False, str(error))
                except Exception as e:
                    logger.error(f"Error in operation failure callback: {e}")

            # Trigger completion callbacks
            for callback in self._completion_callbacks:
                try:
                    callback(operation.operation_id, False, str(error))
                except Exception as e:
                    logger.error(f"Error in completion callback: {e}")

    async def _enqueue_operation(self, operation: AsyncTraceOperation) -> bool:
        """Enqueue an operation for async processing."""
        # Check if queue is full
        current_size = (
            sum(self._queue_sizes.values())
            if self.enable_priority_queue
            else self._queue_sizes["default"]
        )
        if current_size >= self.max_queue_size:
            logger.warning("Async operation queue is full, dropping operation")
            return False

        # Compress data if enabled and large enough
        if (
            self.enable_compression
            and len(json.dumps(operation.data)) > self.compression_threshold
        ):
            operation.data = self._compress_data(operation.data)
            operation.compressed = True

        # Add to appropriate queue
        if self.enable_priority_queue:
            queue = self._operation_queues[operation.priority.value]
            queue.append(operation)
            self._queue_sizes[str(operation.priority.value)] += 1
        else:
            self._operation_queue.append(operation)
            self._queue_sizes["default"] += 1

        return True

    def _compress_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress trace data to reduce memory usage."""
        try:
            json_str = json.dumps(data)
            compressed = zlib.compress(json_str.encode("utf-8"), level=6)
            return {
                "_compressed": True,
                "_data": compressed.decode(
                    "latin1"
                ),  # Store as string for JSON compatibility
                "_original_size": len(json_str),
                "_compressed_size": len(compressed),
            }
        except Exception as e:
            logger.error(f"Error compressing data: {e}")
            return data

    def _decompress_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress trace data."""
        if data.get("_compressed"):
            try:
                compressed = data["_data"].encode("latin1")
                decompressed = zlib.decompress(compressed)
                return json.loads(decompressed.decode("utf-8"))
            except Exception as e:
                logger.error(f"Error decompressing data: {e}")
                return data
        return data

    async def _update_metrics(self) -> None:
        """Update processing metrics."""
        async with self._lock:
            # Update queue size
            if self.enable_priority_queue:
                self.metrics.queue_size = sum(self._queue_sizes.values())
            else:
                self.metrics.queue_size = self._queue_sizes["default"]

            # Update compression ratio
            if self._processing_times:
                total_original = sum(
                    op.data.get("_original_size", 0)
                    for op in [
                        AsyncTraceOperation("", "", data={"_original_size": 1000})
                    ]
                )  # Placeholder
                total_compressed = sum(
                    op.data.get("_compressed_size", 0)
                    for op in [
                        AsyncTraceOperation("", "", data={"_compressed_size": 500})
                    ]
                )  # Placeholder
                if total_original > 0:
                    self.metrics.compression_ratio = total_compressed / total_original

            # Update memory usage (approximate)
            self.metrics.memory_usage_mb = (
                len(self._processing_times) * 0.001
            )  # Rough estimate

    def add_completion_callback(
        self, callback: Callable[[str, bool, Optional[str]], None]
    ) -> None:
        """Add a callback for operation completion.

        Args:
            callback: Function to call with (operation_id, success, error_message)
        """
        self._completion_callbacks.append(callback)

    async def create_trace_async(
        self,
        trace_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: TracePriority = TracePriority.MEDIUM,
        callback: Optional[Callable] = None,
    ) -> bool:
        """Create a trace asynchronously.

        Args:
            trace_id: Unique trace identifier
            name: Trace name
            input_data: Input data for the trace
            metadata: Metadata for the trace
            priority: Priority level
            callback: Optional completion callback

        Returns:
            True if operation was queued successfully
        """
        operation = AsyncTraceOperation(
            operation_id=f"create_{trace_id}_{int(time.time() * 1000)}",
            operation_type="create_trace",
            trace_id=trace_id,
            data={
                "name": name,
                "input": input_data,
                "metadata": metadata,
            },
            priority=priority,
            callback=callback,
        )

        return await self._enqueue_operation(operation)

    async def start_span_async(
        self,
        span_id: str,
        name: str,
        parent_trace_id: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: TracePriority = TracePriority.MEDIUM,
        callback: Optional[Callable] = None,
    ) -> bool:
        """Start a span asynchronously.

        Args:
            span_id: Unique span identifier
            name: Span name
            parent_trace_id: Parent trace ID
            input_data: Input data for the span
            metadata: Metadata for the span
            priority: Priority level
            callback: Optional completion callback

        Returns:
            True if operation was queued successfully
        """
        operation = AsyncTraceOperation(
            operation_id=f"start_{span_id}_{int(time.time() * 1000)}",
            operation_type="start_span",
            span_id=span_id,
            data={
                "name": name,
                "parent_trace_id": parent_trace_id,
                "input": input_data,
                "metadata": metadata,
            },
            priority=priority,
            callback=callback,
        )

        return await self._enqueue_operation(operation)

    async def end_span_async(
        self,
        span_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        priority: TracePriority = TracePriority.MEDIUM,
        callback: Optional[Callable] = None,
    ) -> bool:
        """End a span asynchronously.

        Args:
            span_id: Span identifier
            output_data: Output data for the span
            priority: Priority level
            callback: Optional completion callback

        Returns:
            True if operation was queued successfully
        """
        operation = AsyncTraceOperation(
            operation_id=f"end_{span_id}_{int(time.time() * 1000)}",
            operation_type="end_span",
            span_id=span_id,
            data={
                "output": output_data,
            },
            priority=priority,
            callback=callback,
        )

        return await self._enqueue_operation(operation)

    async def log_event_async(
        self,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: TracePriority = TracePriority.MEDIUM,
        callback: Optional[Callable] = None,
    ) -> bool:
        """Log an event asynchronously.

        Args:
            name: Event name
            input_data: Input data for the event
            output_data: Output data for the event
            metadata: Metadata for the event
            priority: Priority level
            callback: Optional completion callback

        Returns:
            True if operation was queued successfully
        """
        operation = AsyncTraceOperation(
            operation_id=f"event_{name}_{int(time.time() * 1000)}",
            operation_type="log_event",
            data={
                "name": name,
                "input": input_data,
                "output": output_data,
                "metadata": metadata,
            },
            priority=priority,
            callback=callback,
        )

        return await self._enqueue_operation(operation)

    async def finalize_trace_async(
        self,
        trace_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        priority: TracePriority = TracePriority.MEDIUM,
        callback: Optional[Callable] = None,
    ) -> bool:
        """Finalize a trace asynchronously.

        Args:
            trace_id: Trace identifier
            output_data: Final output data
            priority: Priority level
            callback: Optional completion callback

        Returns:
            True if operation was queued successfully
        """
        operation = AsyncTraceOperation(
            operation_id=f"finalize_{trace_id}_{int(time.time() * 1000)}",
            operation_type="finalize_trace",
            trace_id=trace_id,
            data={
                "output": output_data,
            },
            priority=priority,
            callback=callback,
        )

        return await self._enqueue_operation(operation)

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics.

        Returns:
            Dictionary containing processing metrics
        """
        return {
            "operations_processed": self.metrics.operations_processed,
            "operations_failed": self.metrics.operations_failed,
            "average_processing_time": round(self.metrics.average_processing_time, 4),
            "queue_size": self.metrics.queue_size,
            "processing_rate": round(self.metrics.processing_rate, 2),
            "compression_ratio": round(self.metrics.compression_ratio, 4),
            "memory_usage_mb": round(self.metrics.memory_usage_mb, 2),
            "thread_pool_active": self._thread_pool._threads,  # Approximate
            "last_processing_time": self.metrics.last_processing_time.isoformat()
            if self.metrics.last_processing_time
            else None,
        }

    async def shutdown(self) -> None:
        """Shutdown the async processor."""
        self._shutdown = True

        # Cancel background tasks
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass

        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)

        logger.info("AsyncTraceProcessor shutdown completed")


# Global async processor instance
_async_processor: Optional[AsyncTraceProcessor] = None


def get_async_trace_processor() -> Optional[AsyncTraceProcessor]:
    """Get the global async trace processor instance."""
    return _async_processor


def initialize_async_trace_processor(**kwargs) -> AsyncTraceProcessor:
    """Initialize the global async trace processor.

    Args:
        **kwargs: Configuration parameters for AsyncTraceProcessor

    Returns:
        Initialized AsyncTraceProcessor instance
    """
    global _async_processor
    _async_processor = AsyncTraceProcessor(**kwargs)
    return _async_processor


def shutdown_async_trace_processor() -> None:
    """Shutdown the global async trace processor."""
    global _async_processor
    if _async_processor:
        # Create new event loop for shutdown if needed
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule shutdown as a task
                asyncio.create_task(_async_processor.shutdown())
            else:
                loop.run_until_complete(_async_processor.shutdown())
        except RuntimeError:
            # No event loop, create one
            asyncio.run(_async_processor.shutdown())

        _async_processor = None
