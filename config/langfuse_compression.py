"""Advanced compression utilities for Langfuse trace data."""

import json
import logging
import zlib
import gzip
import bz2
from enum import Enum
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import time

try:
    import lz4.frame

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    lz4 = None

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Available compression algorithms."""

    NONE = "none"
    GZIP = "gzip"
    BZ2 = "bz2"
    ZLIB = "zlib"
    LZ4 = "lz4"


@dataclass
class CompressionResult:
    """Result of compression operation."""

    algorithm: CompressionAlgorithm
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    data: bytes


class TraceCompressor:
    """Advanced compressor for Langfuse trace data with multiple algorithms."""

    def __init__(
        self,
        default_algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4,
        compression_threshold: int = 1024,  # bytes
        enable_adaptive: bool = True,
    ):
        """Initialize the trace compressor.

        Args:
            default_algorithm: Default compression algorithm to use
            compression_threshold: Minimum size in bytes to trigger compression
            enable_adaptive: Whether to adaptively choose the best algorithm
        """
        self.default_algorithm = default_algorithm
        self.compression_threshold = compression_threshold
        self.enable_adaptive = enable_adaptive

        # Performance tracking
        self.compression_stats = {
            algorithm: {
                "total_compressed": 0,
                "total_original_size": 0,
                "total_compressed_size": 0,
                "total_time_ms": 0.0,
            }
            for algorithm in CompressionAlgorithm
        }

    def should_compress(self, data: Union[str, Dict[str, Any]]) -> bool:
        """Determine if data should be compressed based on size threshold."""
        if isinstance(data, dict):
            json_str = json.dumps(data)
        else:
            json_str = str(data)

        return len(json_str.encode("utf-8")) >= self.compression_threshold

    def compress_data(
        self,
        data: Union[str, Dict[str, Any]],
        algorithm: Optional[CompressionAlgorithm] = None,
        min_ratio_improvement: float = 0.1,
    ) -> CompressionResult:
        """Compress data using the specified or adaptive algorithm.

        Args:
            data: Data to compress
            algorithm: Compression algorithm to use (None for adaptive)
            min_ratio_improvement: Minimum compression ratio improvement to accept

        Returns:
            CompressionResult with compressed data and metadata
        """
        # Convert to JSON string if dict
        if isinstance(data, dict):
            json_str = json.dumps(data, separators=(",", ":"), sort_keys=True)
        else:
            json_str = str(data)

        original_bytes = json_str.encode("utf-8")
        original_size = len(original_bytes)

        # Skip compression if too small
        if original_size < self.compression_threshold:
            return CompressionResult(
                algorithm=CompressionAlgorithm.NONE,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time_ms=0.0,
                data=original_bytes,
            )

        # Choose algorithm
        if algorithm is None and self.enable_adaptive:
            algorithm = self._choose_best_algorithm(json_str)
        elif algorithm is None:
            algorithm = self.default_algorithm

        # Compress
        start_time = time.time()
        try:
            if algorithm == CompressionAlgorithm.GZIP:
                compressed_bytes = gzip.compress(original_bytes, compresslevel=6)
            elif algorithm == CompressionAlgorithm.BZ2:
                compressed_bytes = bz2.compress(original_bytes, compresslevel=6)
            elif algorithm == CompressionAlgorithm.ZLIB:
                compressed_bytes = zlib.compress(original_bytes, level=6)
            elif algorithm == CompressionAlgorithm.LZ4:
                if LZ4_AVAILABLE and lz4:
                    compressed_bytes = lz4.frame.compress(
                        original_bytes,
                        compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
                    )
                else:
                    # Fallback to zlib if lz4 not available
                    compressed_bytes = zlib.compress(original_bytes, level=6)
            else:
                compressed_bytes = original_bytes

            compression_time_ms = (time.time() - start_time) * 1000
            compressed_size = len(compressed_bytes)
            compression_ratio = (
                compressed_size / original_size if original_size > 0 else 1.0
            )

            # Update stats
            self._update_compression_stats(
                algorithm, original_size, compressed_size, compression_time_ms
            )

            # Check if compression is beneficial
            if algorithm != CompressionAlgorithm.NONE and compression_ratio >= (
                1.0 - min_ratio_improvement
            ):
                # Compression not beneficial enough, return original
                return CompressionResult(
                    algorithm=CompressionAlgorithm.NONE,
                    original_size=original_size,
                    compressed_size=original_size,
                    compression_ratio=1.0,
                    compression_time_ms=compression_time_ms,
                    data=original_bytes,
                )

            return CompressionResult(
                algorithm=algorithm,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                compression_time_ms=compression_time_ms,
                data=compressed_bytes,
            )

        except Exception as e:
            logger.error(f"Compression failed with {algorithm.value}: {e}")
            return CompressionResult(
                algorithm=CompressionAlgorithm.NONE,
                original_size=original_size,
                compressed_size=original_size,
                compression_ratio=1.0,
                compression_time_ms=(time.time() - start_time) * 1000,
                data=original_bytes,
            )

    def decompress_data(
        self, compressed_result: CompressionResult
    ) -> Union[str, Dict[str, Any]]:
        """Decompress data from CompressionResult.

        Args:
            compressed_result: Result from compression operation

        Returns:
            Decompressed data
        """
        try:
            if compressed_result.algorithm == CompressionAlgorithm.NONE:
                data = compressed_result.data.decode("utf-8")
            elif compressed_result.algorithm == CompressionAlgorithm.GZIP:
                data = gzip.decompress(compressed_result.data).decode("utf-8")
            elif compressed_result.algorithm == CompressionAlgorithm.BZ2:
                data = bz2.decompress(compressed_result.data).decode("utf-8")
            elif compressed_result.algorithm == CompressionAlgorithm.ZLIB:
                data = zlib.decompress(compressed_result.data).decode("utf-8")
            elif compressed_result.algorithm == CompressionAlgorithm.LZ4:
                if LZ4_AVAILABLE and lz4:
                    data = lz4.frame.decompress(compressed_result.data).decode("utf-8")
                else:
                    # Fallback to zlib decompression
                    data = zlib.decompress(compressed_result.data).decode("utf-8")
            else:
                data = compressed_result.data.decode("utf-8")

            # Try to parse as JSON
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return data

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            # Return original data as fallback
            try:
                return json.loads(compressed_result.data.decode("utf-8"))
            except:
                return compressed_result.data.decode("utf-8")

    def _choose_best_algorithm(self, json_str: str) -> CompressionAlgorithm:
        """Choose the best compression algorithm for the given data."""
        original_bytes = json_str.encode("utf-8")
        original_size = len(original_bytes)

        # Test each algorithm and choose the best
        best_algorithm = CompressionAlgorithm.NONE
        best_ratio = 1.0

        algorithms_to_test = [
            CompressionAlgorithm.LZ4,  # Fastest
            CompressionAlgorithm.ZLIB,  # Good balance
            CompressionAlgorithm.GZIP,  # Good compression
        ]

        for algorithm in algorithms_to_test:
            try:
                start_time = time.time()
                compressed = original_bytes  # Default to no compression

                if algorithm == CompressionAlgorithm.LZ4:
                    if LZ4_AVAILABLE and lz4:
                        compressed = lz4.frame.compress(
                            original_bytes,
                            compression_level=lz4.frame.COMPRESSIONLEVEL_MINHC,
                        )
                    else:
                        # Skip lz4 if not available
                        continue
                elif algorithm == CompressionAlgorithm.ZLIB:
                    compressed = zlib.compress(original_bytes, level=6)
                elif algorithm == CompressionAlgorithm.GZIP:
                    compressed = gzip.compress(original_bytes, compresslevel=6)

                compression_time = time.time() - start_time
                compressed_size = len(compressed)
                ratio = compressed_size / original_size if original_size > 0 else 1.0

                # Consider both compression ratio and speed
                # Prefer faster algorithms for similar compression ratios
                if ratio < best_ratio or (
                    ratio < best_ratio * 1.05
                    and compression_time < 0.001  # 1ms threshold
                ):
                    best_ratio = ratio
                    best_algorithm = algorithm

            except Exception as e:
                logger.debug(f"Algorithm {algorithm.value} failed: {e}")
                continue

        return best_algorithm

    def _update_compression_stats(
        self,
        algorithm: CompressionAlgorithm,
        original_size: int,
        compressed_size: int,
        compression_time_ms: float,
    ) -> None:
        """Update compression statistics."""
        stats = self.compression_stats[algorithm]
        stats["total_compressed"] += 1
        stats["total_original_size"] += original_size
        stats["total_compressed_size"] += compressed_size
        stats["total_time_ms"] += compression_time_ms

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics.

        Returns:
            Dictionary containing compression performance metrics
        """
        total_stats = {
            "total_compressed": 0,
            "total_original_size": 0,
            "total_compressed_size": 0,
            "total_time_ms": 0.0,
        }

        for stats in self.compression_stats.values():
            for key in total_stats:
                total_stats[key] += stats[key]

        # Calculate averages
        algorithm_stats = {}
        for algorithm, stats in self.compression_stats.items():
            if stats["total_compressed"] > 0:
                avg_original = stats["total_original_size"] / stats["total_compressed"]
                avg_compressed = (
                    stats["total_compressed_size"] / stats["total_compressed"]
                )
                avg_time = stats["total_time_ms"] / stats["total_compressed"]
                avg_ratio = avg_compressed / avg_original if avg_original > 0 else 1.0

                algorithm_stats[algorithm.value] = {
                    "operations": stats["total_compressed"],
                    "avg_original_size": round(avg_original, 2),
                    "avg_compressed_size": round(avg_compressed, 2),
                    "avg_compression_ratio": round(avg_ratio, 4),
                    "avg_compression_time_ms": round(avg_time, 4),
                    "total_size_saved": stats["total_original_size"]
                    - stats["total_compressed_size"],
                }

        return {
            "overall": {
                "total_operations": total_stats["total_compressed"],
                "total_original_mb": round(
                    total_stats["total_original_size"] / 1024 / 1024, 2
                ),
                "total_compressed_mb": round(
                    total_stats["total_compressed_size"] / 1024 / 1024, 2
                ),
                "total_size_saved_mb": round(
                    (
                        total_stats["total_original_size"]
                        - total_stats["total_compressed_size"]
                    )
                    / 1024
                    / 1024,
                    2,
                ),
                "overall_compression_ratio": round(
                    total_stats["total_compressed_size"]
                    / total_stats["total_original_size"],
                    4,
                )
                if total_stats["total_original_size"] > 0
                else 1.0,
                "avg_compression_time_ms": round(
                    total_stats["total_time_ms"] / total_stats["total_compressed"], 4
                )
                if total_stats["total_compressed"] > 0
                else 0.0,
            },
            "by_algorithm": algorithm_stats,
            "compression_threshold": self.compression_threshold,
            "default_algorithm": self.default_algorithm.value,
            "adaptive_enabled": self.enable_adaptive,
        }

    def optimize_for_trace_data(
        self, sample_traces: List[Dict[str, Any]]
    ) -> CompressionAlgorithm:
        """Analyze sample traces and recommend optimal compression algorithm.

        Args:
            sample_traces: List of sample trace data for analysis

        Returns:
            Recommended compression algorithm
        """
        if not sample_traces:
            return self.default_algorithm

        # Test each algorithm on sample data
        algorithm_scores = {}

        for algorithm in CompressionAlgorithm:
            if algorithm == CompressionAlgorithm.NONE:
                continue

            total_ratio = 0.0
            total_time = 0.0
            valid_samples = 0

            for trace in sample_traces[:10]:  # Test on first 10 samples
                try:
                    result = self.compress_data(trace, algorithm)
                    if result.algorithm == algorithm:
                        total_ratio += result.compression_ratio
                        total_time += result.compression_time_ms
                        valid_samples += 1
                except Exception:
                    continue

            if valid_samples > 0:
                avg_ratio = total_ratio / valid_samples
                avg_time = total_time / valid_samples
                # Score combines compression ratio (lower is better) and speed (lower is better)
                # Weight ratio more heavily since it's more important for storage
                score = (avg_ratio * 0.7) + (
                    avg_time / 1000 * 0.3
                )  # Normalize time to seconds
                algorithm_scores[algorithm] = score

        if not algorithm_scores:
            return self.default_algorithm

        # Return algorithm with lowest score (best compression + speed)
        best_algorithm = min(algorithm_scores.keys(), key=lambda k: algorithm_scores[k])
        logger.info(
            f"Recommended compression algorithm: {best_algorithm.value} (score: {algorithm_scores[best_algorithm]:.4f})"
        )

        return best_algorithm


# Global compressor instance
_compressor: Optional[TraceCompressor] = None


def get_trace_compressor() -> Optional[TraceCompressor]:
    """Get the global trace compressor instance."""
    return _compressor


def initialize_trace_compressor(**kwargs) -> TraceCompressor:
    """Initialize the global trace compressor.

    Args:
        **kwargs: Configuration parameters for TraceCompressor

    Returns:
        Initialized TraceCompressor instance
    """
    global _compressor
    _compressor = TraceCompressor(**kwargs)
    return _compressor


def shutdown_trace_compressor() -> None:
    """Shutdown the global trace compressor."""
    global _compressor
    _compressor = None
