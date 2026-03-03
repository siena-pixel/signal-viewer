"""
Thread-Safe LRU Cache for Signal Data

Implements a memory-aware least-recently-used cache for time series signal data.
Tracks memory usage and evicts oldest entries when memory budget is exceeded.

Cache keys: "{file_path}::{batch}::{signal_idx}"
"""

import threading
from collections import OrderedDict
from typing import Dict, Optional, Tuple
import numpy as np


class SignalCache:
    """
    Thread-safe LRU cache for signal time series data.

    Features:
      - Memory-aware eviction based on configurable budget
      - LRU eviction policy when memory limit exceeded
      - Cache statistics (hits, misses, hit rate)
      - Thread-safe access via locks
      - Efficient memory tracking
    """

    def __init__(self, max_memory_bytes: int = 500_000_000):
        """
        Initialize signal cache.

        Args:
            max_memory_bytes: Maximum memory budget in bytes (default 500 MB)
        """
        self.max_memory_bytes = max_memory_bytes
        self._cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._memory_usage: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def put(self, key: str, time_array: np.ndarray, signal_array: np.ndarray) -> None:
        """
        Store signal data in cache.

        Evicts oldest entries if total memory exceeds budget.

        Args:
            key: Cache key (typically "{file_path}::{batch}::{signal_idx}")
            time_array: Time values array [num_samples]
            signal_array: Signal values array [num_samples]

        Raises:
            ValueError: If arrays have inconsistent lengths
        """
        if len(time_array) != len(signal_array):
            raise ValueError(
                f"Array length mismatch: time={len(time_array)}, signal={len(signal_array)}"
            )

        with self._lock:
            # Calculate memory usage
            time_memory = time_array.nbytes
            signal_memory = signal_array.nbytes
            total_memory = time_memory + signal_memory

            # Remove old entry if exists
            if key in self._cache:
                del self._memory_usage[key]

            # Add new entry
            self._cache[key] = (time_array.copy(), signal_array.copy())
            self._memory_usage[key] = total_memory

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Evict if necessary
            self._evict_if_needed()

    def get(self, key: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve signal data from cache.

        Updates LRU order on hit.

        Args:
            key: Cache key

        Returns:
            Tuple of (time_array, signal_array) if found, None otherwise
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1

            time_array, signal_array = self._cache[key]
            return (time_array.copy(), signal_array.copy())

    def invalidate(self, key: str) -> None:
        """
        Remove specific cache entry.

        Args:
            key: Cache key
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._memory_usage[key]

    def clear(self) -> None:
        """Remove all cache entries."""
        with self._lock:
            self._cache.clear()
            self._memory_usage.clear()

    def stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with:
              - hits: int (total cache hits)
              - misses: int (total cache misses)
              - hit_rate: float (0.0 to 1.0)
              - memory_used: int (bytes)
              - memory_budget: int (bytes)
              - entries: int (number of cached signals)
              - memory_percent: float (0-100)
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0
            memory_used = sum(self._memory_usage.values())
            memory_percent = (memory_used / self.max_memory_bytes * 100) if self.max_memory_bytes > 0 else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "memory_used": memory_used,
                "memory_budget": self.max_memory_bytes,
                "entries": len(self._cache),
                "memory_percent": memory_percent,
            }

    def _evict_if_needed(self) -> None:
        """
        Evict oldest entries if memory budget exceeded.

        Called within lock context.
        """
        total_memory = sum(self._memory_usage.values())

        while total_memory > self.max_memory_bytes and len(self._cache) > 0:
            # Pop oldest (first) entry
            oldest_key, _ = self._cache.popitem(last=False)
            evicted_memory = self._memory_usage.pop(oldest_key)
            total_memory -= evicted_memory

    def __repr__(self) -> str:
        """String representation."""
        stats = self.stats()
        return (
            f"SignalCache(entries={stats['entries']}, "
            f"memory={stats['memory_used']/1e6:.1f}MB, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
