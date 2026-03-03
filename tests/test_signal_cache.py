"""Unit tests for signal cache module."""

import unittest
import threading
import time
import numpy as np

from signal_viewer.core.signal_cache import SignalCache


class TestSignalCachePutGet(unittest.TestCase):
    """Test basic put/get operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = SignalCache(max_memory_bytes=1_000_000)

    def test_put_get_roundtrip(self):
        """Test putting and getting data."""
        key = "signal_001"
        time_array = np.linspace(0, 10, 1000)
        signal_array = np.sin(time_array)

        self.cache.put(key, time_array, signal_array)
        result = self.cache.get(key)

        self.assertIsNotNone(result)
        time_retrieved, signal_retrieved = result
        np.testing.assert_array_almost_equal(time_retrieved, time_array)
        np.testing.assert_array_almost_equal(signal_retrieved, signal_array)

    def test_get_nonexistent_key(self):
        """Test getting non-existent key returns None."""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)

    def test_put_mismatched_lengths(self):
        """Test put with mismatched array lengths raises ValueError."""
        key = "signal_001"
        time_array = np.linspace(0, 10, 100)
        signal_array = np.sin(np.linspace(0, 10, 50))  # Different length

        with self.assertRaises(ValueError):
            self.cache.put(key, time_array, signal_array)

    def test_get_returns_copy(self):
        """Test that get returns a copy, not the original."""
        key = "signal_001"
        time_array = np.linspace(0, 10, 100)
        signal_array = np.sin(time_array)

        self.cache.put(key, time_array, signal_array)
        time_retrieved, signal_retrieved = self.cache.get(key)

        # Modify the retrieved copy
        time_retrieved[0] = 999
        signal_retrieved[0] = 999

        # Get again and check original is unchanged
        time_retrieved2, signal_retrieved2 = self.cache.get(key)
        self.assertNotEqual(time_retrieved2[0], 999)
        self.assertNotEqual(signal_retrieved2[0], 999)


class TestSignalCacheEviction(unittest.TestCase):
    """Test LRU eviction when memory exceeded."""

    def test_lru_eviction(self):
        """Test that oldest entry is evicted when memory exceeded."""
        # Create cache with very small memory budget to force eviction
        cache = SignalCache(max_memory_bytes=800)

        # Add first entry (40 entries * 8 bytes * 2 = 640 bytes)
        time1 = np.linspace(0, 10, 40)
        signal1 = np.sin(time1)
        cache.put("key1", time1, signal1)

        # Add second entry (more data to exceed budget)
        time2 = np.linspace(0, 10, 40)
        signal2 = np.cos(time2)
        cache.put("key2", time2, signal2)

        # Add third entry (should force eviction of key1)
        time3 = np.linspace(0, 10, 40)
        signal3 = np.ones(40)
        cache.put("key3", time3, signal3)

        # key1 should be evicted due to LRU
        self.assertIsNone(cache.get("key1"))

    def test_get_updates_lru_order(self):
        """Test that get updates LRU order."""
        cache = SignalCache(max_memory_bytes=20_000)

        # Add entries
        time1 = np.linspace(0, 10, 50)
        signal1 = np.sin(time1)
        cache.put("key1", time1, signal1)

        time2 = np.linspace(0, 10, 50)
        signal2 = np.cos(time2)
        cache.put("key2", time2, signal2)

        # Access key1 (makes it most recent)
        _ = cache.get("key1")

        # Add new entry (should evict key2, not key1)
        time3 = np.linspace(0, 10, 50)
        signal3 = np.tan(np.linspace(0, 1, 50))
        cache.put("key3", time3, signal3)

        # key1 should still exist (was accessed)
        self.assertIsNotNone(cache.get("key1"))


class TestSignalCacheInvalidate(unittest.TestCase):
    """Test invalidate functionality."""

    def test_invalidate_removes_entry(self):
        """Test that invalidate removes an entry."""
        cache = SignalCache()
        key = "signal_001"
        time_array = np.linspace(0, 10, 100)
        signal_array = np.sin(time_array)

        cache.put(key, time_array, signal_array)
        self.assertIsNotNone(cache.get(key))

        cache.invalidate(key)
        self.assertIsNone(cache.get(key))

    def test_invalidate_nonexistent_key(self):
        """Test invalidate on non-existent key is safe."""
        cache = SignalCache()
        cache.invalidate("nonexistent_key")  # Should not raise


class TestSignalCacheClear(unittest.TestCase):
    """Test clear functionality."""

    def test_clear_removes_all(self):
        """Test that clear removes all entries."""
        cache = SignalCache()

        # Add multiple entries
        for i in range(5):
            time = np.linspace(0, 10, 100)
            signal = np.sin(time + i)
            cache.put(f"key{i}", time, signal)

        # Clear cache
        cache.clear()

        # All should be gone
        for i in range(5):
            self.assertIsNone(cache.get(f"key{i}"))


class TestSignalCacheStats(unittest.TestCase):
    """Test cache statistics."""

    def test_stats_format(self):
        """Test stats returns correct structure."""
        cache = SignalCache(max_memory_bytes=1_000_000)
        stats = cache.stats()

        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)
        self.assertIn("memory_used", stats)
        self.assertIn("memory_budget", stats)
        self.assertIn("entries", stats)
        self.assertIn("memory_percent", stats)

    def test_stats_initial(self):
        """Test initial stats values."""
        cache = SignalCache()
        stats = cache.stats()

        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["hit_rate"], 0.0)
        self.assertEqual(stats["memory_used"], 0)
        self.assertEqual(stats["entries"], 0)

    def test_stats_hit_rate(self):
        """Test hit rate calculation."""
        cache = SignalCache()

        time = np.linspace(0, 10, 100)
        signal = np.sin(time)
        cache.put("key1", time, signal)

        # One hit
        cache.get("key1")
        # One miss
        cache.get("nonexistent")

        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_stats_memory_tracking(self):
        """Test memory usage tracking."""
        cache = SignalCache()

        time = np.linspace(0, 10, 1000).astype(np.float64)
        signal = np.sin(time).astype(np.float64)
        cache.put("key1", time, signal)

        stats = cache.stats()
        expected_memory = time.nbytes + signal.nbytes
        self.assertEqual(stats["memory_used"], expected_memory)
        self.assertEqual(stats["entries"], 1)

    def test_stats_memory_percent(self):
        """Test memory percentage calculation."""
        cache = SignalCache(max_memory_bytes=1_000_000)

        time = np.linspace(0, 10, 100)
        signal = np.sin(time)
        cache.put("key1", time, signal)

        stats = cache.stats()
        memory_pct = stats["memory_percent"]
        self.assertGreater(memory_pct, 0)
        self.assertLess(memory_pct, 100)


class TestSignalCacheThreadSafety(unittest.TestCase):
    """Test thread safety."""

    def test_concurrent_put_get(self):
        """Test concurrent put and get operations."""
        cache = SignalCache(max_memory_bytes=10_000_000)
        results = []

        def worker(thread_id):
            for i in range(10):
                key = f"key_{thread_id}_{i}"
                time = np.linspace(0, 10, 100)
                signal = np.sin(time + thread_id + i)
                cache.put(key, time, signal)

                # Try to get it back
                result = cache.get(key)
                results.append(result is not None)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All operations should succeed
        self.assertTrue(all(results))

    def test_concurrent_invalidate(self):
        """Test concurrent invalidate operations."""
        cache = SignalCache()

        # Pre-populate
        for i in range(20):
            time = np.linspace(0, 10, 100)
            signal = np.sin(time + i)
            cache.put(f"key_{i}", time, signal)

        def worker(thread_id):
            for i in range(5):
                cache.invalidate(f"key_{thread_id * 5 + i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = cache.stats()
        self.assertEqual(stats["entries"], 0)


class TestSignalCacheRepr(unittest.TestCase):
    """Test string representation."""

    def test_repr(self):
        """Test __repr__ method."""
        cache = SignalCache()
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)
        cache.put("key1", time, signal)

        repr_str = repr(cache)
        self.assertIn("SignalCache", repr_str)
        self.assertIn("entries", repr_str)


if __name__ == "__main__":
    unittest.main()
