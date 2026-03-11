"""Unit tests for the LRU signal cache."""

import threading
import unittest
import numpy as np

from signal_viewer.core.signal_cache import SignalCache


# ---------------------------------------------------------------------------
# Basic put / get
# ---------------------------------------------------------------------------

class TestSignalCachePutGet(unittest.TestCase):
    """Round-trip and basic semantics."""

    def setUp(self):
        self.cache = SignalCache(max_memory_bytes=1_000_000)

    def test_roundtrip(self):
        t = np.linspace(0, 10, 1000)
        s = np.sin(t)
        self.cache.put("k1", t, s)
        t2, s2 = self.cache.get("k1")
        np.testing.assert_array_almost_equal(t2, t)
        np.testing.assert_array_almost_equal(s2, s)

    def test_missing_key_returns_none(self):
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            self.cache.put("k", np.zeros(100), np.zeros(50))

    def test_get_returns_copy(self):
        t = np.linspace(0, 10, 100)
        s = np.sin(t)
        self.cache.put("k", t, s)

        t2, s2 = self.cache.get("k")
        t2[0] = 999
        s2[0] = 999

        t3, s3 = self.cache.get("k")
        self.assertNotEqual(t3[0], 999)
        self.assertNotEqual(s3[0], 999)


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestSignalCacheEviction(unittest.TestCase):
    """Memory-budget enforcement and LRU ordering."""

    def test_lru_eviction(self):
        cache = SignalCache(max_memory_bytes=800)
        for i, name in enumerate(("k1", "k2", "k3")):
            t = np.linspace(0, 10, 40)
            cache.put(name, t, np.sin(t + i))
        # k1 should have been evicted
        self.assertIsNone(cache.get("k1"))

    def test_get_refreshes_lru_order(self):
        cache = SignalCache(max_memory_bytes=20_000)
        for i, name in enumerate(("k1", "k2")):
            t = np.linspace(0, 10, 50)
            cache.put(name, t, np.sin(t + i))

        _ = cache.get("k1")  # refresh k1

        t3 = np.linspace(0, 10, 50)
        cache.put("k3", t3, np.tan(np.linspace(0, 1, 50)))

        # k1 should survive (accessed more recently than k2)
        self.assertIsNotNone(cache.get("k1"))


# ---------------------------------------------------------------------------
# Invalidate / clear
# ---------------------------------------------------------------------------

class TestSignalCacheInvalidate(unittest.TestCase):

    def test_invalidate_removes_entry(self):
        cache = SignalCache()
        t = np.linspace(0, 10, 100)
        cache.put("k", t, np.sin(t))
        self.assertIsNotNone(cache.get("k"))
        cache.invalidate("k")
        self.assertIsNone(cache.get("k"))

    def test_invalidate_nonexistent_key_is_noop(self):
        SignalCache().invalidate("nope")  # should not raise


class TestSignalCacheClear(unittest.TestCase):

    def test_clear_removes_all(self):
        cache = SignalCache()
        for i in range(5):
            t = np.linspace(0, 10, 100)
            cache.put(f"k{i}", t, np.sin(t + i))
        cache.clear()
        for i in range(5):
            self.assertIsNone(cache.get(f"k{i}"))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestSignalCacheStats(unittest.TestCase):
    """Counters, memory tracking, and hit-rate calculation."""

    def test_stat_keys(self):
        stats = SignalCache(max_memory_bytes=1_000_000).stats()
        for key in ("hits", "misses", "hit_rate", "memory_used",
                     "memory_budget", "entries", "memory_percent"):
            self.assertIn(key, stats)

    def test_initial_values(self):
        stats = SignalCache().stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["hit_rate"], 0.0)
        self.assertEqual(stats["memory_used"], 0)
        self.assertEqual(stats["entries"], 0)

    def test_hit_rate(self):
        cache = SignalCache()
        t = np.linspace(0, 10, 100)
        cache.put("k", t, np.sin(t))
        cache.get("k")          # hit
        cache.get("missing")    # miss
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_memory_tracking(self):
        cache = SignalCache()
        t = np.linspace(0, 10, 1000).astype(np.float64)
        s = np.sin(t).astype(np.float64)
        cache.put("k", t, s)
        stats = cache.stats()
        self.assertEqual(stats["memory_used"], t.nbytes + s.nbytes)
        self.assertEqual(stats["entries"], 1)

    def test_memory_percent(self):
        cache = SignalCache(max_memory_bytes=1_000_000)
        t = np.linspace(0, 10, 100)
        cache.put("k", t, np.sin(t))
        pct = cache.stats()["memory_percent"]
        self.assertGreater(pct, 0)
        self.assertLess(pct, 100)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestSignalCacheThreadSafety(unittest.TestCase):

    def test_concurrent_put_get(self):
        cache = SignalCache(max_memory_bytes=10_000_000)
        results = []

        def worker(tid):
            for i in range(10):
                t = np.linspace(0, 10, 100)
                cache.put(f"k_{tid}_{i}", t, np.sin(t + tid + i))
                results.append(cache.get(f"k_{tid}_{i}") is not None)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertTrue(all(results))

    def test_concurrent_invalidate(self):
        cache = SignalCache()
        for i in range(20):
            t = np.linspace(0, 10, 100)
            cache.put(f"k_{i}", t, np.sin(t + i))

        def worker(tid):
            for i in range(5):
                cache.invalidate(f"k_{tid * 5 + i}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(cache.stats()["entries"], 0)


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------

class TestSignalCacheRepr(unittest.TestCase):

    def test_repr(self):
        cache = SignalCache()
        t = np.linspace(0, 10, 100)
        cache.put("k", t, np.sin(t))
        r = repr(cache)
        self.assertIn("SignalCache", r)
        self.assertIn("entries", r)


if __name__ == "__main__":
    unittest.main()
