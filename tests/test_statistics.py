"""Unit tests for statistics module."""

import unittest
import numpy as np

from signal_viewer.processing.statistics import (
    compute_descriptive_stats,
    compute_rolling_stats,
    compute_histogram,
    compute_percentiles,
    compute_rainflow,
    _extract_turning_points,
    _rainflow_4point,
)


class TestComputeDescriptiveStats(unittest.TestCase):
    """Test descriptive statistics computation."""

    def test_descriptive_stats_known_values(self):
        """Test descriptive stats with known values."""
        signal = np.array([1, 2, 3, 4, 5], dtype=float)

        stats = compute_descriptive_stats(signal)

        self.assertEqual(stats["count"], 5)
        self.assertAlmostEqual(stats["mean"], 3.0)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 5.0)
        self.assertAlmostEqual(stats["median"], 3.0)

    def test_descriptive_stats_contains_all_keys(self):
        """Test that stats contains all required keys."""
        signal = np.random.randn(100)

        stats = compute_descriptive_stats(signal)

        required_keys = [
            "count",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "rms",
            "peak_to_peak",
            "skewness",
            "kurtosis",
            "iqr",
            "q25",
            "q75",
        ]

        for key in required_keys:
            self.assertIn(key, stats)

    def test_descriptive_stats_empty_signal(self):
        """Test with empty signal."""
        stats = compute_descriptive_stats(np.array([]))

        self.assertEqual(stats["count"], 0)
        self.assertTrue(np.isnan(stats["mean"]))
        self.assertTrue(np.isnan(stats["std"]))

    def test_descriptive_stats_with_nans(self):
        """Test that NaNs are handled."""
        signal = np.array([1, 2, np.nan, 4, 5])

        stats = compute_descriptive_stats(signal)

        self.assertEqual(stats["count"], 4)  # NaN filtered out
        self.assertAlmostEqual(stats["mean"], 3.0)

    def test_descriptive_stats_rms(self):
        """Test RMS calculation."""
        signal = np.array([3, 4])  # RMS = sqrt((9 + 16) / 2) = 5

        stats = compute_descriptive_stats(signal)

        # RMS = sqrt(mean(x^2)) = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        self.assertAlmostEqual(stats["rms"], np.sqrt(12.5), places=1)

    def test_descriptive_stats_iqr(self):
        """Test IQR calculation."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        stats = compute_descriptive_stats(signal)

        # Q1 = 3.25, Q3 = 7.75, IQR = 4.5
        self.assertGreater(stats["iqr"], 0)

    def test_descriptive_stats_symmetry(self):
        """Test skewness of symmetric distribution."""
        signal = np.array([1, 2, 3, 4, 5])

        stats = compute_descriptive_stats(signal)

        # Should be approximately symmetric (skewness near 0)
        self.assertLess(np.abs(stats["skewness"]), 0.5)


class TestComputeRollingStats(unittest.TestCase):
    """Test rolling statistics computation."""

    def test_rolling_stats_output_shape(self):
        """Test rolling stats output shapes."""
        signal = np.sin(np.linspace(0, 10, 1000))

        rolling = compute_rolling_stats(signal, window_size=51)

        self.assertEqual(len(rolling["rolling_mean"]), len(signal))
        self.assertEqual(len(rolling["rolling_std"]), len(signal))
        self.assertEqual(len(rolling["rolling_min"]), len(signal))
        self.assertEqual(len(rolling["rolling_max"]), len(signal))

    def test_rolling_stats_empty_signal(self):
        """Test rolling stats with empty signal."""
        rolling = compute_rolling_stats(np.array([]), window_size=10)

        self.assertEqual(len(rolling["rolling_mean"]), 0)
        self.assertEqual(len(rolling["rolling_std"]), 0)
        self.assertEqual(len(rolling["rolling_min"]), 0)
        self.assertEqual(len(rolling["rolling_max"]), 0)

    def test_rolling_stats_values(self):
        """Test rolling stats computed correctly."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

        rolling = compute_rolling_stats(signal, window_size=3)

        # First value: centered window [1, 2, 3]
        # Mean should be 2
        self.assertAlmostEqual(rolling["rolling_mean"][1], 2.0)

    def test_rolling_stats_min_max(self):
        """Test rolling min/max values."""
        signal = np.array([5, 1, 5, 1, 5, 1, 5], dtype=float)

        rolling = compute_rolling_stats(signal, window_size=3)

        # Should capture oscillations
        self.assertTrue(np.any(rolling["rolling_min"] < 3))
        self.assertTrue(np.any(rolling["rolling_max"] > 3))

    def test_rolling_stats_with_nans(self):
        """Test rolling stats handles NaNs."""
        signal = np.array([1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10])

        rolling = compute_rolling_stats(signal, window_size=5)

        self.assertEqual(len(rolling["rolling_mean"]), len(signal))


class TestComputeHistogram(unittest.TestCase):
    """Test histogram computation."""

    def test_histogram_output_shape(self):
        """Test histogram output shapes."""
        signal = np.random.randn(1000)

        bin_edges, counts = compute_histogram(signal, n_bins=50)

        self.assertEqual(len(bin_edges), 51)  # n_bins + 1
        self.assertEqual(len(counts), 50)

    def test_histogram_counts_sum(self):
        """Test that histogram counts sum to signal length."""
        signal = np.random.randn(1000)

        bin_edges, counts = compute_histogram(signal, n_bins=50)

        self.assertEqual(np.sum(counts), len(signal))

    def test_histogram_empty_signal(self):
        """Test histogram with empty signal."""
        bin_edges, counts = compute_histogram(np.array([]))

        self.assertEqual(len(bin_edges), 0)
        self.assertEqual(len(counts), 0)

    def test_histogram_with_nans(self):
        """Test histogram handles NaNs."""
        signal = np.array([1, 2, 3, 4, 5, np.nan, 6, 7, 8, 9, 10])

        bin_edges, counts = compute_histogram(signal, n_bins=10)

        # NaN should be filtered out
        self.assertEqual(np.sum(counts), 10)

    def test_histogram_range(self):
        """Test that histogram covers full range of values."""
        signal = np.array([0, 1, 2, 3, 4, 5])

        bin_edges, counts = compute_histogram(signal, n_bins=5)

        # Edges should span from min to max
        self.assertLessEqual(bin_edges[0], np.min(signal))
        self.assertGreaterEqual(bin_edges[-1], np.max(signal))


class TestComputePercentiles(unittest.TestCase):
    """Test percentile computation."""

    def test_percentiles_default(self):
        """Test percentiles with default values."""
        signal = np.arange(100)

        percentiles = compute_percentiles(signal)

        self.assertIn("p1", percentiles)
        self.assertIn("p50", percentiles)
        self.assertIn("p99", percentiles)

    def test_percentiles_custom(self):
        """Test percentiles with custom values."""
        signal = np.arange(100)

        percentiles = compute_percentiles(signal, percentiles=[25, 50, 75])

        self.assertIn("p25", percentiles)
        self.assertIn("p50", percentiles)
        self.assertIn("p75", percentiles)
        self.assertEqual(len(percentiles), 3)

    def test_percentiles_values(self):
        """Test percentile values are correct."""
        signal = np.arange(100)

        percentiles = compute_percentiles(signal, percentiles=[0, 50, 100])

        # p0 should be near 0
        self.assertLess(percentiles["p0"], 5)
        # p50 should be near 50
        self.assertAlmostEqual(percentiles["p50"], 50, delta=5)
        # p100 should be near 100
        self.assertGreater(percentiles["p100"], 95)

    def test_percentiles_empty_signal(self):
        """Test percentiles with empty signal."""
        percentiles = compute_percentiles(np.array([]))

        self.assertTrue(np.isnan(percentiles["p50"]))

    def test_percentiles_with_nans(self):
        """Test percentiles handle NaNs."""
        signal = np.array([1, 2, 3, 4, 5, np.nan, 6, 7, 8, 9, 10])

        percentiles = compute_percentiles(signal, percentiles=[50])

        self.assertFalse(np.isnan(percentiles["p50"]))


class TestExtractTurningPoints(unittest.TestCase):
    """Test turning point extraction."""

    def test_simple_wave(self):
        """Test turning points of a simple oscillation."""
        signal = np.array([0, 1, 0, -1, 0, 1, 0])
        tp = _extract_turning_points(signal)
        # Should include peaks (1) and valleys (-1) plus endpoints
        self.assertIn(1.0, tp)
        self.assertIn(-1.0, tp)

    def test_monotonic(self):
        """Monotonic signal has only endpoints as turning points."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tp = _extract_turning_points(signal)
        self.assertEqual(len(tp), 2)
        self.assertEqual(tp[0], 1.0)
        self.assertEqual(tp[-1], 5.0)

    def test_short_signal(self):
        """Signal shorter than 3 is returned as-is."""
        signal = np.array([5.0, 3.0])
        tp = _extract_turning_points(signal)
        np.testing.assert_array_equal(tp, signal)

    def test_plateau(self):
        """Plateaus should be absorbed (not produce extra turning points)."""
        signal = np.array([0, 1, 1, 1, 0, -1, -1, 0])
        tp = _extract_turning_points(signal)
        # After absorbing plateaus, peaks at 1 and valleys at -1
        self.assertTrue(len(tp) <= len(signal))


class TestRainflow4Point(unittest.TestCase):
    """Test the 4-point rainflow algorithm."""

    def test_returns_numpy_arrays(self):
        """Verify output types are numpy arrays."""
        tp = np.array([0, 2, -1, 3, -2, 1])
        result = _rainflow_4point(tp)
        self.assertEqual(len(result), 6)
        for arr in result:
            self.assertIsInstance(arr, np.ndarray)

    def test_simple_cycle(self):
        """A simple full cycle should be detected."""
        # 0, 2, 0, 4 → inner range |2-0|=2, outer range |4-0|=4 → full cycle
        tp = np.array([0.0, 2.0, 0.0, 4.0])
        full_r, half_r, full_mx, full_mn, half_mx, half_mn = _rainflow_4point(tp)
        self.assertEqual(len(full_r), 1)
        self.assertAlmostEqual(full_r[0], 2.0)
        self.assertAlmostEqual(full_mx[0], 2.0)
        self.assertAlmostEqual(full_mn[0], 0.0)

    def test_no_full_cycles(self):
        """Monotonically increasing turning points → no full cycles."""
        tp = np.array([0.0, 1.0, 3.0])
        full_r, half_r, _, _, _, _ = _rainflow_4point(tp)
        self.assertEqual(len(full_r), 0)
        self.assertEqual(len(half_r), 2)


class TestComputeRainflow(unittest.TestCase):
    """Test the full rainflow computation."""

    def test_empty_signal(self):
        """Empty signal returns empty result."""
        result = compute_rainflow(np.array([]))
        self.assertEqual(result['total_cycles'], 0)
        self.assertEqual(result['total_half_cycles'], 0)
        self.assertEqual(result['ranges'], [])

    def test_sine_wave(self):
        """Sine wave should produce recognizable cycles."""
        t = np.linspace(0, 10 * np.pi, 5000)
        signal = np.sin(t)
        result = compute_rainflow(signal, n_bins=10)
        # Should detect cycles
        self.assertGreater(result['total_cycles'] + result['total_half_cycles'], 0)
        self.assertEqual(len(result['counts']), 10)
        self.assertEqual(len(result['bin_edges']), 11)
        # All ranges should be lists (JSON-serializable)
        self.assertIsInstance(result['ranges'], list)
        self.assertIsInstance(result['cycle_maxs'], list)
        self.assertIsInstance(result['cycle_mins'], list)

    def test_cycle_max_min_signs(self):
        """For signal oscillating around zero, maxs should be positive, mins negative."""
        t = np.linspace(0, 6 * np.pi, 3000)
        signal = 2 * np.sin(t)
        result = compute_rainflow(signal, n_bins=5)
        if result['cycle_maxs']:
            self.assertTrue(any(v > 0 for v in result['cycle_maxs']))
        if result['cycle_mins']:
            self.assertTrue(any(v < 0 for v in result['cycle_mins']))

    def test_json_serializable(self):
        """All result values must be JSON-serializable (no numpy types)."""
        import json
        signal = np.random.randn(1000)
        result = compute_rainflow(signal, n_bins=10)
        # This will raise if any numpy types remain
        json.dumps(result)

    def test_consistency_lengths(self):
        """ranges, cycle_maxs, cycle_mins must have the same length."""
        signal = np.random.randn(5000)
        result = compute_rainflow(signal, n_bins=10)
        n = len(result['ranges'])
        self.assertEqual(len(result['cycle_maxs']), n)
        self.assertEqual(len(result['cycle_mins']), n)
        self.assertEqual(result['total_cycles'] + result['total_half_cycles'], n)

    def test_performance_large_signal(self):
        """Rainflow on a large signal completes within reasonable time."""
        import time
        signal = np.random.randn(500_000)
        start = time.time()
        result = compute_rainflow(signal, n_bins=20)
        elapsed = time.time() - start
        # Should complete in under 5 seconds even on slow hardware
        self.assertLess(elapsed, 5.0, f"Rainflow took {elapsed:.2f}s on 500k samples")
        self.assertGreater(result['total_cycles'] + result['total_half_cycles'], 0)


if __name__ == "__main__":
    unittest.main()
