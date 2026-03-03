"""Unit tests for statistics module."""

import unittest
import numpy as np

from signal_viewer.processing.statistics import (
    compute_descriptive_stats,
    compute_rolling_stats,
    compute_histogram,
    compute_percentiles,
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


if __name__ == "__main__":
    unittest.main()
