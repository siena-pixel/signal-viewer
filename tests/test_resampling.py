"""Unit tests for the resampling module (LTTB and simple decimation)."""

import unittest
import numpy as np

from signal_viewer.processing.resampling import lttb_downsample, simple_decimate


# ---------------------------------------------------------------------------
# LTTB downsampling
# ---------------------------------------------------------------------------

class TestLTTBDownsample(unittest.TestCase):
    """Tests for the Largest-Triangle-Three-Buckets algorithm."""

    def test_basic_output_length(self):
        t = np.linspace(0, 10, 10_000)
        t_ds, s_ds = lttb_downsample(t, np.sin(t), target_points=100)
        self.assertEqual(len(t_ds), 100)
        self.assertEqual(len(s_ds), 100)

    def test_preserves_first_and_last(self):
        t = np.linspace(0, 10, 1000)
        s = np.sin(t)
        t_ds, s_ds = lttb_downsample(t, s, target_points=50)
        self.assertEqual(t_ds[0], t[0])
        self.assertEqual(t_ds[-1], t[-1])
        self.assertEqual(s_ds[0], s[0])
        self.assertEqual(s_ds[-1], s[-1])

    def test_shorter_than_target_returns_unchanged(self):
        t = np.linspace(0, 10, 50)
        s = np.sin(t)
        t_ds, s_ds = lttb_downsample(t, s, target_points=100)
        np.testing.assert_array_equal(t_ds, t)
        np.testing.assert_array_equal(s_ds, s)

    def test_various_target_lengths(self):
        t = np.linspace(0, 10, 5000)
        s = np.cos(t) + 0.1 * np.sin(10 * t)
        for target in (50, 100, 200):
            t_ds, s_ds = lttb_downsample(t, s, target_points=target)
            self.assertEqual(len(t_ds), target)
            self.assertEqual(len(s_ds), target)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            lttb_downsample(
                np.linspace(0, 10, 100),
                np.sin(np.linspace(0, 10, 50)),
                target_points=50,
            )

    def test_visual_preservation_peaks(self):
        """Downsampled signal should retain a significant fraction of peaks."""
        t = np.linspace(0, 10, 5000)
        s = np.sin(2 * np.pi * t) + 0.5 * np.sin(8 * np.pi * t)
        _, s_ds = lttb_downsample(t, s, target_points=500)

        count_peaks = lambda arr: sum(
            1 for i in range(1, len(arr) - 1)
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]
        )
        self.assertGreater(count_peaks(s_ds), count_peaks(s) // 5)

    def test_returns_copies(self):
        t = np.linspace(0, 10, 1000)
        s = np.sin(t)
        t_ds, s_ds = lttb_downsample(t, s, target_points=100)
        t_ds[0] = 999
        s_ds[0] = 999
        self.assertNotEqual(t[0], 999)
        self.assertNotEqual(s[0], 999)


# ---------------------------------------------------------------------------
# Simple decimation
# ---------------------------------------------------------------------------

class TestSimpleDecimate(unittest.TestCase):
    """Tests for uniform-stride decimation."""

    def test_basic_length(self):
        t = np.linspace(0, 10, 1000)
        t_d, s_d = simple_decimate(t, np.sin(t), factor=5)
        self.assertLessEqual(len(t_d), len(t) // 5 + 1)
        self.assertLessEqual(len(s_d), len(t) // 5 + 1)

    def test_includes_last_point(self):
        t = np.linspace(0, 10, 100)
        s = np.sin(t)
        t_d, s_d = simple_decimate(t, s, factor=10)
        self.assertEqual(t_d[-1], t[-1])
        self.assertEqual(s_d[-1], s[-1])

    def test_factor_one_returns_copy(self):
        t = np.linspace(0, 10, 100)
        s = np.sin(t)
        t_d, s_d = simple_decimate(t, s, factor=1)
        np.testing.assert_array_equal(t_d, t)
        np.testing.assert_array_equal(s_d, s)

    def test_invalid_factor_raises(self):
        t = np.linspace(0, 10, 100)
        s = np.sin(t)
        with self.assertRaises(ValueError):
            simple_decimate(t, s, factor=0)
        with self.assertRaises(ValueError):
            simple_decimate(t, s, factor=-1)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            simple_decimate(
                np.linspace(0, 10, 100),
                np.sin(np.linspace(0, 10, 50)),
                factor=5,
            )

    def test_selects_every_nth(self):
        t = np.arange(100, dtype=float)
        s = t * 2
        t_d, s_d = simple_decimate(t, s, factor=10)
        expected_idx = list(range(0, 100, 10))
        if 99 not in expected_idx:
            expected_idx.append(99)
        np.testing.assert_array_equal(t_d[:-1], t[expected_idx[:-1]])
        np.testing.assert_array_equal(s_d[:-1], s[expected_idx[:-1]])

    def test_returns_copies(self):
        t = np.linspace(0, 10, 100)
        s = np.sin(t)
        t_d, s_d = simple_decimate(t, s, factor=5)
        t_d[0] = 999
        s_d[0] = 999
        self.assertNotEqual(t[0], 999)
        self.assertNotEqual(s[0], 999)


if __name__ == "__main__":
    unittest.main()
