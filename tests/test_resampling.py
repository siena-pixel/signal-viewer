"""Unit tests for resampling module."""

import unittest
import numpy as np

from signal_viewer.processing.resampling import lttb_downsample, simple_decimate


class TestLTTBDownsample(unittest.TestCase):
    """Test LTTB downsampling algorithm."""

    def test_lttb_basic(self):
        """Test basic LTTB downsampling."""
        time = np.linspace(0, 10, 10000)
        signal = np.sin(time)

        time_ds, signal_ds = lttb_downsample(time, signal, target_points=100)

        self.assertEqual(len(time_ds), 100)
        self.assertEqual(len(signal_ds), 100)

    def test_lttb_preserves_first_last(self):
        """Test that LTTB preserves first and last points."""
        time = np.linspace(0, 10, 1000)
        signal = np.sin(time)

        time_ds, signal_ds = lttb_downsample(time, signal, target_points=50)

        # First and last should be preserved
        self.assertEqual(time_ds[0], time[0])
        self.assertEqual(time_ds[-1], time[-1])
        self.assertEqual(signal_ds[0], signal[0])
        self.assertEqual(signal_ds[-1], signal[-1])

    def test_lttb_shorter_than_target(self):
        """Test LTTB with data shorter than target."""
        time = np.linspace(0, 10, 50)
        signal = np.sin(time)

        time_ds, signal_ds = lttb_downsample(time, signal, target_points=100)

        # Should return unchanged
        np.testing.assert_array_equal(time_ds, time)
        np.testing.assert_array_equal(signal_ds, signal)

    def test_lttb_output_length(self):
        """Test output length matches target."""
        time = np.linspace(0, 10, 5000)
        signal = np.cos(time) + 0.1 * np.sin(10 * time)

        for target in [50, 100, 200]:
            time_ds, signal_ds = lttb_downsample(time, signal, target_points=target)
            self.assertEqual(len(time_ds), target)
            self.assertEqual(len(signal_ds), target)

    def test_lttb_mismatched_lengths(self):
        """Test ValueError for mismatched array lengths."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(np.linspace(0, 10, 50))

        with self.assertRaises(ValueError):
            lttb_downsample(time, signal, target_points=50)

    def test_lttb_visual_preservation(self):
        """Test that LTTB preserves visual shape (peak detection)."""
        # Create a signal with clear peaks
        time = np.linspace(0, 10, 5000)
        signal = np.sin(2 * np.pi * time) + 0.5 * np.sin(8 * np.pi * time)

        time_ds, signal_ds = lttb_downsample(time, signal, target_points=500)

        # Find peaks in both
        def find_peaks(s):
            peaks = []
            for i in range(1, len(s) - 1):
                if s[i] > s[i - 1] and s[i] > s[i + 1]:
                    peaks.append(i)
            return len(peaks)

        peaks_original = find_peaks(signal)
        peaks_downsampled = find_peaks(signal_ds)

        # Downsampled should still have significant peaks
        self.assertGreater(peaks_downsampled, peaks_original // 5)

    def test_lttb_copies_arrays(self):
        """Test that LTTB returns copies, not references."""
        time = np.linspace(0, 10, 1000)
        signal = np.sin(time)

        time_ds, signal_ds = lttb_downsample(time, signal, target_points=100)

        # Modify downsampled
        time_ds[0] = 999
        signal_ds[0] = 999

        # Original should be unchanged
        self.assertNotEqual(time[0], 999)
        self.assertNotEqual(signal[0], 999)


class TestSimpleDecimate(unittest.TestCase):
    """Test simple decimation algorithm."""

    def test_decimate_basic(self):
        """Test basic decimation."""
        time = np.linspace(0, 10, 1000)
        signal = np.sin(time)

        time_d, signal_d = simple_decimate(time, signal, factor=5)

        # Should have roughly 1/5 of samples (plus potentially last point)
        self.assertTrue(len(time_d) <= len(time) // 5 + 1)
        self.assertTrue(len(signal_d) <= len(signal) // 5 + 1)

    def test_decimate_includes_last_point(self):
        """Test that decimation includes the last point."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)

        time_d, signal_d = simple_decimate(time, signal, factor=10)

        # Last point should always be included
        self.assertEqual(time_d[-1], time[-1])
        self.assertEqual(signal_d[-1], signal[-1])

    def test_decimate_factor_one(self):
        """Test decimation with factor 1 returns copy."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)

        time_d, signal_d = simple_decimate(time, signal, factor=1)

        np.testing.assert_array_equal(time_d, time)
        np.testing.assert_array_equal(signal_d, signal)

    def test_decimate_invalid_factor(self):
        """Test ValueError for invalid decimation factor."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)

        with self.assertRaises(ValueError):
            simple_decimate(time, signal, factor=0)

        with self.assertRaises(ValueError):
            simple_decimate(time, signal, factor=-1)

    def test_decimate_mismatched_lengths(self):
        """Test ValueError for mismatched array lengths."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(np.linspace(0, 10, 50))

        with self.assertRaises(ValueError):
            simple_decimate(time, signal, factor=5)

    def test_decimate_selects_every_nth(self):
        """Test that decimation selects every nth point."""
        time = np.arange(100, dtype=float)
        signal = np.arange(100, dtype=float) * 2

        time_d, signal_d = simple_decimate(time, signal, factor=10)

        # Should select indices 0, 10, 20, ..., 90, 99
        expected_indices = list(range(0, 100, 10))
        if 99 not in expected_indices:
            expected_indices.append(99)

        np.testing.assert_array_equal(time_d[:-1], time[expected_indices[:-1]])
        np.testing.assert_array_equal(signal_d[:-1], signal[expected_indices[:-1]])

    def test_decimate_copies_arrays(self):
        """Test that decimation returns copies."""
        time = np.linspace(0, 10, 100)
        signal = np.sin(time)

        time_d, signal_d = simple_decimate(time, signal, factor=5)

        # Modify decimated
        time_d[0] = 999
        signal_d[0] = 999

        # Original should be unchanged
        self.assertNotEqual(time[0], 999)
        self.assertNotEqual(signal[0], 999)


if __name__ == "__main__":
    unittest.main()
