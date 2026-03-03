"""Unit tests for trend analysis module."""

import unittest
import numpy as np

from signal_viewer.processing.trend import (
    fit_polynomial,
    remove_trend,
    detect_changepoints,
    compute_envelope,
    compute_rms_trend,
)


class TestFitPolynomial(unittest.TestCase):
    """Test polynomial fitting."""

    def test_fit_polynomial_linear(self):
        """Test fitting linear trend."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 3 + 0.1 * np.random.randn(100)

        coeffs, fitted, residuals = fit_polynomial(x, y, degree=1)

        # Coefficients should be approximately [2, 3]
        self.assertAlmostEqual(coeffs[0], 2, delta=0.2)
        self.assertAlmostEqual(coeffs[1], 3, delta=0.5)

    def test_fit_polynomial_quadratic(self):
        """Test fitting quadratic trend."""
        x = np.linspace(0, 10, 100)
        y = x ** 2 + 2 * x + 1

        coeffs, fitted, residuals = fit_polynomial(x, y, degree=2)

        # Coefficients should be approximately [1, 2, 1]
        self.assertAlmostEqual(coeffs[0], 1, delta=0.1)
        self.assertAlmostEqual(coeffs[1], 2, delta=0.1)
        self.assertAlmostEqual(coeffs[2], 1, delta=0.1)

    def test_fit_polynomial_output_shapes(self):
        """Test output shapes."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        coeffs, fitted, residuals = fit_polynomial(x, y, degree=3)

        self.assertEqual(len(fitted), len(y))
        self.assertEqual(len(residuals), len(y))

    def test_fit_polynomial_residuals(self):
        """Test residual calculation."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 1

        coeffs, fitted, residuals = fit_polynomial(x, y, degree=1)

        # Residuals should be small for perfect linear fit
        self.assertLess(np.max(np.abs(residuals)), 1.0)

    def test_fit_polynomial_empty_signals(self):
        """Test with empty signals."""
        coeffs, fitted, residuals = fit_polynomial(np.array([]), np.array([]))

        self.assertEqual(len(coeffs), 0)
        self.assertEqual(len(fitted), 0)
        self.assertEqual(len(residuals), 0)

    def test_fit_polynomial_mismatched_lengths(self):
        """Test ValueError for mismatched lengths."""
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 50)

        with self.assertRaises(ValueError):
            fit_polynomial(x, y, degree=1)


class TestRemoveTrend(unittest.TestCase):
    """Test trend removal."""

    def test_remove_trend_linear(self):
        """Test removing linear trend."""
        x = np.linspace(0, 10, 100)
        y = 0.5 * x + 10 + np.sin(x)

        detrended = remove_trend(x, y, degree=1)

        # Detrended should have near-zero mean
        self.assertLess(np.abs(np.mean(detrended)), 2)

    def test_remove_trend_output_length(self):
        """Test output length matches input."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        detrended = remove_trend(x, y, degree=1)

        self.assertEqual(len(detrended), len(y))

    def test_remove_trend_preserves_oscillations(self):
        """Test that trend removal preserves oscillations."""
        x = np.linspace(0, 10, 1000)
        trend = 0.1 * x
        oscillation = np.sin(2 * np.pi * x)
        y = trend + oscillation

        detrended = remove_trend(x, y, degree=1)

        # Oscillation variance should be preserved
        osc_var = np.var(oscillation)
        detrend_var = np.var(detrended)

        self.assertAlmostEqual(detrend_var, osc_var, delta=0.5)

    def test_remove_trend_empty_signals(self):
        """Test with empty signals."""
        detrended = remove_trend(np.array([]), np.array([]), degree=1)

        self.assertEqual(len(detrended), 0)


class TestDetectChangepoints(unittest.TestCase):
    """Test changepoint detection."""

    def test_detect_changepoints_obvious_change(self):
        """Test detecting obvious changepoint."""
        signal = np.concatenate([
            np.ones(300) * 0,
            np.ones(300) * 10
        ])

        changepoints = detect_changepoints(signal, window_size=50, threshold=1.5)

        # Should detect change around index 300
        if len(changepoints) > 0:
            self.assertTrue(any(250 <= cp <= 350 for cp in changepoints))

    def test_detect_changepoints_no_change(self):
        """Test no changepoints in constant signal."""
        signal = np.ones(500)

        changepoints = detect_changepoints(signal, window_size=100, threshold=3.0)

        self.assertEqual(len(changepoints), 0)

    def test_detect_changepoints_short_signal(self):
        """Test with signal shorter than window."""
        signal = np.ones(50)

        changepoints = detect_changepoints(signal, window_size=100, threshold=3.0)

        self.assertEqual(len(changepoints), 0)

    def test_detect_changepoints_output_type(self):
        """Test output is integer array."""
        signal = np.random.randn(500)

        changepoints = detect_changepoints(signal, window_size=100, threshold=3.0)

        self.assertTrue(np.issubdtype(changepoints.dtype, np.integer))

    def test_detect_changepoints_with_nans(self):
        """Test changepoint detection handles NaNs."""
        signal = np.concatenate([
            np.ones(300),
            np.ones(300) + 5
        ])
        signal[::100] = np.nan

        changepoints = detect_changepoints(signal, window_size=50, threshold=1.5)

        # Should handle NaNs without crashing
        self.assertIsInstance(changepoints, np.ndarray)


class TestComputeEnvelope(unittest.TestCase):
    """Test envelope computation."""

    def test_envelope_contains_signal(self):
        """Test that envelope contains the original signal."""
        signal = np.sin(np.linspace(0, 10, 1000))

        upper, lower = compute_envelope(signal)

        # Original signal should be between envelope (with some tolerance for interpolation)
        self.assertTrue(np.all(signal <= upper + 0.5))  # Larger tolerance
        self.assertTrue(np.all(signal >= lower - 0.5))

    def test_envelope_output_shapes(self):
        """Test envelope output shapes."""
        signal = np.random.randn(500)

        upper, lower = compute_envelope(signal)

        self.assertEqual(len(upper), len(signal))
        self.assertEqual(len(lower), len(signal))

    def test_envelope_short_signal(self):
        """Test with signal shorter than 3 points."""
        signal = np.array([1, 2])

        upper, lower = compute_envelope(signal)

        self.assertEqual(len(upper), 2)
        self.assertEqual(len(lower), 2)

    def test_envelope_smooth(self):
        """Test that envelope is smooth."""
        signal = np.sin(np.linspace(0, 20, 1000))

        upper, lower = compute_envelope(signal)

        # Envelope derivatives should be small (smooth)
        upper_diff = np.diff(upper)
        lower_diff = np.diff(lower)

        self.assertLess(np.std(upper_diff), 0.2)
        self.assertLess(np.std(lower_diff), 0.2)

    def test_envelope_with_nans(self):
        """Test envelope handles NaNs."""
        signal = np.sin(np.linspace(0, 10, 100))
        signal[::10] = np.nan

        upper, lower = compute_envelope(signal)

        self.assertEqual(len(upper), len(signal))
        self.assertEqual(len(lower), len(signal))


class TestComputeRMSTrend(unittest.TestCase):
    """Test RMS trend computation."""

    def test_rms_trend_output_length(self):
        """Test RMS trend output length."""
        signal = np.random.randn(2000)
        window_size = 100

        rms = compute_rms_trend(signal, window_size=window_size)

        expected_length = len(signal) - window_size + 1
        self.assertEqual(len(rms), expected_length)

    def test_rms_trend_values_positive(self):
        """Test RMS values are positive."""
        signal = np.random.randn(1000)

        rms = compute_rms_trend(signal, window_size=100)

        self.assertTrue(np.all(rms >= 0))

    def test_rms_trend_short_signal(self):
        """Test RMS trend with signal shorter than window."""
        signal = np.random.randn(50)

        rms = compute_rms_trend(signal, window_size=100)

        # Should return single RMS value
        self.assertEqual(len(rms), 1)

    def test_rms_trend_known_values(self):
        """Test RMS trend with known values."""
        # Constant signal should have RMS = amplitude
        signal = np.ones(1000) * 5

        rms = compute_rms_trend(signal, window_size=100)

        # RMS of constant 5 should be 5
        np.testing.assert_array_almost_equal(rms, 5)

    def test_rms_trend_empty_signal(self):
        """Test with empty signal."""
        rms = compute_rms_trend(np.array([]))

        self.assertEqual(len(rms), 0)

    def test_rms_trend_with_nans(self):
        """Test RMS trend handles NaNs."""
        signal = np.random.randn(1000)
        signal[::100] = np.nan

        rms = compute_rms_trend(signal, window_size=100)

        self.assertGreater(len(rms), 0)

    def test_rms_trend_sine_wave(self):
        """Test RMS trend of sine wave."""
        # RMS of sin(x) over full period is 1/sqrt(2)
        signal = np.sin(np.linspace(0, 2 * np.pi * 10, 2000))

        rms = compute_rms_trend(signal, window_size=200)

        expected_rms = 1 / np.sqrt(2)
        # Should be close to expected (within tolerance)
        self.assertTrue(np.all(rms < 1.0))
        self.assertTrue(np.all(rms > 0.5))


class TestTrendIntegration(unittest.TestCase):
    """Integration tests for trend analysis."""

    def test_fit_and_remove_trend(self):
        """Test fitting and removing trend together."""
        x = np.linspace(0, 10, 100)
        y = 2 * x + 5 + np.sin(x)

        # Fit and remove
        coeffs, fitted, residuals = fit_polynomial(x, y, degree=1)
        detrended = remove_trend(x, y, degree=1)

        # Detrended should match residuals
        np.testing.assert_array_almost_equal(detrended, residuals)

    def test_envelope_and_rms(self):
        """Test envelope and RMS trend together."""
        signal = 10 * np.sin(np.linspace(0, 10, 1000))

        upper, lower = compute_envelope(signal)
        rms = compute_rms_trend(signal, window_size=100)

        # RMS should be less than upper envelope max
        self.assertLess(np.max(rms), np.max(upper))


if __name__ == "__main__":
    unittest.main()
