"""Unit tests for the trend analysis module."""

import unittest
import numpy as np

from signal_viewer.processing.trend import (
    fit_polynomial,
    remove_trend,
    detect_changepoints,
    compute_envelope,
    compute_rms_trend,
)


# ---------------------------------------------------------------------------
# Polynomial fitting
# ---------------------------------------------------------------------------

class TestFitPolynomial(unittest.TestCase):
    """Tests for fit_polynomial (coefficients, fitted curve, residuals)."""

    def test_linear(self):
        x = np.linspace(0, 10, 100)
        y = 2 * x + 3 + 0.1 * np.random.randn(100)
        coeffs, _, _ = fit_polynomial(x, y, degree=1)
        self.assertAlmostEqual(coeffs[0], 2, delta=0.2)
        self.assertAlmostEqual(coeffs[1], 3, delta=0.5)

    def test_quadratic(self):
        x = np.linspace(0, 10, 100)
        y = x ** 2 + 2 * x + 1
        coeffs, _, _ = fit_polynomial(x, y, degree=2)
        self.assertAlmostEqual(coeffs[0], 1, delta=0.1)
        self.assertAlmostEqual(coeffs[1], 2, delta=0.1)
        self.assertAlmostEqual(coeffs[2], 1, delta=0.1)

    def test_output_shapes(self):
        x = np.linspace(0, 10, 100)
        _, fitted, residuals = fit_polynomial(x, np.sin(x), degree=3)
        self.assertEqual(len(fitted), 100)
        self.assertEqual(len(residuals), 100)

    def test_residuals_small_for_exact_fit(self):
        x = np.linspace(0, 10, 100)
        _, _, residuals = fit_polynomial(x, 2 * x + 1, degree=1)
        self.assertLess(np.max(np.abs(residuals)), 1.0)

    def test_empty_signals(self):
        coeffs, fitted, residuals = fit_polynomial(np.array([]), np.array([]))
        self.assertEqual(len(coeffs), 0)
        self.assertEqual(len(fitted), 0)
        self.assertEqual(len(residuals), 0)

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            fit_polynomial(np.linspace(0, 10, 100), np.zeros(50), degree=1)


# ---------------------------------------------------------------------------
# Trend removal
# ---------------------------------------------------------------------------

class TestRemoveTrend(unittest.TestCase):
    """Tests for remove_trend."""

    def test_linear_detrend(self):
        x = np.linspace(0, 10, 100)
        y = 0.5 * x + 10 + np.sin(x)
        detrended = remove_trend(x, y, degree=1)
        self.assertLess(np.abs(np.mean(detrended)), 2)

    def test_output_length(self):
        x = np.linspace(0, 10, 100)
        self.assertEqual(len(remove_trend(x, np.sin(x), degree=1)), 100)

    def test_preserves_oscillation_variance(self):
        x = np.linspace(0, 10, 1000)
        osc = np.sin(2 * np.pi * x)
        detrended = remove_trend(x, 0.1 * x + osc, degree=1)
        self.assertAlmostEqual(np.var(detrended), np.var(osc), delta=0.5)

    def test_empty_signals(self):
        self.assertEqual(
            len(remove_trend(np.array([]), np.array([]), degree=1)), 0
        )


# ---------------------------------------------------------------------------
# Changepoint detection
# ---------------------------------------------------------------------------

class TestDetectChangepoints(unittest.TestCase):
    """Tests for detect_changepoints."""

    def test_obvious_change(self):
        signal = np.concatenate([np.zeros(300), np.ones(300) * 10])
        cps = detect_changepoints(signal, window_size=50, threshold=1.5)
        if len(cps) > 0:
            self.assertTrue(any(250 <= cp <= 350 for cp in cps))

    def test_constant_signal_no_change(self):
        cps = detect_changepoints(np.ones(500), window_size=100, threshold=3.0)
        self.assertEqual(len(cps), 0)

    def test_short_signal(self):
        cps = detect_changepoints(np.ones(50), window_size=100, threshold=3.0)
        self.assertEqual(len(cps), 0)

    def test_output_dtype_integer(self):
        cps = detect_changepoints(
            np.random.randn(500), window_size=100, threshold=3.0
        )
        self.assertTrue(np.issubdtype(cps.dtype, np.integer))

    def test_handles_nans(self):
        signal = np.concatenate([np.ones(300), np.ones(300) + 5])
        signal[::100] = np.nan
        cps = detect_changepoints(signal, window_size=50, threshold=1.5)
        self.assertIsInstance(cps, np.ndarray)


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

class TestComputeEnvelope(unittest.TestCase):
    """Tests for compute_envelope."""

    def test_envelope_contains_signal(self):
        s = np.sin(np.linspace(0, 10, 1000))
        upper, lower = compute_envelope(s)
        self.assertTrue(np.all(s <= upper + 0.5))
        self.assertTrue(np.all(s >= lower - 0.5))

    def test_output_shapes(self):
        s = np.random.randn(500)
        upper, lower = compute_envelope(s)
        self.assertEqual(len(upper), 500)
        self.assertEqual(len(lower), 500)

    def test_short_signal(self):
        upper, lower = compute_envelope(np.array([1, 2]))
        self.assertEqual(len(upper), 2)
        self.assertEqual(len(lower), 2)

    def test_smoothness(self):
        upper, lower = compute_envelope(np.sin(np.linspace(0, 20, 1000)))
        self.assertLess(np.std(np.diff(upper)), 0.2)
        self.assertLess(np.std(np.diff(lower)), 0.2)

    def test_handles_nans(self):
        s = np.sin(np.linspace(0, 10, 100))
        s[::10] = np.nan
        upper, lower = compute_envelope(s)
        self.assertEqual(len(upper), 100)
        self.assertEqual(len(lower), 100)


# ---------------------------------------------------------------------------
# RMS trend
# ---------------------------------------------------------------------------

class TestComputeRMSTrend(unittest.TestCase):
    """Tests for compute_rms_trend."""

    def test_output_length(self):
        rms = compute_rms_trend(np.random.randn(2000), window_size=100)
        self.assertEqual(len(rms), 2000 - 100 + 1)

    def test_values_positive(self):
        rms = compute_rms_trend(np.random.randn(1000), window_size=100)
        self.assertTrue(np.all(rms >= 0))

    def test_short_signal_single_value(self):
        rms = compute_rms_trend(np.random.randn(50), window_size=100)
        self.assertEqual(len(rms), 1)

    def test_constant_signal_rms_equals_amplitude(self):
        rms = compute_rms_trend(np.ones(1000) * 5, window_size=100)
        np.testing.assert_array_almost_equal(rms, 5)

    def test_empty_signal(self):
        self.assertEqual(len(compute_rms_trend(np.array([]))), 0)

    def test_handles_nans(self):
        s = np.random.randn(1000)
        s[::100] = np.nan
        self.assertGreater(
            len(compute_rms_trend(s, window_size=100)), 0
        )

    def test_sine_wave_rms(self):
        """RMS of a sine should be around 1/√2 ≈ 0.707."""
        s = np.sin(np.linspace(0, 2 * np.pi * 10, 2000))
        rms = compute_rms_trend(s, window_size=200)
        self.assertTrue(np.all(rms < 1.0))
        self.assertTrue(np.all(rms > 0.5))


# ---------------------------------------------------------------------------
# Integration (fit + remove, envelope + RMS)
# ---------------------------------------------------------------------------

class TestTrendIntegration(unittest.TestCase):

    def test_detrended_matches_residuals(self):
        x = np.linspace(0, 10, 100)
        y = 2 * x + 5 + np.sin(x)
        _, _, residuals = fit_polynomial(x, y, degree=1)
        detrended = remove_trend(x, y, degree=1)
        np.testing.assert_array_almost_equal(detrended, residuals)

    def test_rms_below_envelope_max(self):
        s = 10 * np.sin(np.linspace(0, 10, 1000))
        upper, _ = compute_envelope(s)
        rms = compute_rms_trend(s, window_size=100)
        self.assertLess(np.max(rms), np.max(upper))


if __name__ == "__main__":
    unittest.main()
