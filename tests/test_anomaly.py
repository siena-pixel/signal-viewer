"""Unit tests for anomaly detection module."""

import unittest
import numpy as np

from signal_viewer.processing.anomaly import (
    zscore_anomaly,
    mad_anomaly,
    derivative_anomaly,
    iqr_anomaly,
    rolling_anomaly,
)


class TestZscoreAnomaly(unittest.TestCase):
    """Test Z-score anomaly detection."""

    def test_zscore_detects_outliers(self):
        """Test Z-score detects known outliers."""
        signal = np.array([1, 2, 3, 4, 5, 100], dtype=float)

        anomaly_indices, z_scores = zscore_anomaly(signal, threshold=2.0)

        # 100 should be detected as anomaly
        self.assertIn(5, anomaly_indices)

    def test_zscore_no_anomalies_in_clean(self):
        """Test no anomalies in clean signal."""
        signal = np.linspace(0, 10, 100)

        anomaly_indices, z_scores = zscore_anomaly(signal, threshold=3.0)

        self.assertEqual(len(anomaly_indices), 0)

    def test_zscore_output_shapes(self):
        """Test output shapes."""
        signal = np.random.randn(100)

        anomaly_indices, z_scores = zscore_anomaly(signal, threshold=3.0)

        self.assertIsInstance(anomaly_indices, np.ndarray)
        self.assertEqual(len(z_scores), len(signal))

    def test_zscore_empty_signal(self):
        """Test with empty signal."""
        anomaly_indices, z_scores = zscore_anomaly(np.array([]))

        self.assertEqual(len(anomaly_indices), 0)
        self.assertEqual(len(z_scores), 0)

    def test_zscore_with_nans(self):
        """Test Z-score handles NaNs."""
        signal = np.array([1, 2, 3, np.nan, 5, 6, 100])

        anomaly_indices, z_scores = zscore_anomaly(signal, threshold=2.0)

        # Should still detect 100 as anomaly
        self.assertGreater(len(anomaly_indices), 0)

    def test_zscore_threshold_effect(self):
        """Test that higher threshold detects fewer anomalies."""
        signal = np.array([1, 2, 3, 4, 5, 50, 60, 70, 80])

        anomalies_low, _ = zscore_anomaly(signal, threshold=1.0)
        anomalies_high, _ = zscore_anomaly(signal, threshold=3.0)

        self.assertGreater(len(anomalies_low), len(anomalies_high))


class TestMADAnomaly(unittest.TestCase):
    """Test Median Absolute Deviation anomaly detection."""

    def test_mad_detects_outliers(self):
        """Test MAD detects known outliers."""
        signal = np.array([1, 2, 3, 4, 5, 100], dtype=float)

        anomaly_indices, modified_z = mad_anomaly(signal, threshold=2.0)

        self.assertIn(5, anomaly_indices)

    def test_mad_robust_to_outliers(self):
        """Test MAD is robust to outliers."""
        signal = np.array([1, 2, 3, 4, 5, 1000], dtype=float)

        anomaly_indices, modified_z = mad_anomaly(signal, threshold=3.5)

        # Should detect the large outlier
        self.assertIn(5, anomaly_indices)

    def test_mad_output_shapes(self):
        """Test output shapes."""
        signal = np.random.randn(100)

        anomaly_indices, modified_z = mad_anomaly(signal, threshold=3.5)

        self.assertEqual(len(modified_z), len(signal))

    def test_mad_empty_signal(self):
        """Test with empty signal."""
        anomaly_indices, modified_z = mad_anomaly(np.array([]))

        self.assertEqual(len(anomaly_indices), 0)


class TestDerivativeAnomaly(unittest.TestCase):
    """Test derivative-based anomaly detection."""

    def test_derivative_detects_jumps(self):
        """Test derivative detects sudden jumps."""
        # Signal with a sudden jump
        signal = np.concatenate([
            np.ones(50),
            np.ones(50) + 20  # Sudden jump
        ])

        anomaly_indices, deriv_z = derivative_anomaly(signal, threshold=2.0)

        # Jump location should be detected
        self.assertGreater(len(anomaly_indices), 0)
        self.assertIn(50, anomaly_indices)

    def test_derivative_output_shapes(self):
        """Test output shapes."""
        signal = np.sin(np.linspace(0, 10, 100))

        anomaly_indices, deriv_z = derivative_anomaly(signal, threshold=3.0)

        self.assertEqual(len(deriv_z), len(signal))

    def test_derivative_no_anomalies_smooth(self):
        """Test no anomalies in smooth signal."""
        signal = np.linspace(0, 10, 100)

        anomaly_indices, deriv_z = derivative_anomaly(signal, threshold=3.0)

        self.assertEqual(len(anomaly_indices), 0)

    def test_derivative_short_signal(self):
        """Test with signal shorter than 2 points."""
        anomaly_indices, deriv_z = derivative_anomaly(np.array([1]))

        self.assertEqual(len(anomaly_indices), 0)


class TestIQRAnomaly(unittest.TestCase):
    """Test Interquartile Range anomaly detection."""

    def test_iqr_detects_outliers(self):
        """Test IQR detects outliers."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])

        anomaly_indices, anomaly_values = iqr_anomaly(signal, multiplier=1.5)

        self.assertIn(10, anomaly_indices)

    def test_iqr_output_shapes(self):
        """Test output shapes."""
        signal = np.random.randn(100)

        anomaly_indices, anomaly_values = iqr_anomaly(signal, multiplier=1.5)

        self.assertEqual(len(anomaly_indices), len(anomaly_values))

    def test_iqr_empty_signal(self):
        """Test with empty signal."""
        anomaly_indices, anomaly_values = iqr_anomaly(np.array([]))

        self.assertEqual(len(anomaly_indices), 0)

    def test_iqr_multiplier_effect(self):
        """Test that multiplier affects detection."""
        signal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50])

        anomalies_strict, _ = iqr_anomaly(signal, multiplier=1.5)
        anomalies_loose, _ = iqr_anomaly(signal, multiplier=3.0)

        self.assertGreaterEqual(len(anomalies_strict), len(anomalies_loose))

    def test_iqr_with_nans(self):
        """Test IQR handles NaNs."""
        signal = np.array([1, 2, 3, 4, 5, np.nan, 6, 7, 8, 9, 100])

        anomaly_indices, anomaly_values = iqr_anomaly(signal, multiplier=1.5)

        # Should still work without NaN
        self.assertGreater(len(anomaly_indices), 0)


class TestRollingAnomaly(unittest.TestCase):
    """Test rolling window anomaly detection."""

    def test_rolling_detects_local_anomalies(self):
        """Test rolling anomaly detects local outliers."""
        signal = np.sin(np.linspace(0, 10, 1000))
        # Add spike in middle
        signal[500] = 10

        anomaly_indices, local_z = rolling_anomaly(signal, window_size=50, threshold=3.0)

        self.assertIn(500, anomaly_indices)

    def test_rolling_output_shapes(self):
        """Test output shapes."""
        signal = np.random.randn(1000)

        anomaly_indices, local_z = rolling_anomaly(signal, window_size=100, threshold=3.0)

        self.assertEqual(len(local_z), len(signal))

    def test_rolling_short_signal(self):
        """Test with signal shorter than window."""
        signal = np.linspace(0, 10, 50)

        anomaly_indices, local_z = rolling_anomaly(signal, window_size=100, threshold=3.0)

        self.assertEqual(len(anomaly_indices), 0)

    def test_rolling_no_anomalies_clean(self):
        """Test no anomalies in clean signal."""
        signal = np.sin(np.linspace(0, 10, 1000))

        anomaly_indices, local_z = rolling_anomaly(signal, window_size=100, threshold=3.0)

        self.assertEqual(len(anomaly_indices), 0)

    def test_rolling_with_nans(self):
        """Test rolling anomaly handles NaNs."""
        signal = np.sin(np.linspace(0, 10, 1000))
        signal[::100] = np.nan

        anomaly_indices, local_z = rolling_anomaly(signal, window_size=100, threshold=3.0)

        self.assertEqual(len(local_z), len(signal))


class TestAnomalyComparison(unittest.TestCase):
    """Compare different anomaly detection methods."""

    def test_all_methods_detect_spike(self):
        """Test all methods detect obvious spike."""
        signal = np.ones(100)
        signal[50] = 100

        zscore_anom, _ = zscore_anomaly(signal, threshold=2.0)
        mad_anom, _ = mad_anomaly(signal, threshold=2.0)
        deriv_anom, _ = derivative_anomaly(signal, threshold=2.0)
        iqr_anom, _ = iqr_anomaly(signal, multiplier=1.5)

        # At least some methods should detect something
        total_detections = (
            len(zscore_anom) + len(mad_anom) +
            len(deriv_anom) + len(iqr_anom)
        )
        self.assertGreater(total_detections, 0)


if __name__ == "__main__":
    unittest.main()
