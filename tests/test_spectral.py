"""Unit tests for spectral analysis module."""

import unittest
import numpy as np

from signal_viewer.processing.spectral import (
    compute_fft,
    compute_psd_welch,
    compute_spectrogram,
    find_dominant_frequencies,
)


class TestComputeFFT(unittest.TestCase):
    """Test FFT computation."""

    def test_fft_sine_wave(self):
        """Test FFT of known sine wave."""
        sampling_rate = 1000.0  # Hz
        duration = 1.0  # seconds
        frequency = 10.0  # Hz

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)

        frequencies, magnitude_db = compute_fft(signal, sampling_rate)

        # Find peak
        peak_idx = np.argmax(magnitude_db)
        peak_freq = frequencies[peak_idx]

        # Peak should be at 10 Hz (within reasonable tolerance)
        self.assertAlmostEqual(peak_freq, frequency, delta=5)

    def test_fft_output_shape(self):
        """Test FFT output shapes."""
        signal = np.sin(np.linspace(0, 10, 1000))
        frequencies, magnitude_db = compute_fft(signal, 100.0)

        self.assertEqual(len(frequencies), len(magnitude_db))
        self.assertGreater(len(frequencies), 0)

    def test_fft_empty_signal(self):
        """Test FFT with empty signal."""
        frequencies, magnitude_db = compute_fft(np.array([]), 100.0)

        self.assertEqual(len(frequencies), 0)
        self.assertEqual(len(magnitude_db), 0)

    def test_fft_with_nans(self):
        """Test FFT handles NaNs."""
        signal = np.sin(np.linspace(0, 10, 100))
        signal[::10] = np.nan

        frequencies, magnitude_db = compute_fft(signal, 100.0)

        self.assertGreater(len(frequencies), 0)
        self.assertFalse(np.any(np.isnan(magnitude_db)))

    def test_fft_output_dB(self):
        """Test that FFT output is in dB scale."""
        signal = np.sin(np.linspace(0, 10, 1000))
        frequencies, magnitude_db = compute_fft(signal, 100.0)

        # dB values should generally be negative
        self.assertTrue(np.all(magnitude_db <= 0))


class TestComputePSDWelch(unittest.TestCase):
    """Test Welch's PSD method."""

    def test_psd_welch_output_shape(self):
        """Test PSD Welch output shapes."""
        signal = np.sin(np.linspace(0, 10, 2000))
        frequencies, psd = compute_psd_welch(signal, 100.0)

        self.assertEqual(len(frequencies), len(psd))
        self.assertGreater(len(frequencies), 0)

    def test_psd_welch_known_signal(self):
        """Test PSD Welch on known signal."""
        sampling_rate = 1000.0
        duration = 2.0
        frequency = 50.0

        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        signal = np.sin(2 * np.pi * frequency * t)

        frequencies, psd = compute_psd_welch(signal, sampling_rate)

        # Find peak
        peak_idx = np.argmax(psd)
        peak_freq = frequencies[peak_idx]

        # Peak should be near 50 Hz
        self.assertAlmostEqual(peak_freq, frequency, delta=20)

    def test_psd_welch_empty_signal(self):
        """Test PSD Welch with empty signal."""
        frequencies, psd = compute_psd_welch(np.array([]), 100.0)

        self.assertEqual(len(frequencies), 0)
        self.assertEqual(len(psd), 0)

    def test_psd_welch_with_nans(self):
        """Test PSD Welch handles NaNs."""
        signal = np.sin(np.linspace(0, 10, 1000))
        signal[::100] = np.nan

        frequencies, psd = compute_psd_welch(signal, 100.0)

        self.assertGreater(len(frequencies), 0)
        self.assertFalse(np.any(np.isnan(psd)))

    def test_psd_welch_parameters(self):
        """Test PSD Welch with different parameters."""
        signal = np.sin(np.linspace(0, 10, 2000))

        freq1, psd1 = compute_psd_welch(signal, 100.0, nperseg=256)
        freq2, psd2 = compute_psd_welch(signal, 100.0, nperseg=512)

        # Different segment sizes should give different frequencies
        self.assertNotEqual(len(freq1), len(freq2))


class TestComputeSpectrogram(unittest.TestCase):
    """Test spectrogram computation."""

    def test_spectrogram_output_shape(self):
        """Test spectrogram output shapes."""
        signal = np.sin(np.linspace(0, 10, 2000))
        times, frequencies, Sxx = compute_spectrogram(signal, 100.0)

        self.assertEqual(len(times), Sxx.shape[1])
        self.assertEqual(len(frequencies), Sxx.shape[0])
        self.assertGreater(len(times), 0)
        self.assertGreater(len(frequencies), 0)

    def test_spectrogram_empty_signal(self):
        """Test spectrogram with empty signal."""
        times, frequencies, Sxx = compute_spectrogram(np.array([]), 100.0)

        self.assertEqual(len(times), 0)
        self.assertEqual(len(frequencies), 0)
        self.assertEqual(Sxx.shape, (0, 0))

    def test_spectrogram_with_nans(self):
        """Test spectrogram handles NaNs."""
        signal = np.sin(np.linspace(0, 10, 1000))
        signal[::100] = np.nan

        times, frequencies, Sxx = compute_spectrogram(signal, 100.0)

        self.assertGreater(len(times), 0)
        self.assertFalse(np.any(np.isnan(Sxx)))

    def test_spectrogram_time_frequency_resolution(self):
        """Test time-frequency resolution changes with parameters."""
        signal = np.sin(np.linspace(0, 10, 2000))

        times1, freq1, Sxx1 = compute_spectrogram(signal, 100.0, nperseg=128)
        times2, freq2, Sxx2 = compute_spectrogram(signal, 100.0, nperseg=512)

        # Larger nperseg gives better frequency resolution but worse time resolution
        self.assertGreater(len(freq2), len(freq1))  # More frequencies
        self.assertLess(len(times2), len(times1))   # Fewer time points


class TestFindDominantFrequencies(unittest.TestCase):
    """Test dominant frequency detection."""

    def test_find_dominant_frequencies_basic(self):
        """Test finding dominant frequencies."""
        # Create PSD with clear peak
        frequencies = np.linspace(0, 50, 1000)
        psd = np.exp(-((frequencies - 20) ** 2) / 10)  # Gaussian peak at 20 Hz

        peaks = find_dominant_frequencies(frequencies, psd, n_peaks=1)

        self.assertEqual(len(peaks), 1)
        freq, power = peaks[0]
        self.assertGreater(freq, 15)
        self.assertLess(freq, 25)

    def test_find_dominant_frequencies_multiple_peaks(self):
        """Test finding multiple peaks."""
        frequencies = np.linspace(0, 100, 1000)
        psd = (
            np.exp(-((frequencies - 20) ** 2) / 10)
            + np.exp(-((frequencies - 50) ** 2) / 10)
            + np.exp(-((frequencies - 80) ** 2) / 10)
        )

        peaks = find_dominant_frequencies(frequencies, psd, n_peaks=5)

        self.assertEqual(len(peaks), 3)  # Only 3 peaks available
        # Should be sorted by power (descending)
        self.assertTrue(peaks[0][1] >= peaks[1][1])
        self.assertTrue(peaks[1][1] >= peaks[2][1])

    def test_find_dominant_frequencies_empty(self):
        """Test with empty inputs."""
        peaks = find_dominant_frequencies(np.array([]), np.array([]))
        self.assertEqual(len(peaks), 0)

    def test_find_dominant_frequencies_returns_tuples(self):
        """Test that results are (frequency, power) tuples."""
        frequencies = np.linspace(0, 50, 100)
        psd = np.sin(frequencies)

        peaks = find_dominant_frequencies(frequencies, psd, n_peaks=3)

        for freq, power in peaks:
            self.assertIsInstance(freq, (int, float, np.number))
            self.assertIsInstance(power, (int, float, np.number))

    def test_find_dominant_frequencies_sorted(self):
        """Test that results are sorted by power."""
        frequencies = np.linspace(0, 50, 100)
        psd = np.random.rand(100)

        peaks = find_dominant_frequencies(frequencies, psd, n_peaks=10)

        # Check sorted by power (descending)
        for i in range(len(peaks) - 1):
            self.assertGreaterEqual(peaks[i][1], peaks[i + 1][1])


if __name__ == "__main__":
    unittest.main()
