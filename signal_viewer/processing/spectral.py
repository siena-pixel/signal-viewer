"""Spectral analysis using FFT and Welch's method."""

from typing import Tuple, List, Dict
import numpy as np


def _get_window(window_type: str, length: int) -> np.ndarray:
    """Generate window function."""
    if window_type == "hann":
        return np.hanning(length)
    elif window_type == "hamming":
        return np.hamming(length)
    elif window_type == "blackman":
        return np.blackman(length)
    elif window_type == "rectangular":
        return np.ones(length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")


def compute_fft(
    signal: np.ndarray, sampling_rate: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute FFT of a real signal.
    
    Uses rfft for real signals (only positive frequencies).
    Returns magnitude in dB scale.
    
    Args:
        signal: 1D array of signal values.
        sampling_rate: Sampling frequency in Hz.
    
    Returns:
        Tuple of (frequencies, magnitude_db) arrays.
        frequencies: Positive frequencies in Hz.
        magnitude_db: Magnitude spectrum in dB (20*log10(|FFT|/N)).
    """
    if len(signal) == 0:
        return np.array([]), np.array([])
    
    # Remove NaN values
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.array([]), np.array([])
    
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    
    # Convert to dB (avoid log of zero)
    magnitude = np.abs(fft_vals) / n
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-10))
    
    return frequencies, magnitude_db


def compute_psd_welch(
    signal: np.ndarray,
    sampling_rate: float,
    nperseg: int = 1024,
    noverlap: int = None,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.
    
    Segments the signal, applies window, computes FFT of each segment,
    and averages the periodograms for a smoother estimate.
    
    Args:
        signal: 1D array of signal values.
        sampling_rate: Sampling frequency in Hz.
        nperseg: Length of each segment (default: 1024).
        noverlap: Number of points to overlap (default: nperseg // 2).
        window: Window type ('hann', 'hamming', 'blackman', 'rectangular').
    
    Returns:
        Tuple of (frequencies, psd) arrays.
        frequencies: Frequency values in Hz.
        psd: Power spectral density in dB/Hz.
    """
    if len(signal) == 0:
        return np.array([]), np.array([])
    
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.array([]), np.array([])
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    nperseg = min(nperseg, len(signal))
    noverlap = min(noverlap, nperseg - 1)
    
    # Generate window
    win = _get_window(window, nperseg)
    
    # Compute scaling factor
    win_sum = np.sum(win)
    win_sum_sq = np.sum(win ** 2)
    scale = sampling_rate / (win_sum_sq * (sampling_rate ** 2))
    
    # Segment the signal
    step = nperseg - noverlap
    n_segments = (len(signal) - noverlap) // step
    
    if n_segments < 1:
        # Signal too short, use entire signal
        windowed = signal * _get_window(window, len(signal))
        fft_vals = np.fft.rfft(windowed)
        frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sampling_rate)
        psd = (np.abs(fft_vals) ** 2) * scale
    else:
        # Average periodograms
        psd_vals = []
        for i in range(n_segments):
            start = i * step
            end = start + nperseg
            segment = signal[start:end] * win
            fft_vals = np.fft.rfft(segment)
            psd_segment = (np.abs(fft_vals) ** 2) * scale
            psd_vals.append(psd_segment)
        
        psd = np.mean(psd_vals, axis=0)
        frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sampling_rate)
    
    # Convert to dB
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-10))
    
    return frequencies, psd_db


def compute_spectrogram(
    signal: np.ndarray,
    sampling_rate: float,
    nperseg: int = 256,
    noverlap: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute spectrogram using Short-Time Fourier Transform (STFT).
    
    Applies sliding window FFT to show time-frequency evolution.
    Returns power in dB scale.
    
    Args:
        signal: 1D array of signal values.
        sampling_rate: Sampling frequency in Hz.
        nperseg: Length of each segment (default: 256).
        noverlap: Number of points to overlap (default: nperseg // 2).
    
    Returns:
        Tuple of (times, frequencies, Sxx).
        times: Time values for each segment center.
        frequencies: Frequency values in Hz.
        Sxx: Spectrogram power in dB (n_freq x n_time).
    """
    if len(signal) == 0:
        return np.array([]), np.array([]), np.array([]).reshape(0, 0)
    
    signal = signal[~np.isnan(signal)]
    if len(signal) == 0:
        return np.array([]), np.array([]), np.array([]).reshape(0, 0)
    
    if noverlap is None:
        noverlap = nperseg // 2
    
    nperseg = min(nperseg, len(signal))
    noverlap = min(noverlap, nperseg - 1)
    
    # Generate window
    win = _get_window("hann", nperseg)
    win_sum_sq = np.sum(win ** 2)
    
    # Segment the signal
    step = nperseg - noverlap
    n_segments = (len(signal) - noverlap) // step
    
    if n_segments < 1:
        n_segments = 1
        step = len(signal)
    
    # Compute time values (center of each window)
    times = np.arange(n_segments) * step + nperseg / 2
    times = times / sampling_rate
    
    # Compute FFT for each segment
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / sampling_rate)
    Sxx = np.zeros((len(frequencies), n_segments))
    
    for i in range(n_segments):
        start = i * step
        end = min(start + nperseg, len(signal))
        
        if end - start < nperseg:
            # Pad last segment if needed
            segment = np.zeros(nperseg)
            segment[: end - start] = signal[start:end]
        else:
            segment = signal[start:end]
        
        windowed = segment * win
        fft_vals = np.fft.rfft(windowed)
        power = (np.abs(fft_vals) ** 2) / win_sum_sq
        Sxx[:, i] = power
    
    # Convert to dB
    Sxx = 10.0 * np.log10(np.maximum(Sxx, 1e-10))
    
    return times, frequencies, Sxx


def find_dominant_frequencies(
    frequencies: np.ndarray, psd: np.ndarray, n_peaks: int = 5
) -> List[Tuple[float, float]]:
    """
    Find dominant frequency peaks in PSD.
    
    Identifies peaks in power spectral density and returns
    them sorted by power (descending).
    
    Args:
        frequencies: 1D array of frequency values in Hz.
        psd: 1D array of power spectral density values (linear or dB).
        n_peaks: Maximum number of peaks to return (default: 5).
    
    Returns:
        List of (frequency, power) tuples sorted by power (highest first).
    """
    if len(frequencies) == 0 or len(psd) == 0:
        return []
    
    # Find peaks using simple method: local maxima
    peaks = []
    
    for i in range(1, len(psd) - 1):
        if psd[i] > psd[i - 1] and psd[i] > psd[i + 1]:
            peaks.append((frequencies[i], psd[i]))
    
    # Sort by power (descending)
    peaks.sort(key=lambda x: x[1], reverse=True)
    
    return peaks[:n_peaks]
