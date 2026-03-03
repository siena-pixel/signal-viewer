"""Trend analysis and changepoint detection for time series data."""

from typing import Tuple
import numpy as np


def fit_polynomial(
    time: np.ndarray, signal: np.ndarray, degree: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit polynomial trend to time series.
    
    Args:
        time: 1D array of time values.
        signal: 1D array of signal values (same length as time).
        degree: Polynomial degree (default: 1 for linear).
    
    Returns:
        Tuple of (coefficients, fitted_values, residuals).
        coefficients: Polynomial coefficients (highest degree first).
        fitted_values: Fitted signal values.
        residuals: signal - fitted_values.
    """
    if len(time) == 0 or len(signal) == 0:
        return np.array([]), np.array([]), np.array([])
    
    if len(time) != len(signal):
        raise ValueError("time and signal must have the same length")
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    # Fit polynomial
    coeffs = np.polyfit(time, signal_clean, degree)
    fitted = np.polyval(coeffs, time)
    residuals = signal_clean - fitted
    
    return coeffs, fitted, residuals


def remove_trend(
    time: np.ndarray, signal: np.ndarray, degree: int = 1
) -> np.ndarray:
    """
    Remove polynomial trend from time series.
    
    Args:
        time: 1D array of time values.
        signal: 1D array of signal values (same length as time).
        degree: Polynomial degree (default: 1 for linear).
    
    Returns:
        Detrended signal (trend removed).
    """
    if len(time) == 0 or len(signal) == 0:
        return signal.copy()
    
    coeffs, fitted, residuals = fit_polynomial(time, signal, degree)
    return residuals


def detect_changepoints(
    signal: np.ndarray, window_size: int = 100, threshold: float = 3.0
) -> np.ndarray:
    """
    Detect changepoints using CUSUM-like approach.
    
    Computes running mean and flags points where there are significant shifts
    in the local mean value.
    
    Args:
        signal: 1D array of signal values.
        window_size: Size of rolling window for mean calculation (default: 100).
        threshold: Z-score threshold for mean shifts (default: 3.0).
    
    Returns:
        Array of changepoint indices.
    """
    if len(signal) < window_size:
        return np.array([], dtype=np.int64)
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    half_win = window_size // 2
    
    rolling_mean = np.zeros_like(signal_clean)
    rolling_std = np.zeros_like(signal_clean)
    
    # Compute rolling statistics
    for i in range(len(signal_clean)):
        start = max(0, i - half_win)
        end = min(len(signal_clean), i + half_win + 1)
        window = signal_clean[start:end]
        rolling_mean[i] = np.mean(window)
        rolling_std[i] = np.std(window)
    
    # Compute Z-score of mean values
    mean_of_means = np.mean(rolling_mean)
    std_of_means = np.std(rolling_mean)
    
    if std_of_means == 0:
        return np.array([], dtype=np.int64)
    
    mean_z_scores = np.abs((rolling_mean - mean_of_means) / std_of_means)
    
    # Detect changepoints as peaks in Z-scores
    changepoints = np.where(mean_z_scores > threshold)[0]
    
    # Remove consecutive changepoints (keep only the first one)
    if len(changepoints) > 0:
        gaps = np.diff(changepoints)
        distinct_points = [changepoints[0]]
        for i in range(len(gaps)):
            if gaps[i] > window_size // 2:
                distinct_points.append(changepoints[i + 1])
        changepoints = np.array(distinct_points, dtype=np.int64)
    
    return changepoints


def compute_envelope(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute upper and lower envelope of signal.
    
    Finds local extrema and interpolates to get smooth envelope.
    
    Args:
        signal: 1D array of signal values.
    
    Returns:
        Tuple of (upper_envelope, lower_envelope) arrays.
    """
    if len(signal) < 3:
        return signal.copy(), signal.copy()
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    # Find local maxima (upper envelope)
    upper_indices = [0]
    for i in range(1, len(signal_clean) - 1):
        if signal_clean[i] > signal_clean[i - 1] and signal_clean[i] > signal_clean[i + 1]:
            upper_indices.append(i)
    upper_indices.append(len(signal_clean) - 1)
    
    # Find local minima (lower envelope)
    lower_indices = [0]
    for i in range(1, len(signal_clean) - 1):
        if signal_clean[i] < signal_clean[i - 1] and signal_clean[i] < signal_clean[i + 1]:
            lower_indices.append(i)
    lower_indices.append(len(signal_clean) - 1)
    
    # Ensure we have at least 2 points for interpolation
    if len(upper_indices) < 2:
        upper_indices = [0, len(signal_clean) - 1]
    if len(lower_indices) < 2:
        lower_indices = [0, len(signal_clean) - 1]
    
    # Interpolate envelopes
    upper_envelope = np.interp(
        np.arange(len(signal_clean)),
        upper_indices,
        signal_clean[upper_indices],
    )
    lower_envelope = np.interp(
        np.arange(len(signal_clean)),
        lower_indices,
        signal_clean[lower_indices],
    )
    
    return upper_envelope, lower_envelope


def compute_rms_trend(signal: np.ndarray, window_size: int = 1000) -> np.ndarray:
    """
    Compute running RMS trend of signal.
    
    Useful for monitoring signal energy over time.
    RMS = sqrt(mean(signal^2)) in each window.
    
    Args:
        signal: 1D array of signal values.
        window_size: Size of rolling window (default: 1000).
    
    Returns:
        Array of RMS values (one per window position).
    """
    if len(signal) == 0:
        return np.array([])
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    if len(signal_clean) < window_size:
        # If signal is shorter than window, return single RMS
        rms = np.sqrt(np.mean(signal_clean ** 2))
        return np.array([rms])
    
    rms_trend = np.zeros(len(signal_clean) - window_size + 1)
    
    for i in range(len(rms_trend)):
        window = signal_clean[i : i + window_size]
        rms_trend[i] = np.sqrt(np.mean(window ** 2))
    
    return rms_trend
