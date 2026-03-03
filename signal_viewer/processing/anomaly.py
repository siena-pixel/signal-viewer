"""Anomaly detection algorithms for time series data."""

from typing import Tuple
import numpy as np


def zscore_anomaly(signal: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using Z-score method.
    
    Points with |z-score| > threshold are flagged as anomalies.
    Sensitive to outliers.
    
    Args:
        signal: 1D array of signal values.
        threshold: Z-score threshold (default: 3.0 = ~99.7% in normal dist).
    
    Returns:
        Tuple of (anomaly_indices, z_scores) arrays.
        anomaly_indices: Indices of anomalous points.
        z_scores: Z-scores for all points.
    """
    if len(signal) == 0:
        return np.array([], dtype=np.int64), np.array([])
    
    signal_clean = signal[~np.isnan(signal)]
    if len(signal_clean) == 0:
        return np.array([], dtype=np.int64), np.array([])
    
    mean = np.mean(signal_clean)
    std = np.std(signal_clean)
    
    if std == 0:
        return np.array([], dtype=np.int64), np.zeros_like(signal)
    
    z_scores = np.abs((signal - mean) / std)
    anomaly_indices = np.where(z_scores > threshold)[0]
    
    return anomaly_indices, z_scores


def mad_anomaly(signal: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using Median Absolute Deviation (MAD).
    
    More robust to outliers than Z-score method.
    Modified Z-score = 0.6745 * (x - median) / MAD
    
    Args:
        signal: 1D array of signal values.
        threshold: Modified Z-score threshold (default: 3.5).
    
    Returns:
        Tuple of (anomaly_indices, modified_z_scores) arrays.
    """
    if len(signal) == 0:
        return np.array([], dtype=np.int64), np.array([])
    
    signal_clean = signal[~np.isnan(signal)]
    if len(signal_clean) == 0:
        return np.array([], dtype=np.int64), np.array([])
    
    median = np.median(signal_clean)
    mad = np.median(np.abs(signal_clean - median))
    
    if mad == 0:
        return np.array([], dtype=np.int64), np.zeros_like(signal)
    
    # Modified Z-score
    modified_z = 0.6745 * (signal - median) / mad
    anomaly_indices = np.where(np.abs(modified_z) > threshold)[0]
    
    return anomaly_indices, np.abs(modified_z)


def derivative_anomaly(signal: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies as sudden jumps using derivative analysis.
    
    Computes first derivative and flags points with large changes.
    Useful for detecting steps/spikes.
    
    Args:
        signal: 1D array of signal values.
        threshold: Z-score threshold on derivative (default: 3.0).
    
    Returns:
        Tuple of (anomaly_indices, derivative_z_scores) arrays.
    """
    if len(signal) < 2:
        return np.array([], dtype=np.int64), np.array([])
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    # Compute first derivative
    derivative = np.diff(signal_clean)
    
    # Compute Z-score of derivative
    deriv_mean = np.mean(derivative)
    deriv_std = np.std(derivative)
    
    if deriv_std == 0:
        return np.array([], dtype=np.int64), np.zeros(len(signal))
    
    deriv_z = np.abs((derivative - deriv_mean) / deriv_std)
    
    # Pad first element
    deriv_z_padded = np.concatenate([[deriv_z[0]], deriv_z])
    
    anomaly_indices = np.where(deriv_z_padded > threshold)[0]
    
    return anomaly_indices, deriv_z_padded


def iqr_anomaly(signal: np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using Interquartile Range (IQR) method.
    
    Points outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are anomalies.
    Standard multiplier = 1.5 for mild outliers, 3.0 for extreme.
    
    Args:
        signal: 1D array of signal values.
        multiplier: IQR multiplier (default: 1.5).
    
    Returns:
        Tuple of (anomaly_indices, signal_values) arrays.
    """
    if len(signal) == 0:
        return np.array([], dtype=np.int64), np.array([])
    
    signal_clean = signal[~np.isnan(signal)]
    if len(signal_clean) == 0:
        return np.array([], dtype=np.int64), np.array([])
    
    q1 = np.percentile(signal_clean, 25)
    q3 = np.percentile(signal_clean, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    anomaly_mask = (signal < lower_bound) | (signal > upper_bound)
    anomaly_indices = np.where(anomaly_mask)[0]
    
    return anomaly_indices, signal[anomaly_indices]


def rolling_anomaly(
    signal: np.ndarray, window_size: int = 100, threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect local anomalies using rolling window statistics.
    
    Flags points that are anomalous relative to their local neighborhood
    (instead of global statistics).
    
    Args:
        signal: 1D array of signal values.
        window_size: Size of rolling window (default: 100).
        threshold: Z-score threshold (default: 3.0).
    
    Returns:
        Tuple of (anomaly_indices, local_z_scores) arrays.
    """
    if len(signal) < window_size:
        return np.array([], dtype=np.int64), np.array([])
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    local_z_scores = np.zeros_like(signal_clean)
    half_win = window_size // 2
    
    for i in range(len(signal_clean)):
        start = max(0, i - half_win)
        end = min(len(signal_clean), i + half_win + 1)
        
        window = signal_clean[start:end]
        window_mean = np.mean(window)
        window_std = np.std(window)
        
        if window_std > 0:
            local_z_scores[i] = np.abs((signal_clean[i] - window_mean) / window_std)
        else:
            local_z_scores[i] = 0.0
    
    anomaly_indices = np.where(local_z_scores > threshold)[0]
    
    return anomaly_indices, local_z_scores
