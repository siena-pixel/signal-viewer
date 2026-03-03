"""Correlation and coherence analysis for time series data."""

from typing import Dict, Literal, Tuple
import numpy as np


def cross_correlate(
    signal_a: np.ndarray, signal_b: np.ndarray, mode: str = "full"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-correlation between two signals.

    Uses FFT-based approach for efficiency.
    Result is normalized to [-1, 1].

    Args:
        signal_a: First signal array.
        signal_b: Second signal array.
        mode: 'full' (default), 'valid', or 'same' (as in np.correlate).

    Returns:
        Tuple of (lags, correlation) arrays.
        lags: Lag values (in samples).
        correlation: Normalized cross-correlation values in [-1, 1].
    """
    if len(signal_a) == 0 or len(signal_b) == 0:
        return np.array([]), np.array([])

    a_clean = np.where(np.isnan(signal_a), 0.0, signal_a)
    b_clean = np.where(np.isnan(signal_b), 0.0, signal_b)

    # Use numpy's correlate for accurate cross-correlation
    correlation_full = np.correlate(a_clean, b_clean, mode="full")

    # Normalize by signal lengths and standard deviations
    a_norm = a_clean - np.mean(a_clean)
    b_norm = b_clean - np.mean(b_clean)

    a_std = np.std(a_norm)
    b_std = np.std(b_norm)

    if a_std > 0 and b_std > 0:
        # Normalize to [-1, 1] range
        normalization = a_std * b_std * len(a_clean)
        correlation_full = correlation_full / normalization
        # Clip to [-1, 1] due to numerical precision
        correlation_full = np.clip(correlation_full, -1.0, 1.0)
    else:
        correlation_full = np.zeros_like(correlation_full, dtype=float)

    if mode == "full":
        lags = np.arange(-(len(b_clean) - 1), len(a_clean))
        correlation = correlation_full
    elif mode == "same":
        # Return center portion matching len(a_clean)
        start_idx = len(b_clean) - 1 - len(a_clean) // 2
        end_idx = start_idx + len(a_clean)
        correlation = correlation_full[start_idx:end_idx]
        lags = np.arange(-(len(a_clean) // 2), len(a_clean) - len(a_clean) // 2)
    else:  # valid
        start = len(b_clean) - 1
        correlation = correlation_full[start : start + len(a_clean) - len(b_clean) + 1]
        lags = np.arange(0, len(a_clean) - len(b_clean) + 1)

    return lags, correlation


def compute_coherence(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    sampling_rate: float,
    nperseg: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude-squared coherence between two signals.
    
    Coherence = |Pxy|^2 / (Pxx * Pyy)
    where Pxy is cross-spectral density and Pxx, Pyy are auto-spectral densities.
    
    Uses Welch's method for PSD estimation.
    
    Args:
        signal_a: First signal array.
        signal_b: Second signal array.
        sampling_rate: Sampling frequency in Hz.
        nperseg: Length of each segment for Welch method (default: 1024).
    
    Returns:
        Tuple of (frequencies, coherence) arrays.
        frequencies: Frequency values in Hz.
        coherence: Magnitude-squared coherence in [0, 1].
    """
    if len(signal_a) == 0 or len(signal_b) == 0:
        return np.array([]), np.array([])
    
    from .spectral import compute_psd_welch
    
    a_clean = np.where(np.isnan(signal_a), 0.0, signal_a)
    b_clean = np.where(np.isnan(signal_b), 0.0, signal_b)
    
    # Compute auto-spectral densities
    freq_a, psd_a = compute_psd_welch(a_clean, sampling_rate, nperseg=nperseg)
    freq_b, psd_b = compute_psd_welch(b_clean, sampling_rate, nperseg=nperseg)
    
    # Ensure same frequency grid
    frequencies = freq_a
    psd_a = 10.0 ** (psd_a / 10.0)  # Convert from dB back to linear
    psd_b = 10.0 ** (psd_b / 10.0)
    
    # Compute cross-spectral density using Welch's method
    noverlap = nperseg // 2
    step = nperseg - noverlap
    n_segments = (len(a_clean) - noverlap) // step
    
    if n_segments < 1:
        n_segments = 1
        step = len(a_clean)
    
    csd_vals = []
    for i in range(n_segments):
        start = i * step
        end = min(start + nperseg, len(a_clean))
        
        if end - start < nperseg:
            seg_a = np.zeros(nperseg)
            seg_b = np.zeros(nperseg)
            seg_a[: end - start] = a_clean[start:end]
            seg_b[: end - start] = b_clean[start:end]
        else:
            seg_a = a_clean[start:end]
            seg_b = b_clean[start:end]
        
        win = np.hanning(nperseg)
        seg_a = seg_a * win
        seg_b = seg_b * win
        
        fft_a = np.fft.rfft(seg_a)
        fft_b = np.fft.rfft(seg_b)
        csd_segment = fft_a * np.conj(fft_b)
        csd_vals.append(csd_segment)
    
    csd = np.mean(csd_vals, axis=0)
    csd_mag_sq = np.abs(csd) ** 2
    
    # Coherence = |Cxy|^2 / (Cxx * Cyy)
    coherence = csd_mag_sq / (psd_a * psd_b + 1e-10)
    coherence = np.minimum(coherence, 1.0)
    coherence = np.maximum(coherence, 0.0)
    
    return frequencies, coherence


def find_lag(signal_a: np.ndarray, signal_b: np.ndarray) -> int:
    """
    Find the lag at which cross-correlation is maximum.
    
    Positive lag means signal_b lags signal_a.
    
    Args:
        signal_a: First signal array.
        signal_b: Second signal array.
    
    Returns:
        Lag (in samples) at maximum correlation.
    """
    if len(signal_a) == 0 or len(signal_b) == 0:
        return 0
    
    lags, correlation = cross_correlate(signal_a, signal_b, mode="full")
    
    max_idx = np.argmax(np.abs(correlation))
    return int(lags[max_idx])


def compute_correlation_matrix(signals_dict: Dict[str, np.ndarray]) -> Tuple[list, np.ndarray]:
    """
    Compute Pearson correlation matrix for multiple signals.
    
    Args:
        signals_dict: Dictionary of {name: signal_array} pairs.
    
    Returns:
        Tuple of (signal_names, correlation_matrix).
        signal_names: List of signal names.
        correlation_matrix: NxN correlation matrix (N = number of signals).
    """
    if len(signals_dict) == 0:
        return [], np.array([]).reshape(0, 0)
    
    names = list(signals_dict.keys())
    n_signals = len(names)
    
    correlation_matrix = np.zeros((n_signals, n_signals))
    
    for i, name_i in enumerate(names):
        sig_i = signals_dict[name_i]
        sig_i_clean = sig_i[~np.isnan(sig_i)]
        
        if len(sig_i_clean) == 0:
            continue
        
        for j, name_j in enumerate(names):
            sig_j = signals_dict[name_j]
            sig_j_clean = sig_j[~np.isnan(sig_j)]
            
            if len(sig_j_clean) == 0:
                correlation_matrix[i, j] = 0.0
                continue
            
            # Find overlapping indices
            min_len = min(len(sig_i_clean), len(sig_j_clean))
            sig_i_overlap = sig_i_clean[:min_len]
            sig_j_overlap = sig_j_clean[:min_len]
            
            # Pearson correlation
            cov = np.cov(sig_i_overlap, sig_j_overlap)
            std_i = np.std(sig_i_overlap)
            std_j = np.std(sig_j_overlap)
            
            if std_i > 0 and std_j > 0:
                correlation_matrix[i, j] = cov[0, 1] / (std_i * std_j)
            else:
                correlation_matrix[i, j] = 0.0
    
    return names, correlation_matrix
