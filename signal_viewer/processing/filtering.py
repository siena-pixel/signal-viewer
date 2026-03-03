"""Digital filtering algorithms for time series data."""

from typing import Literal
import numpy as np


def _butterworth_coeffs(
    sampling_rate: float, cutoff_hz: float, order: int, filter_type: str
) -> tuple:
    """
    Compute Butterworth filter coefficients using frequency warping.

    Args:
        sampling_rate: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order.
        filter_type: 'lowpass' or 'highpass'.

    Returns:
        Tuple of (b_coeffs, a_coeffs) for IIR filter.
    """
    # Normalized frequency
    wc = 2.0 * np.tan(np.pi * cutoff_hz / sampling_rate)

    # Butterworth poles (analog domain)
    poles = []
    for k in range(order):
        angle = np.pi * (2 * k + order + 1) / (2 * order)
        poles.append(wc * np.exp(1j * angle))

    # Convert to digital domain using bilinear transform
    b = np.array([1.0 + 0j])
    a = np.array([1.0 + 0j])

    for pole in poles:
        # Bilinear transform: s = 2(z-1)/(z+1)
        # H(z) = (1 + z^-1) / (1 + (s/pole))
        denom_coeff = 2.0 + pole

        b_new = np.array([1.0, 1.0]) / denom_coeff
        a_new = np.array([1.0, -(2.0 - pole) / denom_coeff])

        # Cascade with previous stage
        b = np.convolve(b, b_new)
        a = np.convolve(a, a_new)

    if filter_type == "highpass":
        # Transform lowpass to highpass using spectral inversion
        # Reverse coefficients and alternate signs
        b = b[::-1] * ((-1) ** np.arange(len(b)))
        a = a[::-1] * ((-1) ** np.arange(len(a)))

    # Convert back to real coefficients (imaginary parts should be negligible)
    b = np.real(b)
    a = np.real(a)

    return b, a


def _apply_filter_forward_backward(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    Apply IIR filter forward and backward to eliminate phase shift.

    Equivalent to filtfilt: forward filtering, then reverse and filter again.

    Args:
        signal: Input signal.
        b: Numerator coefficients.
        a: Denominator coefficients.

    Returns:
        Filtered signal.
    """
    # Use frequency-domain approach for numerical stability:
    # Apply transfer function H(z) in frequency domain via FFT.
    n = len(signal)
    freqs = np.fft.rfftfreq(n)
    z = np.exp(2j * np.pi * freqs)

    # Evaluate H(z) = B(z) / A(z) on the unit circle
    numerator = np.zeros_like(z)
    for k, bk in enumerate(b):
        numerator += bk * z ** (-k)
    denominator = np.zeros_like(z)
    for k, ak in enumerate(a):
        denominator += ak * z ** (-k)

    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)
    h = numerator / denominator

    # Forward-backward: magnitude squared, zero phase
    h_fb = np.abs(h) ** 2

    # Apply in frequency domain
    sig_fft = np.fft.rfft(signal)
    filtered_fft = sig_fft * h_fb
    result = np.fft.irfft(filtered_fft, n=n)

    return result


def butterworth_lowpass(
    signal: np.ndarray, sampling_rate: float, cutoff_hz: float, order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth lowpass filter with zero phase shift.
    
    Uses forward-backward filtering to eliminate phase distortion.
    
    Args:
        signal: 1D array of signal values.
        sampling_rate: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency in Hz (-3dB point).
        order: Filter order (default: 4).
    
    Returns:
        Filtered signal (same length as input).
    
    Raises:
        ValueError: If cutoff_hz >= sampling_rate/2.
    """
    if len(signal) == 0:
        return signal.copy()
    
    if cutoff_hz >= sampling_rate / 2.0:
        raise ValueError("cutoff_hz must be less than sampling_rate/2 (Nyquist frequency)")
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    b, a = _butterworth_coeffs(sampling_rate, cutoff_hz, order, "lowpass")
    return _apply_filter_forward_backward(signal_clean, b, a)


def butterworth_highpass(
    signal: np.ndarray, sampling_rate: float, cutoff_hz: float, order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth highpass filter with zero phase shift.
    
    Uses forward-backward filtering to eliminate phase distortion.
    
    Args:
        signal: 1D array of signal values.
        sampling_rate: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency in Hz (-3dB point).
        order: Filter order (default: 4).
    
    Returns:
        Filtered signal (same length as input).
    
    Raises:
        ValueError: If cutoff_hz >= sampling_rate/2.
    """
    if len(signal) == 0:
        return signal.copy()
    
    if cutoff_hz >= sampling_rate / 2.0:
        raise ValueError("cutoff_hz must be less than sampling_rate/2 (Nyquist frequency)")
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    b, a = _butterworth_coeffs(sampling_rate, cutoff_hz, order, "highpass")
    return _apply_filter_forward_backward(signal_clean, b, a)


def butterworth_bandpass(
    signal: np.ndarray,
    sampling_rate: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter with zero phase shift.
    
    Cascades highpass and lowpass filters.
    
    Args:
        signal: 1D array of signal values.
        sampling_rate: Sampling frequency in Hz.
        low_hz: Lower cutoff frequency in Hz.
        high_hz: Upper cutoff frequency in Hz.
        order: Filter order for each stage (default: 4).
    
    Returns:
        Filtered signal (same length as input).
    
    Raises:
        ValueError: If frequencies are invalid.
    """
    if len(signal) == 0:
        return signal.copy()
    
    if low_hz >= high_hz:
        raise ValueError("low_hz must be less than high_hz")
    
    if high_hz >= sampling_rate / 2.0:
        raise ValueError("high_hz must be less than sampling_rate/2 (Nyquist frequency)")
    
    # Apply highpass then lowpass
    filtered = butterworth_highpass(signal, sampling_rate, low_hz, order)
    filtered = butterworth_lowpass(filtered, sampling_rate, high_hz, order)
    
    return filtered


def moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average filter.
    
    Simple smoothing filter that averages each window.
    Edge handling: centered window with padding.
    
    Args:
        signal: 1D array of signal values.
        window_size: Size of the moving window.
    
    Returns:
        Smoothed signal (same length as input).
    """
    if len(signal) == 0:
        return signal.copy()
    
    if window_size < 1:
        return signal.copy()
    
    if window_size == 1:
        return signal.copy()
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    # Pad signal at edges
    pad_size = window_size // 2
    padded = np.pad(signal_clean, (pad_size, pad_size), mode="edge")
    
    # Convolve with uniform kernel
    kernel = np.ones(window_size) / window_size
    filtered = np.convolve(padded, kernel, mode="valid")
    
    return filtered[: len(signal)]


def savitzky_golay(
    signal: np.ndarray, window_size: int = 11, order: int = 3
) -> np.ndarray:
    """
    Apply Savitzky-Golay filter using least-squares polynomial fitting.
    
    Fits low-degree polynomial in sliding window. Better than moving average
    at preserving signal features.
    
    Args:
        signal: 1D array of signal values.
        window_size: Size of the smoothing window (should be odd).
        order: Polynomial order (default: 3).
    
    Returns:
        Smoothed signal (same length as input).
    """
    if len(signal) == 0:
        return signal.copy()
    
    if window_size < order + 1:
        raise ValueError("window_size must be > order")
    
    if window_size % 2 == 0:
        window_size += 1
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    half_win = window_size // 2
    
    filtered = np.zeros_like(signal_clean)
    
    for i in range(len(signal_clean)):
        # Window boundaries with edge handling
        start = max(0, i - half_win)
        end = min(len(signal_clean), i + half_win + 1)
        
        # Adjust for edges
        if start == 0:
            end = min(len(signal_clean), window_size)
        if end == len(signal_clean):
            start = max(0, len(signal_clean) - window_size)
        
        x = np.arange(start, end) - i
        y = signal_clean[start:end]
        
        # Fit polynomial
        coeffs = np.polyfit(x, y, order)
        filtered[i] = np.polyval(coeffs, 0)
    
    return filtered


def detrend(signal: np.ndarray, method: str = "linear", order: int = 1) -> np.ndarray:
    """
    Remove trend from signal.
    
    Args:
        signal: 1D array of signal values.
        method: 'linear' (degree 1), 'constant' (mean), or 'polynomial'.
        order: Polynomial degree for 'polynomial' method.
    
    Returns:
        Detrended signal (same length as input).
    """
    if len(signal) == 0:
        return signal.copy()
    
    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    if method == "constant":
        return signal_clean - np.mean(signal_clean)
    
    elif method == "linear":
        x = np.arange(len(signal_clean))
        coeffs = np.polyfit(x, signal_clean, 1)
        trend = np.polyval(coeffs, x)
        return signal_clean - trend
    
    elif method == "polynomial":
        x = np.arange(len(signal_clean))
        coeffs = np.polyfit(x, signal_clean, order)
        trend = np.polyval(coeffs, x)
        return signal_clean - trend
    
    else:
        raise ValueError(f"Unknown detrend method: {method}")
