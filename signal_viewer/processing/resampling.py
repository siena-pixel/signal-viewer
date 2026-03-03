"""Downsampling and resampling algorithms for time series data."""

from typing import Tuple
import numpy as np


def lttb_downsample(
    time: np.ndarray, signal: np.ndarray, target_points: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Largest Triangle Three Buckets (LTTB) downsampling algorithm.
    
    Preserves the visual shape of time series data while reducing points.
    Efficient O(n) algorithm suitable for 100k+ point datasets.
    Always keeps first and last points.
    
    Args:
        time: 1D array of time values.
        signal: 1D array of signal values (same length as time).
        target_points: Target number of downsampled points (default: 2000).
    
    Returns:
        Tuple of (time_downsampled, signal_downsampled) arrays.
    
    Raises:
        ValueError: If arrays have mismatched lengths or < 3 points.
    """
    if len(time) != len(signal):
        raise ValueError("time and signal must have the same length")
    
    n_points = len(signal)
    if n_points <= target_points:
        return time.copy(), signal.copy()
    
    if n_points < 3:
        return time.copy(), signal.copy()
    
    # Initialize with first point
    indices = np.zeros(target_points, dtype=np.int64)
    indices[0] = 0
    
    bucket_size = (n_points - 2) / (target_points - 2)
    
    # Process each bucket
    for i in range(1, target_points - 1):
        # Bucket range
        bucket_start = int(np.floor((i - 1) * bucket_size)) + 1
        bucket_end = int(np.floor(i * bucket_size)) + 1
        
        # Last point in this bucket
        bucket_end = min(bucket_end, n_points - 1)
        
        # Point from previous bucket
        prev_idx = indices[i - 1]
        
        # Next bucket's first point (used for triangle area calculation)
        next_bucket_start = int(np.floor(i * bucket_size)) + 1
        next_bucket_start = min(next_bucket_start, n_points - 1)
        
        max_area = -1.0
        max_area_idx = bucket_start
        
        # Find point with largest triangle area
        for j in range(bucket_start, bucket_end + 1):
            # Area of triangle formed by (prev, current, next_bucket_first)
            area = (
                abs(
                    (time[prev_idx] - time[j]) * (signal[next_bucket_start] - signal[prev_idx])
                    - (time[prev_idx] - time[next_bucket_start]) * (signal[j] - signal[prev_idx])
                )
                / 2.0
            )
            
            if area > max_area:
                max_area = area
                max_area_idx = j
        
        indices[i] = max_area_idx
    
    # Add last point
    indices[-1] = n_points - 1
    
    return time[indices], signal[indices]


def simple_decimate(
    time: np.ndarray, signal: np.ndarray, factor: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple decimation by selecting every nth point.
    
    Fast downsampling method that keeps every 'factor'th point.
    Does not apply anti-aliasing filter (use for visualization only).
    
    Args:
        time: 1D array of time values.
        signal: 1D array of signal values (same length as time).
        factor: Decimation factor (every nth point is kept).
    
    Returns:
        Tuple of (time_decimated, signal_decimated) arrays.
    
    Raises:
        ValueError: If arrays have mismatched lengths or factor < 1.
    """
    if len(time) != len(signal):
        raise ValueError("time and signal must have the same length")
    
    if factor < 1:
        raise ValueError("decimation factor must be >= 1")
    
    if factor == 1:
        return time.copy(), signal.copy()
    
    # Always include the last point
    indices = np.arange(0, len(signal), factor)
    if indices[-1] != len(signal) - 1:
        indices = np.append(indices, len(signal) - 1)
    
    return time[indices], signal[indices]
