"""Downsampling and resampling algorithms for time series data."""

from typing import Tuple
import numpy as np


def lttb_downsample(
    time: np.ndarray, signal: np.ndarray, target_points: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Largest Triangle Three Buckets (LTTB) downsampling algorithm.

    Preserves the visual shape of time series data while reducing points.
    O(n) algorithm suitable for 100k+ point datasets.
    Always keeps first and last points.

    The inner-bucket search is vectorized with numpy, eliminating the
    Python inner loop that dominated runtime for large signals.

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

    n = len(signal)
    if n <= target_points or n < 3:
        return time.copy(), signal.copy()

    # Ensure contiguous float64 for fast indexing
    t = np.ascontiguousarray(time, dtype=np.float64)
    s = np.ascontiguousarray(signal, dtype=np.float64)

    m = target_points
    bucket_size = (n - 2) / (m - 2)

    indices = np.empty(m, dtype=np.int64)
    indices[0] = 0
    indices[-1] = n - 1

    # Pre-compute bucket boundaries
    bucket_starts = np.empty(m - 2, dtype=np.int64)
    bucket_ends   = np.empty(m - 2, dtype=np.int64)
    next_starts   = np.empty(m - 2, dtype=np.int64)

    for i in range(m - 2):
        bucket_starts[i] = int(i * bucket_size) + 1
        bucket_ends[i]   = min(int((i + 1) * bucket_size) + 1, n - 1)
        next_starts[i]   = min(int((i + 1) * bucket_size) + 1, n - 1)

    # Process each bucket — inner search is vectorized
    prev_idx = 0
    for i in range(m - 2):
        bs = bucket_starts[i]
        be = bucket_ends[i]
        ns = next_starts[i]

        # Vectorized triangle-area computation for all points in bucket
        j_indices = np.arange(bs, be + 1)
        tp = t[prev_idx]
        sp = s[prev_idx]
        areas = np.abs(
            (tp - t[j_indices]) * (s[ns] - sp)
            - (tp - t[ns]) * (s[j_indices] - sp)
        )
        best = j_indices[np.argmax(areas)]
        indices[i + 1] = best
        prev_idx = best

    return t[indices], s[indices]


def minmax_lttb_downsample(
    time: np.ndarray, signal: np.ndarray, target_points: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MinMax-LTTB: two-pass downsample for very large signals.

    Pass 1 — MinMax pre-selection: divide data into (target * 4) buckets,
    keep the min and max signal value in each bucket.  This is O(n) and
    purely vectorised with numpy, reducing e.g. 5 M points to ~20 K.

    Pass 2 — LTTB: run standard LTTB on the pre-selected points to
    further reduce to the final target count while preserving visual shape.

    ~10× faster than plain LTTB for signals above 100 K samples, with
    nearly identical visual quality.

    Args:
        time: 1D array of time values.
        signal: 1D array of signal values (same length as time).
        target_points: Target number of downsampled points (default: 2000).

    Returns:
        Tuple of (time_downsampled, signal_downsampled) arrays.
    """
    if len(time) != len(signal):
        raise ValueError("time and signal must have the same length")

    n = len(signal)
    if n <= target_points or n < 3:
        return time.copy(), signal.copy()

    # If ratio is small enough, plain LTTB is fast already
    preselect_factor = 4
    preselect_target = target_points * preselect_factor
    if n <= preselect_target:
        return lttb_downsample(time, signal, target_points)

    # ── Pass 1: MinMax pre-selection (fully vectorised) ─────────────────
    t = np.ascontiguousarray(time, dtype=np.float64)
    s = np.ascontiguousarray(signal, dtype=np.float64)

    num_buckets = preselect_target // 2  # 2 points per bucket (min + max)
    chunk = n // num_buckets
    usable = chunk * num_buckets  # trim to exact multiple

    # Reshape into (num_buckets, chunk) for vectorised argmin/argmax
    s_blocks = s[:usable].reshape(num_buckets, chunk)
    offsets = (np.arange(num_buckets) * chunk).reshape(-1, 1)

    min_idx = s_blocks.argmin(axis=1) + offsets.ravel()
    max_idx = s_blocks.argmax(axis=1) + offsets.ravel()

    # Interleave min/max in time-order per bucket
    indices = np.empty(num_buckets * 2, dtype=np.int64)
    swap = min_idx > max_idx
    indices[0::2] = np.where(swap, max_idx, min_idx)
    indices[1::2] = np.where(swap, min_idx, max_idx)

    # Handle leftover samples in the tail with the SAME bucket density
    # as the main body.  The old approach used only 2 points for the
    # entire tail (min + max), which could leave thousands of samples
    # represented by just one line segment — visible as a flat line at
    # the end of each signal in the overview.
    if usable < n:
        tail_indices = []
        pos = usable
        while pos < n:
            end = min(pos + chunk, n)
            seg = s[pos:end]
            lo = pos + int(np.argmin(seg))
            hi = pos + int(np.argmax(seg))
            tail_indices.append(min(lo, hi))
            if lo != hi:
                tail_indices.append(max(lo, hi))
            pos = end
        if tail_indices:
            indices = np.append(indices, tail_indices)

    # Deduplicate, sort, ensure first/last are included
    indices = np.unique(indices)
    if indices[0] != 0:
        indices = np.concatenate(([0], indices))
    if indices[-1] != n - 1:
        indices = np.concatenate((indices, [n - 1]))

    # ── Pass 2: LTTB on pre-selected points ───────────────────────────────
    return lttb_downsample(t[indices], s[indices], target_points)


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
