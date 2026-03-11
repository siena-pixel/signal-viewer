"""Statistical analysis of time series data."""

from typing import Dict, List
import ctypes
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load native C rainflow library (optional — falls back to pure Python)
# ---------------------------------------------------------------------------
_c_lib = None
try:
    _so_path = os.path.join(os.path.dirname(__file__), '_rainflow.so')
    if os.path.exists(_so_path):
        _c_lib = ctypes.CDLL(_so_path)
        _c_lib.rainflow_4point.restype = None
        _c_lib.rainflow_4point.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,          # tp, n
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),  # full_r, full_mx
            ctypes.POINTER(ctypes.c_double),                                    # full_mn
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),  # half_r, half_mx
            ctypes.POINTER(ctypes.c_double),                                    # half_mn
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),        # out_fc, out_hc
        ]
        logger.info('Loaded native rainflow accelerator: %s', _so_path)
except Exception as exc:
    logger.debug('Could not load native rainflow library: %s', exc)
    logger.info('Native rainflow accelerator not available — using pure-Python fallback')
    _c_lib = None


def compute_descriptive_stats(signal: np.ndarray) -> Dict[str, float]:
    """
    Compute descriptive statistics of a signal.
    
    Includes mean, std, min, max, median, RMS, peak-to-peak,
    skewness, kurtosis, IQR, and quartiles.
    
    Args:
        signal: 1D array of signal values.
    
    Returns:
        Dictionary with statistical measures.
    """
    if len(signal) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "rms": np.nan,
            "peak_to_peak": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "iqr": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }
    
    signal_clean = signal[~np.isnan(signal)]
    
    if len(signal_clean) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "rms": np.nan,
            "peak_to_peak": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "iqr": np.nan,
            "q25": np.nan,
            "q75": np.nan,
        }
    
    mean = np.mean(signal_clean)
    std = np.std(signal_clean, ddof=1) if len(signal_clean) > 1 else 0.0
    
    # Skewness: E[(X - mean)^3] / std^3
    if std > 0:
        skewness = np.mean((signal_clean - mean) ** 3) / (std ** 3)
    else:
        skewness = 0.0
    
    # Kurtosis: E[(X - mean)^4] / std^4 - 3 (excess kurtosis)
    if std > 0:
        kurtosis = np.mean((signal_clean - mean) ** 4) / (std ** 4) - 3.0
    else:
        kurtosis = 0.0
    
    q25 = np.percentile(signal_clean, 25)
    q75 = np.percentile(signal_clean, 75)
    
    return {
        "count": len(signal_clean),
        "mean": float(mean),
        "std": float(std),
        "min": float(np.min(signal_clean)),
        "max": float(np.max(signal_clean)),
        "median": float(np.median(signal_clean)),
        "rms": float(np.sqrt(np.mean(signal_clean ** 2))),
        "peak_to_peak": float(np.max(signal_clean) - np.min(signal_clean)),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "iqr": float(q75 - q25),
        "q25": float(q25),
        "q75": float(q75),
    }


def compute_rolling_stats(signal: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
    """
    Compute rolling statistics with a sliding window.

    Uses efficient cumsum-based approach for mean and min/max.

    Args:
        signal: 1D array of signal values.
        window_size: Size of the rolling window.

    Returns:
        Dictionary with rolling_mean, rolling_std, rolling_min, rolling_max arrays.
    """
    if len(signal) == 0:
        return {
            "rolling_mean": np.array([]),
            "rolling_std": np.array([]),
            "rolling_min": np.array([]),
            "rolling_max": np.array([]),
        }

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")

    signal_clean = np.where(np.isnan(signal), 0.0, signal)
    
    n = len(signal_clean)
    half_win = window_size // 2
    
    rolling_mean = np.zeros(n)
    rolling_std = np.zeros(n)
    rolling_min = np.zeros(n)
    rolling_max = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - half_win)
        end = min(n, i + half_win + 1)
        
        window = signal_clean[start:end]
        rolling_mean[i] = np.mean(window)
        rolling_std[i] = np.std(window)
        rolling_min[i] = np.min(window)
        rolling_max[i] = np.max(window)
    
    return {
        "rolling_mean": rolling_mean,
        "rolling_std": rolling_std,
        "rolling_min": rolling_min,
        "rolling_max": rolling_max,
    }


def compute_histogram(signal: np.ndarray, n_bins: int = 50) -> tuple:
    """
    Compute histogram of signal values.

    Args:
        signal: 1D array of signal values.
        n_bins: Number of bins (default: 50).

    Returns:
        Tuple of (bin_edges, counts) arrays.
        bin_edges: Array of bin edges (length n_bins + 1).
        counts: Array of counts in each bin (length n_bins).
    """
    if len(signal) == 0:
        return np.array([]), np.array([])

    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    signal_clean = signal[~np.isnan(signal)]
    
    if len(signal_clean) == 0:
        return np.array([]), np.array([])
    
    counts, bin_edges = np.histogram(signal_clean, bins=n_bins)
    
    return bin_edges, counts


def compute_percentiles(
    signal: np.ndarray, percentiles: List[float] = None
) -> Dict[str, float]:
    """
    Compute percentiles of signal.

    Args:
        signal: 1D array of signal values.
        percentiles: List of percentile values (0-100). Default: [1, 5, 25, 50, 75, 95, 99].

    Returns:
        Dictionary with percentile values.
    """
    if percentiles is None:
        percentiles = [1, 5, 25, 50, 75, 95, 99]

    if len(signal) == 0:
        return {f"p{p}": np.nan for p in percentiles}

    signal_clean = signal[~np.isnan(signal)]

    if len(signal_clean) == 0:
        return {f"p{p}": np.nan for p in percentiles}

    result = {}
    for p in percentiles:
        result[f"p{p}"] = float(np.percentile(signal_clean, p))

    return result


def compute_rainflow(signal: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute rainflow cycle counting for fatigue analysis.

    Implements the 4-point rainflow cycle counting algorithm to extract
    stress/strain cycles from a signal. Returns a histogram of cycle ranges
    (amplitudes) binned into n_bins bins.

    Algorithm:
    1. Extract turning points (local peaks and valleys) from the signal
    2. Apply 4-point rainflow method:
       - Process consecutive points checking if inner range <= outer range
       - Extract full cycles when condition is met
       - Continue until no more cycles can be extracted
       - Remaining points contribute to half-cycles
    3. Bin the ranges (cycle amplitudes * 2) into n_bins bins
    4. Return histogram with statistics

    Args:
        signal: 1D array of signal values.
        n_bins: Number of bins for the histogram (default: 10).

    Returns:
        Dictionary with keys:
            - 'ranges': List of cycle ranges (amplitude * 2)
            - 'cycle_maxs': List of max value per cycle
            - 'cycle_mins': List of min value per cycle
            - 'counts': List of counts for each bin
            - 'bin_edges': List of bin edges
            - 'total_cycles': Total number of full cycles
            - 'total_half_cycles': Total number of half cycles
    """
    if len(signal) == 0:
        return {
            "ranges": [],
            "cycle_maxs": [],
            "cycle_mins": [],
            "counts": [],
            "bin_edges": [],
            "total_cycles": 0,
            "total_half_cycles": 0,
        }

    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins}")

    signal_clean = signal[~np.isnan(signal)]

    if len(signal_clean) < 2:
        return {
            "ranges": [],
            "cycle_maxs": [],
            "cycle_mins": [],
            "counts": [],
            "bin_edges": [],
            "total_cycles": 0,
            "total_half_cycles": 0,
        }

    # Step 1: Extract turning points (local peaks and valleys)
    turning_points = _extract_turning_points(signal_clean)

    if len(turning_points) < 2:
        return {
            "ranges": [],
            "cycle_maxs": [],
            "cycle_mins": [],
            "counts": [],
            "bin_edges": [],
            "total_cycles": 0,
            "total_half_cycles": 0,
        }

    # Step 2: Apply 4-point rainflow algorithm (returns numpy arrays)
    full_ranges, half_ranges, full_maxs, full_mins, half_maxs, half_mins = \
        _rainflow_4point(turning_points)

    n_full = len(full_ranges)
    n_half = len(half_ranges)

    if n_full + n_half == 0:
        return {
            "ranges": [],
            "cycle_maxs": [],
            "cycle_mins": [],
            "counts": [],
            "bin_edges": [],
            "total_cycles": 0,
            "total_half_cycles": 0,
        }

    # Step 3: Concatenate and bin (all numpy, no Python-list overhead)
    all_ranges = np.concatenate((full_ranges, half_ranges))
    all_maxs = np.concatenate((full_maxs, half_maxs))
    all_mins = np.concatenate((full_mins, half_mins))
    counts, bin_edges = np.histogram(all_ranges, bins=n_bins)

    # Step 4: Return results (tolist() converts numpy → plain Python for JSON)
    return {
        "ranges": all_ranges.tolist(),
        "cycle_maxs": all_maxs.tolist(),
        "cycle_mins": all_mins.tolist(),
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "total_cycles": n_full,
        "total_half_cycles": n_half,
    }


def _extract_turning_points(signal: np.ndarray) -> np.ndarray:
    """
    Extract local peaks and valleys (turning points) from signal.

    Fully vectorized — no Python loops.

    Args:
        signal: 1D array of signal values.

    Returns:
        Array of turning point values (peaks and valleys only).
    """
    if len(signal) < 3:
        return signal.copy()

    # Differences: positive = rising, negative = falling
    d = np.diff(signal)

    # A turning point is where the sign of the difference changes.
    # sign(d[i-1]) != sign(d[i]) — but we must ignore flat segments (d==0).
    # Replace zeros with the last non-zero sign so plateaus are absorbed.
    signs = np.sign(d)
    # Forward-fill zeros: replace 0 with previous non-zero value
    nz = signs != 0
    if nz.any():
        idx = np.where(nz, np.arange(len(signs)), 0)
        np.maximum.accumulate(idx, out=idx)
        signs = signs[idx]

    # Turning point where consecutive signs differ
    tp_mask = np.empty(len(signal), dtype=np.bool_)
    tp_mask[0] = True
    tp_mask[-1] = True
    tp_mask[1:-1] = signs[:-1] != signs[1:]

    return signal[tp_mask].copy()


def _rainflow_4point(turning_points: np.ndarray) -> tuple:
    """
    Stack-based 4-point rainflow cycle counting — O(n) amortised.

    Dispatches to compiled C when the native library is available
    (typically 30-80× faster than pure Python).  Falls back to an
    optimised pure-Python implementation otherwise.

    Args:
        turning_points: Array of local peaks and valleys (float64).

    Returns:
        Tuple of (full_ranges, half_ranges, full_maxs, full_mins,
                  half_maxs, half_mins).
        Each element is a numpy float64 array.
    """
    tp = np.ascontiguousarray(turning_points, dtype=np.float64)
    n = len(tp)

    if _c_lib is not None:
        return _rainflow_4point_c(tp, n)
    return _rainflow_4point_py(tp, n)


def _rainflow_4point_c(tp: np.ndarray, n: int) -> tuple:
    """C-accelerated 4-point rainflow via ctypes."""
    # Pre-allocate output arrays (worst case: n elements each)
    full_r  = np.empty(n, dtype=np.float64)
    full_mx = np.empty(n, dtype=np.float64)
    full_mn = np.empty(n, dtype=np.float64)
    half_r  = np.empty(n, dtype=np.float64)
    half_mx = np.empty(n, dtype=np.float64)
    half_mn = np.empty(n, dtype=np.float64)

    out_fc = ctypes.c_int(0)
    out_hc = ctypes.c_int(0)

    _c_lib.rainflow_4point(
        tp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        full_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        full_mx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        full_mn.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        half_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        half_mx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        half_mn.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.byref(out_fc),
        ctypes.byref(out_hc),
    )

    fc = out_fc.value
    hc = out_hc.value
    return (full_r[:fc], half_r[:hc],
            full_mx[:fc], full_mn[:fc],
            half_mx[:hc], half_mn[:hc])


def _rainflow_4point_py(tp: np.ndarray, n: int) -> tuple:
    """Pure-Python fallback with pre-allocated numpy arrays."""
    stack   = np.empty(n, dtype=np.intp)
    full_r  = np.empty(n, dtype=np.float64)
    full_mx = np.empty(n, dtype=np.float64)
    full_mn = np.empty(n, dtype=np.float64)

    sp = 0   # stack pointer
    fc = 0   # full cycle count
    _abs = abs

    for i in range(n):
        stack[sp] = i
        sp += 1

        while sp >= 4:
            v2 = tp[stack[sp - 2]]
            v3 = tp[stack[sp - 3]]
            inner = _abs(v2 - v3)
            outer = _abs(tp[stack[sp - 1]] - tp[stack[sp - 4]])

            if inner <= outer:
                full_r[fc] = inner
                if v2 >= v3:
                    full_mx[fc] = v2
                    full_mn[fc] = v3
                else:
                    full_mx[fc] = v3
                    full_mn[fc] = v2
                fc += 1
                stack[sp - 3] = stack[sp - 1]
                sp -= 2
            else:
                break

    hc = max(0, sp - 1)
    half_r  = np.empty(hc, dtype=np.float64)
    half_mx = np.empty(hc, dtype=np.float64)
    half_mn = np.empty(hc, dtype=np.float64)

    for j in range(hc):
        a = tp[stack[j]]
        b = tp[stack[j + 1]]
        half_r[j] = _abs(b - a)
        if a >= b:
            half_mx[j] = a
            half_mn[j] = b
        else:
            half_mx[j] = b
            half_mn[j] = a

    return (full_r[:fc], half_r, full_mx[:fc], full_mn[:fc],
            half_mx, half_mn)
