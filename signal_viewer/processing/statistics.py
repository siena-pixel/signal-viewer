"""Statistical analysis of time series data."""

from typing import Dict, List
import numpy as np


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
            - 'counts': List of counts for each bin
            - 'bin_edges': List of bin edges
            - 'total_cycles': Total number of full cycles
            - 'total_half_cycles': Total number of half cycles
    """
    if len(signal) == 0:
        return {
            "ranges": [],
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
            "counts": [],
            "bin_edges": [],
            "total_cycles": 0,
            "total_half_cycles": 0,
        }

    # Step 2: Apply 4-point rainflow algorithm
    full_cycles, half_cycles = _rainflow_4point(turning_points)

    # Compute ranges (cycle amplitude * 2)
    all_ranges = []
    all_ranges.extend([abs(c[1] - c[0]) for c in full_cycles])  # Full cycles
    all_ranges.extend([abs(c[1] - c[0]) for c in half_cycles])  # Half cycles

    if len(all_ranges) == 0:
        return {
            "ranges": [],
            "counts": [],
            "bin_edges": [],
            "total_cycles": 0,
            "total_half_cycles": 0,
        }

    # Step 3: Bin the ranges
    all_ranges_array = np.array(all_ranges)
    bin_edges, counts = np.histogram(all_ranges_array, bins=n_bins)

    # Step 4: Return results
    return {
        "ranges": all_ranges,
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "total_cycles": len(full_cycles),
        "total_half_cycles": len(half_cycles),
    }


def _extract_turning_points(signal: np.ndarray) -> np.ndarray:
    """
    Extract local peaks and valleys (turning points) from signal.

    Args:
        signal: 1D array of signal values.

    Returns:
        Array of turning point values (peaks and valleys only).
    """
    if len(signal) < 3:
        return signal

    turning_points = [signal[0]]

    for i in range(1, len(signal) - 1):
        # Check if local maximum or minimum
        is_peak = (signal[i] > signal[i - 1] and signal[i] > signal[i + 1]) or \
                  (signal[i] < signal[i - 1] and signal[i] < signal[i + 1])

        if is_peak:
            turning_points.append(signal[i])

    turning_points.append(signal[-1])

    return np.array(turning_points)


def _rainflow_4point(turning_points: np.ndarray) -> tuple:
    """
    Apply 4-point rainflow cycle counting algorithm.

    Extracts full and half cycles from a sequence of turning points.

    Algorithm:
    - Use a stack to track the history of points
    - For each new point, check if the inner two points form a complete cycle
    - If yes, extract the cycle and remove those points
    - Continue until no more complete cycles can be formed
    - Remaining points are partial (half) cycles

    Args:
        turning_points: Array of local peaks and valleys.

    Returns:
        Tuple of (full_cycles, half_cycles) where each cycle is (start, end).
    """
    points = list(turning_points)
    full_cycles = []

    # Process complete cycles
    while len(points) >= 4:
        # Check if the range of points[1:3] is smaller than points[0:4]
        inner_range = abs(points[2] - points[1])
        outer_range = abs(points[3] - points[0])

        if inner_range <= outer_range:
            # Extract a full cycle from points[1] to points[2]
            full_cycles.append((points[1], points[2]))
            # Remove the cycled points
            points.pop(2)
            points.pop(1)
        else:
            # Move to the next set of points
            points.pop(0)

    # Remaining points form half cycles
    half_cycles = []
    for i in range(len(points) - 1):
        half_cycles.append((points[i], points[i + 1]))

    return full_cycles, half_cycles
