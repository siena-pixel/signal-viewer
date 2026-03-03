"""Signal processing modules for engineering time series viewer."""

from .resampling import lttb_downsample, simple_decimate
from .statistics import (
    compute_descriptive_stats,
    compute_rolling_stats,
    compute_histogram,
    compute_percentiles,
    compute_rainflow,
)
from .correlation import (
    cross_correlate,
    compute_coherence,
    find_lag,
    compute_correlation_matrix,
)
from .trend import (
    fit_polynomial,
    remove_trend,
    detect_changepoints,
    compute_envelope,
    compute_rms_trend,
)

__all__ = [
    # Resampling
    "lttb_downsample",
    "simple_decimate",
    # Statistics
    "compute_descriptive_stats",
    "compute_rolling_stats",
    "compute_histogram",
    "compute_percentiles",
    "compute_rainflow",
    # Correlation
    "cross_correlate",
    "compute_coherence",
    "find_lag",
    "compute_correlation_matrix",
    # Trend
    "fit_polynomial",
    "remove_trend",
    "detect_changepoints",
    "compute_envelope",
    "compute_rms_trend",
]
