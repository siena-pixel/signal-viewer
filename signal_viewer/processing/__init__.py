"""Signal processing modules for engineering time series viewer."""

from .resampling import lttb_downsample, simple_decimate
from .spectral import (
    compute_fft,
    compute_psd_welch,
    compute_spectrogram,
    find_dominant_frequencies,
)
from .filtering import (
    butterworth_lowpass,
    butterworth_highpass,
    butterworth_bandpass,
    moving_average,
    savitzky_golay,
    detrend,
)
from .statistics import (
    compute_descriptive_stats,
    compute_rolling_stats,
    compute_histogram,
    compute_percentiles,
)
from .anomaly import (
    zscore_anomaly,
    mad_anomaly,
    derivative_anomaly,
    iqr_anomaly,
    rolling_anomaly,
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
    # Spectral
    "compute_fft",
    "compute_psd_welch",
    "compute_spectrogram",
    "find_dominant_frequencies",
    # Filtering
    "butterworth_lowpass",
    "butterworth_highpass",
    "butterworth_bandpass",
    "moving_average",
    "savitzky_golay",
    "detrend",
    # Statistics
    "compute_descriptive_stats",
    "compute_rolling_stats",
    "compute_histogram",
    "compute_percentiles",
    # Anomaly
    "zscore_anomaly",
    "mad_anomaly",
    "derivative_anomaly",
    "iqr_anomaly",
    "rolling_anomaly",
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
