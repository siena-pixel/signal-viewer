"""
HDF5 Reader Module for Engineering Time Series Viewer

Provides robust HDF5 file reading with lazy loading, caching support, and
comprehensive metadata handling. Includes a MockHDF5File class for testing
without h5py dependency.

HDF5 internal layout (per configured group):
  /GROUP_NAME/
      GROUP_NAME_V    float64 [n_signals, max(n_samples)]  signal values
      GROUP_NAME_TIM  float64 [n_signals]  epoch time (ms) of first sample
      GROUP_NAME_FRE  float64 [n_signals]  sampling frequency (Hz)
      GROUP_NAME_SAM  int64   [n_signals]  valid sample count per signal
      GROUP_NAME_N    str     [n_signals]  signal names
      GROUP_NAME_UNI  str     [n_signals]  units

  Type B groups additionally contain:
      GROUP_NAME_ERR  float64 [n_signals, max(n_samples)]  signal errors
      GROUP_NAME_SQI  float64 [n_signals]  signal quality metric
      GROUP_NAME_TLS  float64 [n_signals]  max error gap (seconds)

Batch types:
  Type A – only base datasets (no _ERR, _SQI, _TLS)
  Type B – base datasets + _ERR, _SQI, _TLS

Both types have _SAM (valid sample count per signal).

Time construction (both types):
  time = TIM[idx] + arange(length) * (1000.0 / FRE[idx])
  where TIM is epoch time in milliseconds and FRE is in Hz.

Dataset names are built as: group_name + suffix (e.g. "GROUP_T0_V").
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..config import HDF5Schema as S

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Mock classes (testing without h5py)
# ---------------------------------------------------------------------------

class MockHDF5File:
    """
    Mock HDF5 file class that mimics h5py.File using numpy arrays.
    Useful for testing without h5py dependency.

    GROUP_T0 is Type A (no _ERR/_SQI/_TLS datasets).
    GROUP_T1 is Type B (has _ERR/_SQI/_TLS datasets).
    Both groups have _SAM.
    """

    def __init__(self, file_path: str, mode: str = "r"):
        self.file_path = file_path
        self.mode = mode
        self._groups: Dict[str, Dict[str, np.ndarray]] = {}
        self._closed = False

        if mode == "r" and Path(file_path).exists():
            self._load_mock_data()

    def _load_mock_data(self) -> None:
        """Load mock data using the configured schema."""
        num_signals = 4
        max_samples = 1000
        rng = np.random.RandomState(42)

        names = np.array(["position", "velocity", "current", "voltage"], dtype=object)
        units = np.array(["m", "m/s", "A", "V"], dtype=object)
        # Epoch time in ms (all same starting epoch)
        epoch_ms = np.array([1700000000000.0] * num_signals, dtype=np.float64)
        samp_freqs = np.array([100.0, 200.0, 100.0, 500.0], dtype=np.float64)
        nsample = np.array([800, 600, 900, 750], dtype=np.int64)

        for group_name in S.GROUP_NAMES:
            values = rng.randn(num_signals, max_samples).astype(np.float64)
            datasets = {
                S.ds(group_name, S.VALUE_SUFFIX): values,
                S.ds(group_name, S.TIME_SUFFIX): epoch_ms.copy(),
                S.ds(group_name, S.SAMPLING_FREQ_SUFFIX): samp_freqs.copy(),
                S.ds(group_name, S.NSAMPLE_SUFFIX): nsample.copy(),
                S.ds(group_name, S.NAMES_SUFFIX): names.copy(),
                S.ds(group_name, S.UNITS_SUFFIX): units.copy(),
            }
            # Second group onward is Type B (add _ERR, _SQI, _TLS)
            if group_name != S.GROUP_NAMES[0]:
                errors = rng.randn(num_signals, max_samples).astype(np.float64) * 0.1
                sqi = rng.uniform(0.8, 1.0, size=num_signals).astype(np.float64)
                tls = rng.uniform(0.0, 5.0, size=num_signals).astype(np.float64)
                datasets[S.ds(group_name, S.ERROR_SUFFIX)] = errors
                datasets[S.ds(group_name, S.SQI_SUFFIX)] = sqi
                datasets[S.ds(group_name, S.TLS_SUFFIX)] = tls

            self._groups[group_name] = datasets

    def keys(self) -> List[str]:
        """Get group names."""
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return list(self._groups.keys())

    def __getitem__(self, key: str) -> "MockHDF5Group":
        """Get a group by name."""
        if self._closed:
            raise ValueError("I/O operation on closed file")
        if key not in self._groups:
            raise KeyError(f"Group '{key}' not found")
        return MockHDF5Group(self._groups[key])

    def __contains__(self, key: str) -> bool:
        """Check if group exists."""
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return key in self._groups

    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MockHDF5Group:
    """Mock HDF5 group to access datasets."""

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        self._data = data_dict

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._data:
            raise KeyError(f"Dataset '{key}' not found")
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# Test file creation
# ---------------------------------------------------------------------------

def create_test_file(file_path: str) -> None:
    """
    Create a test HDF5 file with sample data using the configured schema.

    GROUP_T0 is created as Type A (no _ERR/_SQI/_TLS).
    GROUP_T1 is created as Type B (with _ERR/_SQI/_TLS).
    Both groups have _SAM.

    Args:
        file_path: Path where test file should be created
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    num_signals = 4
    max_samples = 1000
    rng = np.random.RandomState(42)

    names = np.array(["position", "velocity", "current", "voltage"], dtype=object)
    units = np.array(["m", "m/s", "A", "V"], dtype=object)
    epoch_ms = np.array([1700000000000.0] * num_signals, dtype=np.float64)
    samp_freqs = np.array([100.0, 200.0, 100.0, 500.0], dtype=np.float64)
    nsample = np.array([800, 600, 900, 750], dtype=np.int64)

    if HAS_H5PY:
        try:
            with h5py.File(file_path, "w") as f:
                for group_name in S.GROUP_NAMES:
                    grp = f.create_group(group_name)
                    values = rng.randn(num_signals, max_samples).astype(np.float64)
                    grp.create_dataset(S.ds(group_name, S.VALUE_SUFFIX), data=values)
                    grp.create_dataset(S.ds(group_name, S.TIME_SUFFIX), data=epoch_ms)
                    grp.create_dataset(S.ds(group_name, S.SAMPLING_FREQ_SUFFIX), data=samp_freqs)
                    grp.create_dataset(S.ds(group_name, S.NSAMPLE_SUFFIX), data=nsample)
                    grp.create_dataset(S.ds(group_name, S.NAMES_SUFFIX), data=names)
                    grp.create_dataset(S.ds(group_name, S.UNITS_SUFFIX), data=units)
                    # Type B (second group onward): add _ERR, _SQI, _TLS
                    if group_name != S.GROUP_NAMES[0]:
                        errors = rng.randn(num_signals, max_samples).astype(np.float64) * 0.1
                        sqi = rng.uniform(0.8, 1.0, size=num_signals).astype(np.float64)
                        tls = rng.uniform(0.0, 5.0, size=num_signals).astype(np.float64)
                        grp.create_dataset(S.ds(group_name, S.ERROR_SUFFIX), data=errors)
                        grp.create_dataset(S.ds(group_name, S.SQI_SUFFIX), data=sqi)
                        grp.create_dataset(S.ds(group_name, S.TLS_SUFFIX), data=tls)
        except Exception as e:
            raise IOError(f"Failed to create HDF5 test file at {file_path}: {e}") from e
    else:
        # No h5py: create a marker file so HDF5Reader falls back to MockHDF5File
        Path(file_path).touch()


# ---------------------------------------------------------------------------
# HDF5Reader
# ---------------------------------------------------------------------------

class HDF5Reader:
    """
    Robust HDF5 file reader with lazy loading and metadata caching.

    Reads only the groups listed in HDF5Schema.GROUP_NAMES.
    Dataset names are built from group_name + suffix.

    Supports two batch types:
      Type A – no _ERR dataset → base datasets only
      Type B – _ERR dataset present → base + error/quality datasets
    Both types have _SAM (valid sample count per signal).
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        self._lock = threading.Lock()
        self._file_handle = None
        self._groups_cache: Optional[List[str]] = None
        self._metadata_cache: Dict[str, Dict] = {}

        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that file is a readable HDF5 file."""
        try:
            with self._open_file() as f:
                if len(f.keys()) == 0:
                    raise ValueError("HDF5 file contains no groups")
        except Exception as e:
            raise ValueError(f"Invalid HDF5 file: {self.file_path}") from e

    def _open_file(self):
        if HAS_H5PY:
            return h5py.File(self.file_path, "r")
        else:
            return MockHDF5File(str(self.file_path), "r")

    # ------------------------------------------------------------------
    # Group discovery
    # ------------------------------------------------------------------

    def get_groups(self) -> List[str]:
        """
        Get list of configured group names that exist in the file.

        Returns only groups from HDF5Schema.GROUP_NAMES that are
        actually present in the HDF5 file.

        Returns:
            Sorted list of group names
        """
        if self._groups_cache is not None:
            return self._groups_cache

        with self._lock:
            with self._open_file() as f:
                file_keys = list(f.keys())
                self._groups_cache = [g for g in S.GROUP_NAMES if g in file_keys]

        return self._groups_cache

    # Backward-compatible alias
    get_batches = get_groups

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_group_metadata(self, group_name: str) -> Dict:
        """
        Get metadata for a group without loading signal data.

        Args:
            group_name: Name of the HDF5 group

        Returns:
            Dictionary with:
              - signal_count: int
              - sample_count: int  (max column width of value matrix)
              - signal_names: List[str]
              - units: List[str]
              - batch_type: str  ("A" or "B")
              - n_samples: List[int]  (per-signal valid counts)
              For Type B additionally:
              - sqi: List[float]  (signal quality index)
              - tls: List[float]  (max error gap in seconds)

        Raises:
            ValueError: If group does not exist or is missing required datasets
        """
        if group_name in self._metadata_cache:
            return self._metadata_cache[group_name]

        ds_values = S.ds(group_name, S.VALUE_SUFFIX)
        ds_names = S.ds(group_name, S.NAMES_SUFFIX)
        ds_units = S.ds(group_name, S.UNITS_SUFFIX)
        ds_nsample = S.ds(group_name, S.NSAMPLE_SUFFIX)
        ds_err = S.ds(group_name, S.ERROR_SUFFIX)
        ds_sqi = S.ds(group_name, S.SQI_SUFFIX)
        ds_tls = S.ds(group_name, S.TLS_SUFFIX)

        with self._lock:
            with self._open_file() as f:
                if group_name not in f:
                    raise ValueError(f"Group '{group_name}' not found")

                group = f[group_name]

                if ds_values not in group or ds_names not in group:
                    raise ValueError(
                        f"Group '{group_name}' missing required datasets "
                        f"(expected '{ds_values}' and '{ds_names}')"
                    )

                signal_count = group[ds_values].shape[0]
                sample_count = group[ds_values].shape[1]

                signal_names = [
                    n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    for n in group[ds_names][:]
                ]
                units = [
                    u.decode("utf-8") if isinstance(u, bytes) else str(u)
                    for u in group[ds_units][:]
                ] if ds_units in group else [""] * signal_count

                # Both types have _SAM
                if ds_nsample in group:
                    n_samples = [int(v) for v in group[ds_nsample][:]]
                else:
                    n_samples = [sample_count] * signal_count

                # Type B detection: presence of _ERR dataset
                is_type_b = ds_err in group

                metadata = {
                    "signal_count": signal_count,
                    "sample_count": sample_count,
                    "signal_names": signal_names,
                    "units": units,
                    "batch_type": "B" if is_type_b else "A",
                    "n_samples": n_samples,
                }

                # Type B extra fields
                if is_type_b:
                    if ds_sqi in group:
                        metadata["sqi"] = [float(v) for v in group[ds_sqi][:]]
                    if ds_tls in group:
                        metadata["tls"] = [float(v) for v in group[ds_tls][:]]

                self._metadata_cache[group_name] = metadata

        return metadata

    # Backward-compatible alias
    get_batch_metadata = get_group_metadata

    # ------------------------------------------------------------------
    # Signal loading
    # ------------------------------------------------------------------

    def load_signal(
        self,
        group_name: str,
        signal_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single signal with constructed time vector.

        Both Type A and Type B use _SAM for valid sample count:
            signal = values[idx, :n_samples[idx]]

        Time is always constructed:
            t0 = TIM[idx]   (epoch ms)
            fs = FRE[idx]   (Hz)
            time = t0 + arange(length) * (1000.0 / fs)

        Args:
            group_name: Name of the HDF5 group
            signal_index: Index of signal (0-based)

        Returns:
            Tuple of (time_array, signal_values)

        Raises:
            ValueError: If group or signal index invalid
            IndexError: If signal index out of range
        """
        metadata = self.get_group_metadata(group_name)

        if signal_index < 0 or signal_index >= metadata["signal_count"]:
            raise IndexError(
                f"Signal index {signal_index} out of range "
                f"[0, {metadata['signal_count']})"
            )

        ds_values = S.ds(group_name, S.VALUE_SUFFIX)
        ds_time = S.ds(group_name, S.TIME_SUFFIX)
        ds_fs = S.ds(group_name, S.SAMPLING_FREQ_SUFFIX)

        # Get valid sample count from metadata (already read from _SAM)
        n = metadata["n_samples"][signal_index]

        with self._lock:
            with self._open_file() as f:
                group = f[group_name]

                # Read signal values (only valid portion)
                signal_data = group[ds_values][signal_index, :n].astype(np.float64)

                # Read per-signal time parameters
                t0 = float(group[ds_time][signal_index])
                fs = float(group[ds_fs][signal_index])

        # Construct time vector (t0 is epoch ms, fs is Hz → step = 1000/fs ms)
        time_data = t0 + np.arange(n, dtype=np.float64) * (1000.0 / fs)

        return time_data, signal_data

    def load_signal_by_name(
        self, group_name: str, signal_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load signal by name instead of index.

        Args:
            group_name: Name of the HDF5 group
            signal_name: Name of the signal

        Returns:
            Tuple of (time_array, signal_values)

        Raises:
            ValueError: If signal name not found
        """
        metadata = self.get_group_metadata(group_name)

        try:
            signal_index = metadata["signal_names"].index(signal_name)
        except ValueError as e:
            raise ValueError(
                f"Signal '{signal_name}' not found in group '{group_name}'"
            ) from e

        return self.load_signal(group_name, signal_index)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_signal_stats(self, group_name: str, signal_index: int) -> Dict:
        """
        Get statistical summary of a signal (respects n_sample truncation).

        Args:
            group_name: Name of the HDF5 group
            signal_index: Index of signal

        Returns:
            Dictionary with mean, std, min, max, median, samples
        """
        metadata = self.get_group_metadata(group_name)

        if signal_index < 0 or signal_index >= metadata["signal_count"]:
            raise IndexError(f"Signal index {signal_index} out of range")

        ds_values = S.ds(group_name, S.VALUE_SUFFIX)

        # Get valid sample count from metadata
        n = metadata["n_samples"][signal_index]

        with self._lock:
            with self._open_file() as f:
                group = f[group_name]
                signal_data = group[ds_values][signal_index, :n]

        return {
            "mean": float(np.mean(signal_data)),
            "std": float(np.std(signal_data)),
            "min": float(np.min(signal_data)),
            "max": float(np.max(signal_data)),
            "median": float(np.median(signal_data)),
            "samples": n,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the file handle and cleanup resources."""
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return f"HDF5Reader(file={self.file_path.name}, groups={len(self.get_groups())})"
