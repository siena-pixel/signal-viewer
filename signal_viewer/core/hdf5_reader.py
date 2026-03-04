"""
HDF5 Reader Module for Engineering Time Series Viewer

Provides robust HDF5 file reading with lazy loading, caching support, and
comprehensive metadata handling. Includes a MockHDF5File class for testing
without h5py dependency.

HDF5 internal layout (per configured group):
  /GROUP_NAME/
      GROUP_NAME_V   float64 [num_signals, max_samples]  signal values
      GROUP_NAME_T   float64 [num_signals]               start time per signal
      GROUP_NAME_FS  float64 [num_signals]               sampling freq per signal
      GROUP_NAME_NS  int64   [num_signals]  (optional)   valid sample count
      GROUP_NAME_N   str     [num_signals]               signal names
      GROUP_NAME_U   str     [num_signals]               units

Batch types:
  Type A – _NS dataset present: each signal has n_sample[i] valid points.
  Type B – _NS dataset absent:  full value matrix is valid (rectangular).

Time construction (both types):
  time = t0 + arange(length) / fs
  where t0 = _T[signal_index], fs = _FS[signal_index]

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

    GROUP_T0 is Type A (has _NS dataset).
    GROUP_T1 is Type B (no _NS dataset).
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
        start_times = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
        samp_freqs = np.array([100.0, 200.0, 100.0, 500.0], dtype=np.float64)
        nsample_a = np.array([800, 600, 900, 750], dtype=np.int64)

        for group_name in S.GROUP_NAMES:
            values = rng.randn(num_signals, max_samples).astype(np.float64)
            datasets = {
                S.ds(group_name, S.VALUE_SUFFIX): values,
                S.ds(group_name, S.TIME_SUFFIX): start_times.copy(),
                S.ds(group_name, S.SAMPLING_FREQ_SUFFIX): samp_freqs.copy(),
                S.ds(group_name, S.NAMES_SUFFIX): names.copy(),
                S.ds(group_name, S.UNITS_SUFFIX): units.copy(),
            }
            # First group is Type A (has _NS); others are Type B
            if group_name == S.GROUP_NAMES[0]:
                datasets[S.ds(group_name, S.NSAMPLE_SUFFIX)] = nsample_a.copy()

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

    GROUP_T0 is created as Type A (with _NS dataset).
    GROUP_T1 is created as Type B (without _NS dataset).

    Args:
        file_path: Path where test file should be created
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    num_signals = 4
    max_samples = 1000
    rng = np.random.RandomState(42)

    names = np.array(["position", "velocity", "current", "voltage"], dtype=object)
    units = np.array(["m", "m/s", "A", "V"], dtype=object)
    start_times = np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float64)
    samp_freqs = np.array([100.0, 200.0, 100.0, 500.0], dtype=np.float64)
    nsample_a = np.array([800, 600, 900, 750], dtype=np.int64)

    if HAS_H5PY:
        try:
            with h5py.File(file_path, "w") as f:
                for group_name in S.GROUP_NAMES:
                    grp = f.create_group(group_name)
                    values = rng.randn(num_signals, max_samples).astype(np.float64)
                    grp.create_dataset(S.ds(group_name, S.VALUE_SUFFIX), data=values)
                    grp.create_dataset(S.ds(group_name, S.TIME_SUFFIX), data=start_times)
                    grp.create_dataset(S.ds(group_name, S.SAMPLING_FREQ_SUFFIX), data=samp_freqs)
                    grp.create_dataset(S.ds(group_name, S.NAMES_SUFFIX), data=names)
                    grp.create_dataset(S.ds(group_name, S.UNITS_SUFFIX), data=units)
                    # First group → Type A
                    if group_name == S.GROUP_NAMES[0]:
                        grp.create_dataset(S.ds(group_name, S.NSAMPLE_SUFFIX), data=nsample_a)
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
      Type A – _NS dataset present → per-signal valid sample counts
      Type B – _NS dataset absent  → full rectangular value matrix
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
              - n_samples: Optional[List[int]]  (per-signal counts for Type A)

        Raises:
            ValueError: If group does not exist or is missing required datasets
        """
        if group_name in self._metadata_cache:
            return self._metadata_cache[group_name]

        ds_values = S.ds(group_name, S.VALUE_SUFFIX)
        ds_names = S.ds(group_name, S.NAMES_SUFFIX)
        ds_units = S.ds(group_name, S.UNITS_SUFFIX)
        ds_nsample = S.ds(group_name, S.NSAMPLE_SUFFIX)

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

                # Determine batch type
                if ds_nsample in group:
                    batch_type = "A"
                    n_samples = [int(v) for v in group[ds_nsample][:]]
                else:
                    batch_type = "B"
                    n_samples = None

                metadata = {
                    "signal_count": signal_count,
                    "sample_count": sample_count,
                    "signal_names": signal_names,
                    "units": units,
                    "batch_type": batch_type,
                    "n_samples": n_samples,
                }

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

        Type A (n_sample dataset exists):
            signal = values[idx, :n_sample[idx]]
        Type B (no n_sample dataset):
            signal = values[idx, :]

        Time is always constructed:
            t0 = time_dataset[idx]
            fs = freq_dataset[idx]
            time = t0 + arange(length) / fs

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
        ds_nsample = S.ds(group_name, S.NSAMPLE_SUFFIX)

        with self._lock:
            with self._open_file() as f:
                group = f[group_name]

                # Determine valid sample count
                if ds_nsample in group:
                    n = int(group[ds_nsample][signal_index])
                else:
                    n = metadata["sample_count"]

                # Read signal values (only valid portion)
                signal_data = group[ds_values][signal_index, :n].astype(np.float64)

                # Read per-signal time parameters
                t0 = float(group[ds_time][signal_index])
                fs = float(group[ds_fs][signal_index])

        # Construct time vector
        time_data = t0 + np.arange(n, dtype=np.float64) / fs

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
        ds_nsample = S.ds(group_name, S.NSAMPLE_SUFFIX)

        with self._lock:
            with self._open_file() as f:
                group = f[group_name]

                # Determine valid length
                if ds_nsample in group:
                    n = int(group[ds_nsample][signal_index])
                else:
                    n = metadata["sample_count"]

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
