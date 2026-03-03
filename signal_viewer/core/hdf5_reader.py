"""
HDF5 Reader Module for Engineering Time Series Viewer

Provides robust HDF5 file reading with lazy loading, caching support, and
comprehensive metadata handling. Includes a MockHDF5File class for testing
without h5py dependency.

HDF5 internal layout (per configured group):
  /GROUP_NAME/
      GROUP_NAME_V   float64 [num_signals, num_samples]  (values)
      GROUP_NAME_T   float64 [num_samples]               (time)
      GROUP_NAME_P   float64 [num_samples]               (positions)
      GROUP_NAME_N   str     [num_signals]               (signal names)
      GROUP_NAME_U   str     [num_signals]               (units)

Dataset names are built as: group_name + suffix (e.g., "GROUP_T0_V").
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


class MockHDF5File:
    """
    Mock HDF5 file class that mimics h5py.File using numpy arrays.
    Useful for testing without h5py dependency.
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
        num_samples = 1000

        for group_name in S.GROUP_NAMES:
            self._groups[group_name] = {
                S.ds(group_name, S.VALUE_SUFFIX): np.random.randn(num_signals, num_samples).astype(np.float64),
                S.ds(group_name, S.TIME_SUFFIX): np.linspace(0, 10, num_samples).astype(np.float64),
                S.ds(group_name, S.POSITION_SUFFIX): np.linspace(0, 10, num_samples).astype(np.float64),
                S.ds(group_name, S.UNITS_SUFFIX): np.array(["m", "m/s", "A", "V"], dtype=object),
                S.ds(group_name, S.NAMES_SUFFIX): np.array(["position", "velocity", "current", "voltage"], dtype=object),
            }

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


def create_test_file(file_path: str) -> None:
    """
    Create a test HDF5 file with sample data using the configured schema.

    Args:
        file_path: Path where test file should be created
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    if HAS_H5PY:
        try:
            with h5py.File(file_path, "w") as f:
                num_signals = 4
                num_samples = 1000

                for group_name in S.GROUP_NAMES:
                    grp = f.create_group(group_name)
                    grp.create_dataset(
                        S.ds(group_name, S.VALUE_SUFFIX),
                        data=np.random.randn(num_signals, num_samples).astype(np.float64),
                    )
                    grp.create_dataset(
                        S.ds(group_name, S.TIME_SUFFIX),
                        data=np.linspace(0, 10, num_samples).astype(np.float64),
                    )
                    grp.create_dataset(
                        S.ds(group_name, S.POSITION_SUFFIX),
                        data=np.linspace(0, 10, num_samples).astype(np.float64),
                    )
                    grp.create_dataset(
                        S.ds(group_name, S.UNITS_SUFFIX),
                        data=np.array(["m", "m/s", "A", "V"], dtype=object),
                    )
                    grp.create_dataset(
                        S.ds(group_name, S.NAMES_SUFFIX),
                        data=np.array(["position", "velocity", "current", "voltage"], dtype=object),
                    )
        except Exception as e:
            raise IOError(f"Failed to create HDF5 test file at {file_path}: {e}") from e
    else:
        mock_file = MockHDF5File(file_path, mode="w")
        mock_file._load_mock_data()


class HDF5Reader:
    """
    Robust HDF5 file reader with lazy loading and metadata caching.

    Reads only the groups listed in HDF5Schema.GROUP_NAMES.
    Dataset names are built from group_name + suffix.
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
                # Return only configured groups that exist in the file
                self._groups_cache = [g for g in S.GROUP_NAMES if g in file_keys]

        return self._groups_cache

    # Keep backward-compatible alias
    get_batches = get_groups

    def get_group_metadata(self, group_name: str) -> Dict:
        """
        Get metadata for a group without loading signal data.

        Args:
            group_name: Name of the HDF5 group

        Returns:
            Dictionary with:
              - signal_count: int
              - sample_count: int
              - signal_names: List[str]
              - units: List[str]

        Raises:
            ValueError: If group does not exist or is missing required datasets
        """
        if group_name in self._metadata_cache:
            return self._metadata_cache[group_name]

        ds_values = S.ds(group_name, S.VALUE_SUFFIX)
        ds_names = S.ds(group_name, S.NAMES_SUFFIX)
        ds_units = S.ds(group_name, S.UNITS_SUFFIX)

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

                metadata = {
                    "signal_count": signal_count,
                    "sample_count": sample_count,
                    "signal_names": signal_names,
                    "units": units,
                }

                self._metadata_cache[group_name] = metadata

        return metadata

    # Keep backward-compatible alias
    get_batch_metadata = get_group_metadata

    def load_signal(
        self,
        group_name: str,
        signal_index: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single signal row with optional slicing.

        Args:
            group_name: Name of the HDF5 group
            signal_index: Index of signal (0-based)
            start: Start sample index (inclusive)
            end: End sample index (exclusive), None for all

        Returns:
            Tuple of (time_array, signal_values)

        Raises:
            ValueError: If group or signal index invalid
            IndexError: If indices out of range
        """
        metadata = self.get_group_metadata(group_name)

        if signal_index < 0 or signal_index >= metadata["signal_count"]:
            raise IndexError(
                f"Signal index {signal_index} out of range "
                f"[0, {metadata['signal_count']})"
            )

        ds_values = S.ds(group_name, S.VALUE_SUFFIX)
        ds_time = S.ds(group_name, S.TIME_SUFFIX)

        with self._lock:
            with self._open_file() as f:
                group = f[group_name]

                # Resolve end index
                if end is None:
                    end = metadata["sample_count"]
                elif end > metadata["sample_count"]:
                    raise IndexError(
                        f"end ({end}) exceeds sample count ({metadata['sample_count']})"
                    )

                # Load time array (may be per-signal or shared)
                time_array = group[ds_time][:]
                if len(time_array.shape) == 2:
                    time_data = time_array[signal_index, start:end]
                else:
                    time_data = time_array[start:end]

                # Load single signal row (lazy slice)
                signal_data = group[ds_values][signal_index, start:end]

        return time_data.astype(np.float64), signal_data.astype(np.float64)

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

    def get_signal_stats(self, group_name: str, signal_index: int) -> Dict:
        """
        Get statistical summary of a signal without loading all data.

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

        with self._lock:
            with self._open_file() as f:
                group = f[group_name]
                signal_data = group[ds_values][signal_index, :]

        return {
            "mean": float(np.mean(signal_data)),
            "std": float(np.std(signal_data)),
            "min": float(np.min(signal_data)),
            "max": float(np.max(signal_data)),
            "median": float(np.median(signal_data)),
            "samples": len(signal_data),
        }

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
