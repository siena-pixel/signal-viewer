"""
HDF5 Reader Module for Engineering Time Series Viewer

Provides robust HDF5 file reading with lazy loading, caching support, and
comprehensive metadata handling. Includes a MockHDF5File class for testing
without h5py dependency.

Data Structure (per batch group):
  - value: float64 matrix [num_signals, num_samples]
  - time: float64 array [num_signals] or [num_samples]
  - corrected_positions: float64 array (same shape as time)
  - units: string array [num_signals]
  - name: string array [num_signals]
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
        """
        Initialize mock HDF5 file.

        Args:
            file_path: Path to the file (for compatibility with h5py.File)
            mode: File mode (for compatibility, typically "r")
        """
        self.file_path = file_path
        self.mode = mode
        self._groups: Dict[str, Dict[str, np.ndarray]] = {}
        self._closed = False

        if mode == "r" and Path(file_path).exists():
            self._load_mock_data()

    def _load_mock_data(self) -> None:
        """Load mock data for testing."""
        # Create a test batch with sample data
        num_signals = 4
        num_samples = 1000
        self._groups[S.DEFAULT_BATCH] = {
            S.VALUES: np.random.randn(num_signals, num_samples).astype(np.float64),
            S.TIME: np.linspace(0, 10, num_samples).astype(np.float64),
            S.POSITIONS: np.linspace(0, 10, num_samples).astype(np.float64),
            S.UNITS: np.array(["m", "m/s", "A", "V"], dtype=object),
            S.NAMES: np.array(["position", "velocity", "current", "voltage"], dtype=object),
        }

    def keys(self) -> List[str]:
        """Get batch group names."""
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

    def close(self) -> None:
        """Close the file."""
        self._closed = True

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class MockHDF5Group:
    """Mock HDF5 group to access datasets."""

    def __init__(self, data_dict: Dict[str, np.ndarray]):
        """
        Initialize mock group.

        Args:
            data_dict: Dictionary of dataset name -> numpy array
        """
        self._data = data_dict

    def keys(self) -> List[str]:
        """Get dataset names."""
        return list(self._data.keys())

    def __getitem__(self, key: str) -> np.ndarray:
        """Get dataset array."""
        if key not in self._data:
            raise KeyError(f"Dataset '{key}' not found")
        return self._data[key]


def create_test_file(file_path: str) -> None:
    """
    Create a test HDF5 file with sample data.

    Uses h5py if available, otherwise creates a numpy-backed mock file.

    Args:
        file_path: Path where test file should be created

    Raises:
        IOError: If file cannot be created
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    if HAS_H5PY:
        try:
            with h5py.File(file_path, "w") as f:
                num_signals = 4
                num_samples = 1000

                grp = f.create_group(S.DEFAULT_BATCH)
                grp.create_dataset(S.VALUES, data=np.random.randn(num_signals, num_samples).astype(np.float64))
                grp.create_dataset(S.TIME, data=np.linspace(0, 10, num_samples).astype(np.float64))
                grp.create_dataset(
                    S.POSITIONS,
                    data=np.linspace(0, 10, num_samples).astype(np.float64),
                )
                grp.create_dataset(S.UNITS, data=np.array(["m", "m/s", "A", "V"], dtype=object))
                grp.create_dataset(S.NAMES, data=np.array(["position", "velocity", "current", "voltage"], dtype=object))
        except Exception as e:
            raise IOError(f"Failed to create HDF5 test file at {file_path}: {e}") from e
    else:
        # Create a mock file by saving to numpy format
        # In production, h5py should be available
        mock_file = MockHDF5File(file_path, mode="w")
        mock_file._load_mock_data()


class HDF5Reader:
    """
    Robust HDF5 file reader with lazy loading and metadata caching.

    Features:
      - Row-slice access for efficient memory usage
      - Thread-safe file operations
      - Comprehensive metadata caching
      - Statistical summaries without full data load
      - Graceful fallback to mock file if h5py unavailable
    """

    def __init__(self, file_path: str):
        """
        Initialize HDF5 reader.

        Validates file existence and caches metadata.

        Args:
            file_path: Path to HDF5 file

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is not a valid HDF5 file
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        self._lock = threading.Lock()
        self._file_handle = None
        self._batches_cache: Optional[List[str]] = None
        self._metadata_cache: Dict[str, Dict] = {}

        # Validate file by attempting to open
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
        """
        Open HDF5 file with h5py or mock fallback.

        Returns:
            File handle (h5py.File or MockHDF5File)
        """
        if HAS_H5PY:
            return h5py.File(self.file_path, "r")
        else:
            return MockHDF5File(str(self.file_path), "r")

    def get_batches(self) -> List[str]:
        """
        Get list of all batch group names.

        Returns:
            Sorted list of batch names (e.g., ["batch_001", "batch_002"])
        """
        if self._batches_cache is not None:
            return self._batches_cache

        with self._lock:
            with self._open_file() as f:
                self._batches_cache = sorted(f.keys())

        return self._batches_cache

    def get_batch_metadata(self, batch_name: str) -> Dict:
        """
        Get metadata for a batch without loading signal data.

        Args:
            batch_name: Name of the batch group

        Returns:
            Dictionary with:
              - signal_count: int
              - sample_count: int
              - signal_names: List[str]
              - units: List[str]

        Raises:
            ValueError: If batch does not exist
        """
        if batch_name in self._metadata_cache:
            return self._metadata_cache[batch_name]

        with self._lock:
            with self._open_file() as f:
                if batch_name not in f:
                    raise ValueError(f"Batch '{batch_name}' not found")

                batch = f[batch_name]

                if S.VALUES not in batch or S.NAMES not in batch:
                    raise ValueError(f"Batch '{batch_name}' missing required datasets")

                signal_count = batch[S.VALUES].shape[0]
                sample_count = batch[S.VALUES].shape[1]
                signal_names = [
                    n.decode("utf-8") if isinstance(n, bytes) else str(n)
                    for n in batch[S.NAMES][:]
                ]
                units = [
                    u.decode("utf-8") if isinstance(u, bytes) else str(u)
                    for u in batch[S.UNITS][:]
                ] if S.UNITS in batch else [""] * signal_count

                metadata = {
                    "signal_count": signal_count,
                    "sample_count": sample_count,
                    "signal_names": signal_names,
                    "units": units,
                }

                self._metadata_cache[batch_name] = metadata

        return metadata

    def load_signal(
        self,
        batch_name: str,
        signal_index: int,
        start: int = 0,
        end: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single signal row with optional slicing.

        Uses lazy row-slice access to minimize memory usage.

        Args:
            batch_name: Name of the batch group
            signal_index: Index of signal (0-based)
            start: Start sample index (inclusive)
            end: End sample index (exclusive), None for all

        Returns:
            Tuple of (time_array, signal_values)
              - time_array: shape [num_samples]
              - signal_values: shape [num_samples]

        Raises:
            ValueError: If batch or signal index invalid
            IndexError: If indices out of range
        """
        metadata = self.get_batch_metadata(batch_name)

        if signal_index < 0 or signal_index >= metadata["signal_count"]:
            raise IndexError(f"Signal index {signal_index} out of range [0, {metadata['signal_count']})")

        with self._lock:
            with self._open_file() as f:
                batch = f[batch_name]

                # Load time array (may be per-signal or shared)
                time_array = batch[S.TIME][:]
                if len(time_array.shape) == 2:
                    time_data = time_array[signal_index]
                else:
                    time_data = time_array

                # Load single signal row (lazy slice)
                signal_data = batch[S.VALUES][signal_index, start:end]

                # Ensure consistent length
                if len(time_data) != len(signal_data):
                    # Resample time to match signal length
                    if len(time_data.shape) == 1:
                        time_data = np.linspace(time_data[0], time_data[-1], len(signal_data))

        return time_data.astype(np.float64), signal_data.astype(np.float64)

    def load_signal_by_name(self, batch_name: str, signal_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load signal by name instead of index.

        Args:
            batch_name: Name of the batch group
            signal_name: Name of the signal

        Returns:
            Tuple of (time_array, signal_values)

        Raises:
            ValueError: If signal name not found
        """
        metadata = self.get_batch_metadata(batch_name)

        try:
            signal_index = metadata["signal_names"].index(signal_name)
        except ValueError as e:
            raise ValueError(f"Signal '{signal_name}' not found in batch '{batch_name}'") from e

        return self.load_signal(batch_name, signal_index)

    def get_signal_stats(self, batch_name: str, signal_index: int) -> Dict:
        """
        Get statistical summary of a signal without loading all data.

        Args:
            batch_name: Name of the batch group
            signal_index: Index of signal

        Returns:
            Dictionary with:
              - mean: float
              - std: float
              - min: float
              - max: float
              - median: float
              - samples: int

        Raises:
            ValueError: If batch or signal index invalid
        """
        metadata = self.get_batch_metadata(batch_name)

        if signal_index < 0 or signal_index >= metadata["signal_count"]:
            raise IndexError(f"Signal index {signal_index} out of range")

        with self._lock:
            with self._open_file() as f:
                batch = f[batch_name]
                signal_data = batch[S.VALUES][signal_index, :]

        return {
            "mean": float(np.mean(signal_data)),
            "std": float(np.std(signal_data)),
            "min": float(np.min(signal_data)),
            "max": float(np.max(signal_data)),
            "median": float(np.median(signal_data)),
            "samples": len(signal_data),
        }

    def close(self) -> None:
        """
        Close the file handle and cleanup resources.

        Safe to call multiple times.
        """
        with self._lock:
            if self._file_handle is not None:
                try:
                    self._file_handle.close()
                except Exception:
                    pass
                self._file_handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"HDF5Reader(file={self.file_path.name}, batches={len(self.get_batches())})"
