"""
HDF5 Reader Module for Engineering Time Series Viewer

Provides robust HDF5 file reading with lazy loading, caching support, and
comprehensive metadata handling. Includes a MockHDF5File class for testing
without h5py dependency.

Dataset names are configured per group via HDF5Schema.GROUP_DS_NAMES.
Only VALUE is strictly required; all other datasets degrade gracefully:

  NSAMPLE       – missing → use full sample count from VALUE shape
  TIME          – missing → t0 = 0.0
  SAMPLING_FREQ – missing → parse Hz from VALUE dataset name suffix
                             (e.g. _0050 → 50 Hz), else 1.0 Hz
  NAMES         – missing → auto-generated as Signal_0, Signal_1, …
  UNITS         – missing → empty strings

Groups where VALUE is absent or empty (0 signals or 0 samples) are
silently excluded from get_groups().

Batch types:
  Type A – no ERROR key in config → base datasets only
  Type B – ERROR key present → base + error/quality datasets
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

    Groups and dataset names are driven entirely by HDF5Schema.GROUP_DS_NAMES.
    Groups with an ERROR key are populated as Type B; others as Type A.
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
        epoch_ms = np.array([1700000000000.0] * num_signals, dtype=np.float64)
        samp_freqs = np.array([100.0, 200.0, 100.0, 500.0], dtype=np.float64)
        nsample = np.array([800, 600, 900, 750], dtype=np.int64)

        for group_name, ds_map in S.GROUP_DS_NAMES.items():
            values = rng.randn(num_signals, max_samples).astype(np.float64)
            datasets = {
                ds_map['VALUE']:         values,
                ds_map['TIME']:          epoch_ms.copy(),
                ds_map['SAMPLING_FREQ']: samp_freqs.copy(),
                ds_map['NSAMPLE']:       nsample.copy(),
                ds_map['NAMES']:         names.copy(),
                ds_map['UNITS']:         units.copy(),
            }
            # Type B: add ERROR, SQI, TLS if configured
            if 'ERROR' in ds_map:
                datasets[ds_map['ERROR']] = rng.randn(num_signals, max_samples).astype(np.float64) * 0.1
            if 'SQI' in ds_map:
                datasets[ds_map['SQI']] = rng.uniform(0.8, 1.0, size=num_signals).astype(np.float64)
            if 'TLS' in ds_map:
                datasets[ds_map['TLS']] = rng.uniform(0.0, 5.0, size=num_signals).astype(np.float64)

            self._groups[group_name] = datasets

    def keys(self) -> List[str]:
        if self._closed:
            raise ValueError("I/O operation on closed file")
        return list(self._groups.keys())

    def __getitem__(self, key: str) -> "MockHDF5Group":
        if self._closed:
            raise ValueError("I/O operation on closed file")
        if key not in self._groups:
            raise KeyError(f"Group '{key}' not found")
        return MockHDF5Group(self._groups[key])

    def __contains__(self, key: str) -> bool:
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

    Groups with an ERROR key in GROUP_DS_NAMES are created as Type B;
    others as Type A.

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
                for group_name, ds_map in S.GROUP_DS_NAMES.items():
                    grp = f.create_group(group_name)
                    values = rng.randn(num_signals, max_samples).astype(np.float64)
                    grp.create_dataset(ds_map['VALUE'], data=values)
                    grp.create_dataset(ds_map['TIME'], data=epoch_ms)
                    grp.create_dataset(ds_map['SAMPLING_FREQ'], data=samp_freqs)
                    grp.create_dataset(ds_map['NSAMPLE'], data=nsample)
                    grp.create_dataset(ds_map['NAMES'], data=names)
                    grp.create_dataset(ds_map['UNITS'], data=units)
                    if 'ERROR' in ds_map:
                        errors = rng.randn(num_signals, max_samples).astype(np.float64) * 0.1
                        grp.create_dataset(ds_map['ERROR'], data=errors)
                    if 'SQI' in ds_map:
                        sqi = rng.uniform(0.8, 1.0, size=num_signals).astype(np.float64)
                        grp.create_dataset(ds_map['SQI'], data=sqi)
                    if 'TLS' in ds_map:
                        tls = rng.uniform(0.0, 5.0, size=num_signals).astype(np.float64)
                        grp.create_dataset(ds_map['TLS'], data=tls)
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

    Reads only the groups listed in HDF5Schema.GROUP_DS_NAMES.
    Dataset names are resolved per group via the config mapping.

    Supports two batch types:
      Type A – no ERROR key → base datasets only
      Type B – ERROR key present → base + error/quality datasets
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
        try:
            with self._open_file() as f:
                if len(f.keys()) == 0:
                    raise ValueError("HDF5 file contains no groups")
        except Exception as e:
            raise ValueError(f"Invalid HDF5 file: {self.file_path}") from e

    def _open_file(self):
        if HAS_H5PY:
            return h5py.File(self.file_path, "r")
        return MockHDF5File(str(self.file_path), "r")

    # ------------------------------------------------------------------
    # Group discovery
    # ------------------------------------------------------------------

    def get_groups(self) -> List[str]:
        """Return configured group names that exist in the file.

        Groups are excluded when:
        - The group is not present in the file
        - VALUE dataset is missing from the group
        - VALUE dataset is empty (0 signals or 0 samples)
        """
        if self._groups_cache is not None:
            return self._groups_cache
        with self._lock:
            with self._open_file() as f:
                file_keys = list(f.keys())
                valid = []
                for g in S.group_names():
                    if g not in file_keys:
                        continue
                    ds_map = S.GROUP_DS_NAMES[g]
                    if 'VALUE' not in ds_map:
                        continue
                    group = f[g]
                    if ds_map['VALUE'] not in group:
                        continue
                    val = group[ds_map['VALUE']]
                    if hasattr(val, 'shape'):
                        if val.ndim < 2 or val.shape[0] == 0 or val.shape[1] == 0:
                            continue
                    valid.append(g)
                self._groups_cache = valid
        return self._groups_cache

    # Backward-compatible alias
    get_batches = get_groups

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_group_metadata(self, group_name: str) -> Dict:
        """
        Get metadata for a group without loading signal data.

        Returns:
            Dictionary with signal_count, sample_count, signal_names, units,
            batch_type ("A" or "B"), n_samples, and for Type B: sqi, tls.
        """
        if group_name in self._metadata_cache:
            return self._metadata_cache[group_name]

        if group_name not in S.GROUP_DS_NAMES:
            raise ValueError(f"Group '{group_name}' not in schema config")

        ds_map = S.GROUP_DS_NAMES[group_name]

        with self._lock:
            with self._open_file() as f:
                if group_name not in f:
                    raise ValueError(f"Group '{group_name}' not found in file")

                group = f[group_name]

                # VALUE is the only strictly required dataset
                if 'VALUE' not in ds_map or ds_map['VALUE'] not in group:
                    raise ValueError(
                        f"Group '{group_name}' missing required VALUE dataset"
                    )

                val_ds = group[ds_map['VALUE']]
                signal_count = val_ds.shape[0]
                sample_count = val_ds.shape[1]

                if signal_count == 0 or sample_count == 0:
                    raise ValueError(
                        f"Group '{group_name}' VALUE dataset is empty"
                    )

                # NAMES — optional: auto-generate if absent or blank
                ds_names = ds_map.get('NAMES')
                if ds_names and ds_names in group:
                    raw_names = [
                        n.decode("utf-8").strip() if isinstance(n, bytes) else str(n).strip()
                        for n in group[ds_names][:]
                    ]
                    # Replace any blank entries with auto-generated names
                    signal_names = [
                        name if name else f"Signal_{i}"
                        for i, name in enumerate(raw_names)
                    ]
                else:
                    signal_names = [f"Signal_{i}" for i in range(signal_count)]

                # UNITS — optional: default to empty strings
                ds_units = ds_map.get('UNITS')
                if ds_units and ds_units in group:
                    units_list = [
                        u.decode("utf-8").strip() if isinstance(u, bytes) else str(u).strip()
                        for u in group[ds_units][:]
                    ]
                else:
                    units_list = [""] * signal_count

                # NSAMPLE — optional: default to full sample count
                ds_nsample = ds_map.get('NSAMPLE')
                if ds_nsample and ds_nsample in group:
                    n_samples = [int(v) for v in group[ds_nsample][:]]
                else:
                    n_samples = [sample_count] * signal_count

                # Type B detection: ERROR key configured AND present in file
                is_type_b = 'ERROR' in ds_map and ds_map['ERROR'] in group

                metadata = {
                    "signal_count": signal_count,
                    "sample_count": sample_count,
                    "signal_names": signal_names,
                    "units": units_list,
                    "batch_type": "B" if is_type_b else "A",
                    "n_samples": n_samples,
                }

                if is_type_b:
                    if 'SQI' in ds_map and ds_map['SQI'] in group:
                        metadata["sqi"] = [float(v) for v in group[ds_map['SQI']][:]]
                    if 'TLS' in ds_map and ds_map['TLS'] in group:
                        metadata["tls"] = [float(v) for v in group[ds_map['TLS']][:]]

                self._metadata_cache[group_name] = metadata

        return metadata

    # Backward-compatible alias
    get_batch_metadata = get_group_metadata

    # ------------------------------------------------------------------
    # Signal loading
    # ------------------------------------------------------------------

    def load_signal(self, group_name: str, signal_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single signal with constructed time vector.

        time = t0 + arange(n) * (1000.0 / fs)

        Fallbacks:
          TIME          missing → t0 = 0.0
          SAMPLING_FREQ missing → parsed from VALUE name suffix, else 1.0 Hz

        Returns:
            Tuple of (time_array, signal_values)
        """
        metadata = self.get_group_metadata(group_name)
        if signal_index < 0 or signal_index >= metadata["signal_count"]:
            raise IndexError(
                f"Signal index {signal_index} out of range [0, {metadata['signal_count']})"
            )

        ds_map = S.GROUP_DS_NAMES[group_name]
        n = metadata["n_samples"][signal_index]

        with self._lock:
            with self._open_file() as f:
                group = f[group_name]
                signal_data = group[ds_map['VALUE']][signal_index, :n].astype(np.float64)

                # TIME — optional: default 0.0
                ds_time = ds_map.get('TIME')
                if ds_time and ds_time in group:
                    t0 = float(group[ds_time][signal_index])
                else:
                    t0 = 0.0

                # SAMPLING_FREQ — optional: parse from VALUE name, else 1.0
                ds_freq = ds_map.get('SAMPLING_FREQ')
                if ds_freq and ds_freq in group:
                    fs = float(group[ds_freq][signal_index])
                else:
                    parsed = S.parse_freq_from_name(ds_map['VALUE'])
                    fs = parsed if parsed else 1.0

        time_data = t0 + np.arange(n, dtype=np.float64) * (1000.0 / fs)
        return time_data, signal_data

    def load_signal_by_name(self, group_name: str, signal_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load signal by name instead of index."""
        metadata = self.get_group_metadata(group_name)
        try:
            signal_index = metadata["signal_names"].index(signal_name)
        except ValueError as e:
            raise ValueError(f"Signal '{signal_name}' not found in group '{group_name}'") from e
        return self.load_signal(group_name, signal_index)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_signal_stats(self, group_name: str, signal_index: int) -> Dict:
        """Get statistical summary of a signal (respects n_sample truncation)."""
        metadata = self.get_group_metadata(group_name)
        if signal_index < 0 or signal_index >= metadata["signal_count"]:
            raise IndexError(f"Signal index {signal_index} out of range")

        ds_map = S.GROUP_DS_NAMES[group_name]
        n = metadata["n_samples"][signal_index]

        with self._lock:
            with self._open_file() as f:
                signal_data = f[group_name][ds_map['VALUE']][signal_index, :n]

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
