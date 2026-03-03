"""
Filesystem Metadata Index for Time Series Data

Scans and indexes HDF5 files organized by serial number and step directories.
Provides efficient lookup and caching of file metadata without parsing HDF5 content.

Folder structure: root/serial_number/step_N/serial_date_crc.h5
"""

import re
import threading
from pathlib import Path
from typing import Dict, List, Optional
import json

from ..config import HDF5Schema as S
from .hdf5_reader import HDF5Reader


class MetadataIndex:
    """
    Filesystem scanner and metadata indexer for HDF5 signal files.

    Organizes files hierarchically:
      - serial_number (e.g., "SN001", "SN002")
        - step_N (e.g., "step_1", "step_5")
          - files (serial_date_crc.h5)

    Features:
      - Automatic scanning on initialization
      - Fast rescan for detecting new files
      - Graceful handling of missing/empty directories
      - Natural sorting of serial numbers and step indices
      - Lazy HDF5 metadata loading
      - Thread-safe operations
    """

    def __init__(self, root_path: str):
        """
        Initialize metadata index by scanning filesystem.

        Args:
            root_path: Root directory containing serial_number folders

        Raises:
            ValueError: If root path does not exist
        """
        self.root_path = Path(root_path)

        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")

        if not self.root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")

        self._lock = threading.RLock()
        self._index: Dict[str, Dict[int, List[Dict]]] = {}
        self._file_info_cache: Dict[str, Dict] = {}

        self.rescan()

    def rescan(self) -> None:
        """
        Re-scan filesystem to detect new or removed files.

        Safe to call multiple times. Updates index in place.
        """
        with self._lock:
            new_index: Dict[str, Dict[int, List[Dict]]] = {}

            # Iterate through serial number directories
            for serial_dir in sorted(self.root_path.iterdir()):
                if not serial_dir.is_dir():
                    continue

                serial_num = serial_dir.name

                # Skip hidden directories
                if serial_num.startswith("."):
                    continue

                new_index[serial_num] = {}

                # Iterate through step_N directories
                step_dirs = [d for d in serial_dir.iterdir() if d.is_dir() and d.name.startswith(S.STEP_PREFIX)]

                for step_dir in sorted(step_dirs, key=self._natural_sort_step):
                    step_num = self._parse_step_number(step_dir.name)

                    if step_num is None:
                        continue

                    files_list = []

                    # Find all .h5 files in this step directory
                    h5_files = sorted(step_dir.glob(S.FILE_EXTENSION))

                    for h5_file in h5_files:
                        file_info = {
                            "path": str(h5_file),
                            "filename": h5_file.name,
                            "size": h5_file.stat().st_size,
                            "modified": h5_file.stat().st_mtime,
                        }
                        files_list.append(file_info)

                    if files_list:
                        new_index[serial_num][step_num] = files_list

            self._index = new_index
            self._file_info_cache.clear()

    def get_serial_numbers(self) -> List[str]:
        """
        Get list of all serial numbers.

        Returns:
            Sorted list of serial number directory names
        """
        with self._lock:
            return sorted(self._index.keys(), key=self._natural_sort_string)

    def get_steps(self, serial_num: str) -> List[int]:
        """
        Get list of all step numbers for a serial number.

        Args:
            serial_num: Serial number identifier

        Returns:
            Sorted list of step indices

        Raises:
            ValueError: If serial number not found
        """
        with self._lock:
            if serial_num not in self._index:
                raise ValueError(f"Serial number not found: {serial_num}")

            return sorted(self._index[serial_num].keys())

    def get_files(self, serial_num: str, step: int) -> List[Dict]:
        """
        Get files for a specific serial number and step.

        Args:
            serial_num: Serial number identifier
            step: Step index number

        Returns:
            List of dicts with: path, filename, size, modified

        Raises:
            ValueError: If serial number or step not found
        """
        with self._lock:
            if serial_num not in self._index:
                raise ValueError(f"Serial number not found: {serial_num}")

            if step not in self._index[serial_num]:
                raise ValueError(f"Step {step} not found for serial number {serial_num}")

            files = self._index[serial_num][step]
            return [f.copy() for f in files]

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get detailed info about an HDF5 file by reading its contents.

        Opens the HDF5 file briefly to get batch and signal information.
        Results are cached.

        Args:
            file_path: Absolute path to HDF5 file

        Returns:
            Dictionary with:
              - path: str
              - filename: str
              - size: int
              - modified: float
              - batches: List[str]
              - signal_counts: Dict[batch_name -> int]

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is not a valid HDF5 file
        """
        file_path = str(Path(file_path).resolve())

        with self._lock:
            if file_path in self._file_info_cache:
                return self._file_info_cache[file_path].copy()

        try:
            with HDF5Reader(file_path) as reader:
                batches = reader.get_batches()
                signal_counts = {}

                for batch in batches:
                    metadata = reader.get_batch_metadata(batch)
                    signal_counts[batch] = metadata["signal_count"]

                file_obj = Path(file_path)
                info = {
                    "path": file_path,
                    "filename": file_obj.name,
                    "size": file_obj.stat().st_size,
                    "modified": file_obj.stat().st_mtime,
                    "batches": batches,
                    "signal_counts": signal_counts,
                }

                with self._lock:
                    self._file_info_cache[file_path] = info

                return info.copy()

        except Exception as e:
            raise ValueError(f"Failed to read HDF5 file: {file_path}") from e

    def to_dict(self) -> Dict:
        """
        Export full index as nested dictionary for JSON serialization.

        Returns:
            Dictionary structure:
            {
              "root": str,
              "serial_numbers": [
                {
                  "name": str,
                  "steps": [
                    {
                      "index": int,
                      "files": [
                        {
                          "path": str,
                          "filename": str,
                          "size": int,
                          "modified": float
                        },
                        ...
                      ]
                    },
                    ...
                  ]
                },
                ...
              ]
            }
        """
        with self._lock:
            result = {
                "root": str(self.root_path),
                "serial_numbers": [],
            }

            for serial_num in self.get_serial_numbers():
                serial_entry = {
                    "name": serial_num,
                    "steps": [],
                }

                for step in self.get_steps(serial_num):
                    step_entry = {
                        "index": step,
                        "files": self.get_files(serial_num, step),
                    }
                    serial_entry["steps"].append(step_entry)

                result["serial_numbers"].append(serial_entry)

            return result

    def to_json(self) -> str:
        """
        Export full index as JSON string.

        Returns:
            JSON-formatted string of the index
        """
        index_dict = self.to_dict()
        return json.dumps(index_dict, indent=2)

    @staticmethod
    def _parse_step_number(step_dir_name: str) -> Optional[int]:
        """
        Extract step number from directory name.

        Examples:
          - "step_1" -> 1
          - "step_42" -> 42
          - "invalid" -> None

        Args:
            step_dir_name: Directory name

        Returns:
            Step number or None if format is invalid
        """
        match = re.match(S.STEP_REGEX, step_dir_name)
        return int(match.group(1)) if match else None

    @staticmethod
    def _natural_sort_string(s: str) -> tuple:
        """
        Return sort key for natural sorting of strings.

        Handles embedded numbers naturally.

        Example:
          - "SN1" < "SN2" < "SN10" (not "SN1" < "SN10" < "SN2")

        Args:
            s: String to sort

        Returns:
            Tuple suitable for sorting
        """
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        return tuple(convert(c) for c in re.split(r"(\d+)", s))

    @staticmethod
    def _natural_sort_step(path: Path) -> tuple:
        """
        Return sort key for natural sorting of step directories.

        Args:
            path: Path object

        Returns:
            Tuple suitable for sorting
        """
        step_num = MetadataIndex._parse_step_number(path.name)
        return (step_num,) if step_num is not None else (float("inf"),)

    def __repr__(self) -> str:
        """String representation."""
        serial_count = len(self.get_serial_numbers())
        return f"MetadataIndex(root={self.root_path.name}, serials={serial_count})"
