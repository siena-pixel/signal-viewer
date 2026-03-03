"""
Filesystem Metadata Index for Time Series Data

Scans and indexes HDF5 files organized by serial (top-level), folder_1 (pXXX_label),
and folder_2 (any subfolder) directories.
Provides efficient lookup and caching of file metadata without parsing HDF5 content.

Folder structure: root/serial/pXXX_label/folder_2/file.h5

Steps are keyed as "folder_1/folder_2" combinations so the UI can present
every subfolder as a distinct selectable entry.
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
      - serial (e.g., "SN001", "SN002")
        - step = "folder_1/folder_2" (e.g., "p001_motor_test/run_nominal")
          - files (*.h5)

    Features:
      - Automatic scanning on initialization
      - Fast rescan for detecting new files
      - Graceful handling of missing/empty directories
      - Natural sorting of folder names
      - Lazy HDF5 metadata loading
      - Thread-safe operations
    """

    def __init__(self, root_path: str):
        """
        Initialize metadata index by scanning filesystem.

        Args:
            root_path: Root directory containing folder_1 directories

        Raises:
            ValueError: If root path does not exist
        """
        self.root_path = Path(root_path)

        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {root_path}")

        if not self.root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")

        self._lock = threading.RLock()
        self._index: Dict[str, Dict[str, List[Dict]]] = {}
        self._file_info_cache: Dict[str, Dict] = {}

        self.rescan()

    def rescan(self) -> None:
        """
        Re-scan filesystem to detect new or removed files.

        Safe to call multiple times. Updates index in place.
        """
        with self._lock:
            new_index: Dict[str, Dict[str, List[Dict]]] = {}

            # Iterate through serial directories (top level)
            for serial_dir in sorted(self.root_path.iterdir()):
                if not serial_dir.is_dir() or serial_dir.name.startswith("."):
                    continue

                serial_name = serial_dir.name
                new_index[serial_name] = {}

                # Iterate through folder_1 directories (pXXX_label)
                for folder1 in sorted(serial_dir.iterdir()):
                    if not folder1.is_dir() or folder1.name.startswith("."):
                        continue

                    folder1_name = folder1.name

                    # folder_1 must match the configured pattern
                    if not re.match(S.FOLDER1_REGEX, folder1_name):
                        continue

                    # Files directly in folder_1 (no folder_2) → step key = folder_1
                    h5_files_direct = sorted(folder1.glob(S.FILE_EXTENSION))
                    if h5_files_direct:
                        files_list = []
                        for h5_file in h5_files_direct:
                            files_list.append({
                                "path": str(h5_file),
                                "filename": h5_file.name,
                                "size": h5_file.stat().st_size,
                                "modified": h5_file.stat().st_mtime,
                            })
                        new_index[serial_name][folder1_name] = files_list

                    # Each folder_2 subdirectory → step key = folder_1/folder_2
                    folder2_dirs = [d for d in folder1.iterdir()
                                    if d.is_dir() and not d.name.startswith(".")]

                    for folder2 in sorted(folder2_dirs, key=lambda p: self._natural_sort_string(p.name)):
                        h5_files = sorted(folder2.glob(S.FILE_EXTENSION))
                        if h5_files:
                            step_key = f"{folder1_name}/{folder2.name}"
                            files_list = []
                            for h5_file in h5_files:
                                files_list.append({
                                    "path": str(h5_file),
                                    "filename": h5_file.name,
                                    "size": h5_file.stat().st_size,
                                    "modified": h5_file.stat().st_mtime,
                                })
                            new_index[serial_name][step_key] = files_list

            self._index = new_index
            self._file_info_cache.clear()

    def get_serial_numbers(self) -> List[str]:
        """
        Get list of all serial directory names (top-level).

        Returns:
            Sorted list of serial directory names
        """
        with self._lock:
            return sorted(self._index.keys(), key=self._natural_sort_string)

    def get_steps(self, serial_num: str) -> List[str]:
        """
        Get list of all step keys for a serial number.

        Each step is a "folder_1/folder_2" combination (e.g.,
        "p001_motor_test/run_nominal").  If files exist directly
        inside folder_1 (no folder_2), that entry is just "folder_1".

        Args:
            serial_num: Serial number identifier (e.g., "SN001")

        Returns:
            Sorted list of step keys (strings)

        Raises:
            ValueError: If serial number not found
        """
        with self._lock:
            if serial_num not in self._index:
                raise ValueError(f"Serial number not found: {serial_num}")

            return sorted(self._index[serial_num].keys(),
                          key=self._natural_sort_string)

    def get_files(self, serial_num: str, step: str) -> List[Dict]:
        """
        Get files for a specific serial number and step.

        Args:
            serial_num: Serial number identifier
            step: Step key ("folder_1/folder_2" or just "folder_1")

        Returns:
            List of dicts with: path, filename, size, modified

        Raises:
            ValueError: If serial number or step not found
        """
        with self._lock:
            if serial_num not in self._index:
                raise ValueError(f"Serial number not found: {serial_num}")

            if step not in self._index[serial_num]:
                raise ValueError(f"Step '{step}' not found for serial number {serial_num}")

            files = self._index[serial_num][step]
            return [f.copy() for f in files]

    def get_file_info(self, file_path: str) -> Dict:
        """
        Get detailed info about an HDF5 file by reading its contents.

        Opens the HDF5 file briefly to get group and signal information.
        Results are cached.

        Args:
            file_path: Absolute path to HDF5 file

        Returns:
            Dictionary with:
              - path: str
              - filename: str
              - size: int
              - modified: float
              - groups: List[str]
              - signal_counts: Dict[group_name -> int]

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is not a valid HDF5 file or outside index root
        """
        file_path = str(Path(file_path).resolve())
        root_resolved = str(self.root_path.resolve())
        if not file_path.startswith(root_resolved + "/"):
            raise ValueError("File path outside index root")

        with self._lock:
            if file_path in self._file_info_cache:
                return self._file_info_cache[file_path].copy()

        try:
            with HDF5Reader(file_path) as reader:
                groups = reader.get_groups()
                signal_counts = {}

                for group in groups:
                    metadata = reader.get_group_metadata(group)
                    signal_counts[group] = metadata["signal_count"]

                file_obj = Path(file_path)
                info = {
                    "path": file_path,
                    "filename": file_obj.name,
                    "size": file_obj.stat().st_size,
                    "modified": file_obj.stat().st_mtime,
                    "groups": groups,
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
                      "name": str,
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
                        "name": step,
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
    def _natural_sort_string(s: str) -> tuple:
        """
        Return sort key for natural sorting of strings.

        Handles embedded numbers naturally.

        Example:
          - "run_1" < "run_2" < "run_10" (not "run_1" < "run_10" < "run_2")

        Args:
            s: String to sort

        Returns:
            Tuple suitable for sorting
        """
        def convert(text):
            return int(text) if text.isdigit() else text.lower()

        return tuple(convert(c) for c in re.split(r"(\d+)", s))

    def __repr__(self) -> str:
        """String representation."""
        serial_count = len(self.get_serial_numbers())
        return f"MetadataIndex(root={self.root_path.name}, serials={serial_count})"
