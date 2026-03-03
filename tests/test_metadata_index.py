"""Unit tests for metadata index module."""

import unittest
import tempfile
import shutil
from pathlib import Path
import json

from signal_viewer.config import HDF5Schema as S
from signal_viewer.core.metadata_index import MetadataIndex
from signal_viewer.core.hdf5_reader import create_test_file


def create_dummy_h5_file(file_path):
    """Create a dummy empty .h5 file for testing."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    Path(file_path).touch()


class TestMetadataIndex(unittest.TestCase):
    """Test MetadataIndex class."""

    def setUp(self):
        """Set up test fixtures with directory structure."""
        self.tmpdir = tempfile.mkdtemp()
        self.root_path = Path(self.tmpdir) / "data"
        self.root_path.mkdir()

        # Create directory structure using new serial/folder_1/folder_2 convention
        # root_path/SN001/p001_motor/run_nominal/file.h5
        # root_path/SN001/p001_motor/run_overload/file.h5
        # root_path/SN002/p002_vibration/baseline/file.h5

        for serial in ["SN001", "SN002"]:
            serial_dir = self.root_path / serial
            serial_dir.mkdir()

            for folder1 in ["p001_motor", "p002_vibration"]:
                folder1_dir = serial_dir / folder1
                folder1_dir.mkdir()

                for folder2 in ["run_nominal", "run_overload"]:
                    folder2_dir = folder1_dir / folder2
                    folder2_dir.mkdir()

                    file_path = folder2_dir / f"test{S.FILE_SUFFIX}"
                    create_dummy_h5_file(str(file_path))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_initialization(self):
        index = MetadataIndex(str(self.root_path))
        self.assertIsNotNone(index)

    def test_root_path_not_exists(self):
        with self.assertRaises(ValueError):
            MetadataIndex("/nonexistent/path")

    def test_root_path_not_directory(self):
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(ValueError):
                MetadataIndex(f.name)

    def test_get_serial_numbers(self):
        """Test getting serial directory names (top level)."""
        index = MetadataIndex(str(self.root_path))
        serials = index.get_serial_numbers()
        self.assertIsInstance(serials, list)
        self.assertEqual(set(serials), {"SN001", "SN002"})

    def test_get_steps(self):
        """Test getting folder_1/folder_2 step keys for a serial."""
        index = MetadataIndex(str(self.root_path))
        steps = index.get_steps("SN001")
        self.assertIsInstance(steps, list)
        # 2 folder_1s × 2 folder_2s = 4 step combinations
        self.assertEqual(len(steps), 4)
        for step in steps:
            self.assertIsInstance(step, str)
        self.assertIn("p001_motor/run_nominal", steps)
        self.assertIn("p001_motor/run_overload", steps)
        self.assertIn("p002_vibration/run_nominal", steps)
        self.assertIn("p002_vibration/run_overload", steps)

    def test_get_steps_invalid_serial(self):
        index = MetadataIndex(str(self.root_path))
        with self.assertRaises(ValueError):
            index.get_steps("INVALID")

    def test_get_files(self):
        """Test getting files for a serial and folder_1/folder_2 step."""
        index = MetadataIndex(str(self.root_path))
        files = index.get_files("SN001", "p001_motor/run_nominal")
        self.assertIsInstance(files, list)
        self.assertEqual(len(files), 1)
        self.assertIn("path", files[0])
        self.assertIn("filename", files[0])
        self.assertIn("size", files[0])
        self.assertIn("modified", files[0])

    def test_get_files_invalid_serial(self):
        index = MetadataIndex(str(self.root_path))
        with self.assertRaises(ValueError):
            index.get_files("INVALID", "p001_motor/run_nominal")

    def test_get_files_invalid_step(self):
        index = MetadataIndex(str(self.root_path))
        with self.assertRaises(ValueError):
            index.get_files("SN001", "nonexistent_folder")

    def test_get_file_info(self):
        index = MetadataIndex(str(self.root_path))
        files = index.get_files("SN001", "p001_motor/run_nominal")
        self.assertIn("path", files[0])
        self.assertIn("filename", files[0])
        self.assertIn("size", files[0])
        self.assertIn("modified", files[0])

    def test_to_dict(self):
        index = MetadataIndex(str(self.root_path))
        index_dict = index.to_dict()
        self.assertIn("root", index_dict)
        self.assertIn("serial_numbers", index_dict)
        self.assertIsInstance(index_dict["serial_numbers"], list)
        self.assertGreater(len(index_dict["serial_numbers"]), 0)
        serial_entry = index_dict["serial_numbers"][0]
        self.assertIn("name", serial_entry)
        self.assertIn("steps", serial_entry)
        # Steps should have "name" key (folder_1/folder_2 strings)
        step_entry = serial_entry["steps"][0]
        self.assertIn("name", step_entry)
        self.assertIn("/", step_entry["name"])

    def test_to_json(self):
        index = MetadataIndex(str(self.root_path))
        json_str = index.to_json()
        parsed = json.loads(json_str)
        self.assertIn("root", parsed)
        self.assertIn("serial_numbers", parsed)

    def test_rescan(self):
        index = MetadataIndex(str(self.root_path))
        serials_before = index.get_serial_numbers()

        # Add a new serial directory
        new_serial = self.root_path / "SN003"
        new_serial.mkdir()
        folder1_dir = new_serial / "p003_endurance"
        folder1_dir.mkdir()
        folder2_dir = folder1_dir / "cycle_1"
        folder2_dir.mkdir()
        create_test_file(str(folder2_dir / f"test{S.FILE_SUFFIX}"))

        index.rescan()
        serials_after = index.get_serial_numbers()
        self.assertIn("SN003", serials_after)
        self.assertGreater(len(serials_after), len(serials_before))
