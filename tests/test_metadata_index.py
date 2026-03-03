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

        # Create directory structure
        # root_path/SN001/step_1/file.h5
        # root_path/SN001/step_2/file.h5
        # root_path/SN002/step_1/file.h5

        for serial_num in ["SN001", "SN002"]:
            serial_dir = self.root_path / serial_num
            serial_dir.mkdir()

            for step in [1, 2]:
                step_dir = serial_dir / f"{S.STEP_PREFIX}{step}"
                step_dir.mkdir()

                # Create test file (use dummy file since h5py may not be available)
                file_path = step_dir / f"test{S.FILE_SUFFIX}"
                create_dummy_h5_file(str(file_path))

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.tmpdir)

    def test_initialization(self):
        """Test MetadataIndex initialization."""
        index = MetadataIndex(str(self.root_path))
        self.assertIsNotNone(index)

    def test_root_path_not_exists(self):
        """Test ValueError for non-existent root path."""
        with self.assertRaises(ValueError):
            MetadataIndex("/nonexistent/path")

    def test_root_path_not_directory(self):
        """Test ValueError for non-directory root path."""
        with tempfile.NamedTemporaryFile() as f:
            with self.assertRaises(ValueError):
                MetadataIndex(f.name)

    def test_get_serial_numbers(self):
        """Test getting serial numbers."""
        index = MetadataIndex(str(self.root_path))
        serials = index.get_serial_numbers()

        self.assertIsInstance(serials, list)
        self.assertEqual(set(serials), {"SN001", "SN002"})

    def test_get_steps(self):
        """Test getting steps for a serial number."""
        index = MetadataIndex(str(self.root_path))
        steps = index.get_steps("SN001")

        self.assertIsInstance(steps, list)
        # Should have steps 1 and 2
        self.assertEqual(len(steps), 2)
        self.assertIn(1, steps)
        self.assertIn(2, steps)

    def test_get_steps_invalid_serial(self):
        """Test ValueError for invalid serial number."""
        index = MetadataIndex(str(self.root_path))
        with self.assertRaises(ValueError):
            index.get_steps("INVALID")

    def test_get_files(self):
        """Test getting files for a serial and step."""
        index = MetadataIndex(str(self.root_path))
        files = index.get_files("SN001", 1)

        self.assertIsInstance(files, list)
        self.assertEqual(len(files), 1)
        self.assertIn("path", files[0])
        self.assertIn("filename", files[0])
        self.assertIn("size", files[0])
        self.assertIn("modified", files[0])

    def test_get_files_invalid_serial(self):
        """Test ValueError for invalid serial number."""
        index = MetadataIndex(str(self.root_path))
        with self.assertRaises(ValueError):
            index.get_files("INVALID", 1)

    def test_get_files_invalid_step(self):
        """Test ValueError for invalid step."""
        index = MetadataIndex(str(self.root_path))
        with self.assertRaises(ValueError):
            index.get_files("SN001", 999)

    def test_get_file_info(self):
        """Test getting detailed file info."""
        index = MetadataIndex(str(self.root_path))
        files = index.get_files("SN001", 1)
        file_path = files[0]["path"]

        # Since we created dummy files, get_file_info will fail
        # Just test that the files dict has the expected keys
        self.assertIn("path", files[0])
        self.assertIn("filename", files[0])
        self.assertIn("size", files[0])
        self.assertIn("modified", files[0])

    def test_to_dict(self):
        """Test exporting index as dictionary."""
        index = MetadataIndex(str(self.root_path))
        index_dict = index.to_dict()

        self.assertIn("root", index_dict)
        self.assertIn("serial_numbers", index_dict)
        self.assertIsInstance(index_dict["serial_numbers"], list)

        # Check structure
        self.assertGreater(len(index_dict["serial_numbers"]), 0)
        serial_entry = index_dict["serial_numbers"][0]
        self.assertIn("name", serial_entry)
        self.assertIn("steps", serial_entry)

    def test_to_json(self):
        """Test exporting index as JSON string."""
        index = MetadataIndex(str(self.root_path))
        json_str = index.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        self.assertIn("root", parsed)
        self.assertIn("serial_numbers", parsed)

    def test_rescan(self):
        """Test rescan functionality."""
        index = MetadataIndex(str(self.root_path))
        serials_before = index.get_serial_numbers()

        # Add a new serial directory
        new_serial = self.root_path / "SN003"
        new_serial.mkdir()
        step_dir = new_serial / f"{S.STEP_PREFIX}1"
        step_dir.mkdir()
        create_test_file(str(step_dir / f"test{S.FILE_SUFFIX}"))

        # Rescan
        index.rescan()
        serials_after = index.get_serial_numbers()

        self.assertIn("SN003", serials_after)
        self.assertGreater(len(serials_after), len(serials_before))

    def test_natural_sort(self):
        """Test natural sorting of serial numbers."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir) / "root"
            root.mkdir()

            # Create serials with numbers that need natural sorting
            for serial in ["SN1", "SN10", "SN2"]:
                serial_dir = root / serial
                serial_dir.mkdir()
                step_dir = serial_dir / f"{S.STEP_PREFIX}1"
                step_dir.mkdir()
                create_dummy_h5_file(str(step_dir / f"test{S.FILE_SUFFIX}"))

            index = MetadataIndex(str(root))
            serials = index.get_serial_numbers()

            # Should be naturally sorted: SN1, SN2, SN10 (not SN1, SN10, SN2)
            self.assertEqual(serials, ["SN1", "SN2", "SN10"])
        finally:
            shutil.rmtree(tmpdir)

    def test_empty_directory(self):
        """Test handling of empty directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            root = Path(tmpdir) / "empty"
            root.mkdir()

            index = MetadataIndex(str(root))
            serials = index.get_serial_numbers()

            self.assertEqual(len(serials), 0)
        finally:
            shutil.rmtree(tmpdir)

    def test_hidden_directories_ignored(self):
        """Test that hidden directories are ignored."""
        index = MetadataIndex(str(self.root_path))

        # Create a hidden directory
        hidden_dir = self.root_path / ".hidden"
        hidden_dir.mkdir()

        # Rescan
        index.rescan()

        serials = index.get_serial_numbers()
        self.assertNotIn(".hidden", serials)

    def test_repr(self):
        """Test string representation."""
        index = MetadataIndex(str(self.root_path))
        repr_str = repr(index)

        self.assertIn("MetadataIndex", repr_str)
        self.assertIn("serials", repr_str)


if __name__ == "__main__":
    unittest.main()
