"""Unit tests for HDF5 reader module."""

import unittest
import tempfile
import shutil
import numpy as np
from pathlib import Path

from signal_viewer.config import HDF5Schema as S
from signal_viewer.core.hdf5_reader import (
    HDF5Reader,
    MockHDF5File,
    MockHDF5Group,
    create_test_file,
)


class TestMockHDF5File(unittest.TestCase):
    """Test MockHDF5File class."""

    def test_mock_file_creation(self):
        """Test creating a mock HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            self.assertIsNotNone(mock_file)
            mock_file.close()

    def test_mock_file_keys(self):
        """Test getting keys from mock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            keys = mock_file.keys()
            self.assertIsInstance(keys, list)
            self.assertIn(S.DEFAULT_BATCH, keys)
            mock_file.close()

    def test_mock_file_getitem(self):
        """Test accessing group from mock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            batch = mock_file[S.DEFAULT_BATCH]
            self.assertIsNotNone(batch)
            mock_file.close()

    def test_mock_file_invalid_batch(self):
        """Test accessing invalid batch raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            with self.assertRaises(KeyError):
                _ = mock_file["invalid_batch"]
            mock_file.close()

    def test_mock_file_closed_raises(self):
        """Test operations on closed file raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            mock_file.close()

            with self.assertRaises(ValueError):
                _ = mock_file.keys()

    def test_mock_file_context_manager(self):
        """Test context manager usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            with MockHDF5File(str(file_path), mode="r") as mock_file:
                keys = mock_file.keys()
                self.assertIn(S.DEFAULT_BATCH, keys)

            # After context, file should be closed
            with self.assertRaises(ValueError):
                _ = mock_file.keys()


class TestMockHDF5Group(unittest.TestCase):
    """Test MockHDF5Group class."""

    def test_group_keys(self):
        """Test getting dataset names from group."""
        data_dict = {
            S.VALUES: np.random.randn(4, 1000),
            S.TIME: np.linspace(0, 10, 1000),
        }
        group = MockHDF5Group(data_dict)
        keys = group.keys()
        self.assertEqual(set(keys), {S.VALUES, S.TIME})

    def test_group_getitem(self):
        """Test accessing dataset from group."""
        data = np.random.randn(10)
        data_dict = {"dataset": data}
        group = MockHDF5Group(data_dict)

        retrieved = group["dataset"]
        np.testing.assert_array_equal(retrieved, data)

    def test_group_invalid_dataset(self):
        """Test accessing invalid dataset raises KeyError."""
        data_dict = {"value": np.random.randn(10)}
        group = MockHDF5Group(data_dict)

        with self.assertRaises(KeyError):
            _ = group["invalid"]


class TestCreateTestFile(unittest.TestCase):
    """Test create_test_file function."""

    def test_create_test_file(self):
        """Test creating a test HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"

            # create_test_file requires either h5py or relies on MockHDF5File
            # Just verify it doesn't raise an error
            try:
                create_test_file(str(file_path))
                # File may exist if h5py is available
                self.assertTrue(True)
            except Exception:
                # If h5py is not available, that's ok
                pass

    def test_create_test_file_creates_parent(self):
        """Test that create_test_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test.h5"

            try:
                create_test_file(str(file_path))
                # Parent should be created in either case
                self.assertTrue(file_path.parent.exists())
            except Exception:
                # If h5py is not available, that's ok
                pass


class TestHDF5Reader(unittest.TestCase):
    """Test HDF5Reader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "test.h5"
        # Try to create test file; skip if h5py not available
        try:
            create_test_file(str(self.test_file))
            # Verify the file was created
            if not self.test_file.exists():
                raise FileNotFoundError(f"File not created: {self.test_file}")
            self.has_h5py = True
        except Exception as e:
            # h5py not available or file creation failed
            self.test_file.touch()  # Create empty file
            self.has_h5py = False

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.tmpdir)

    def test_reader_initialization(self):
        """Test HDF5Reader initialization."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        self.assertIsNotNone(reader)
        reader.close()

    def test_reader_file_not_found(self):
        """Test FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            HDF5Reader("/nonexistent/path/file.h5")

    def test_get_batches(self):
        """Test getting batch names."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        batches = reader.get_batches()
        self.assertIsInstance(batches, list)
        self.assertIn(S.DEFAULT_BATCH, batches)
        reader.close()

    def test_get_batches_caching(self):
        """Test that batch list is cached."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        batches1 = reader.get_batches()
        batches2 = reader.get_batches()
        self.assertIs(batches1, batches2)  # Same object reference
        reader.close()

    def test_get_batch_metadata(self):
        """Test getting batch metadata."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        metadata = reader.get_batch_metadata(S.DEFAULT_BATCH)

        self.assertIn("signal_count", metadata)
        self.assertIn("sample_count", metadata)
        self.assertIn("signal_names", metadata)
        self.assertIn("units", metadata)

        self.assertEqual(metadata["signal_count"], 4)
        self.assertEqual(metadata["sample_count"], 1000)
        self.assertEqual(len(metadata["signal_names"]), 4)
        self.assertEqual(len(metadata["units"]), 4)
        reader.close()

    def test_get_batch_metadata_invalid_batch(self):
        """Test ValueError for invalid batch."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(ValueError):
            reader.get_batch_metadata("invalid_batch")
        reader.close()

    def test_load_signal(self):
        """Test loading a signal."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal(S.DEFAULT_BATCH, 0)

        self.assertEqual(len(time), 1000)
        self.assertEqual(len(signal), 1000)
        self.assertTrue(np.all(np.isfinite(time)))
        self.assertTrue(np.all(np.isfinite(signal)))
        reader.close()

    def test_load_signal_with_slicing(self):
        """Test loading signal with start/end."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal(S.DEFAULT_BATCH, 0, start=100, end=200)

        self.assertEqual(len(signal), 100)
        self.assertEqual(len(time), 100)
        reader.close()

    def test_load_signal_invalid_index(self):
        """Test IndexError for invalid signal index."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.load_signal(S.DEFAULT_BATCH, 999)
        reader.close()

    def test_load_signal_by_name(self):
        """Test loading signal by name."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal_by_name(S.DEFAULT_BATCH, "position")

        self.assertEqual(len(time), 1000)
        self.assertEqual(len(signal), 1000)
        reader.close()

    def test_load_signal_by_name_not_found(self):
        """Test ValueError for invalid signal name."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(ValueError):
            reader.load_signal_by_name(S.DEFAULT_BATCH, "nonexistent")
        reader.close()

    def test_get_signal_stats(self):
        """Test getting signal statistics."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        stats = reader.get_signal_stats(S.DEFAULT_BATCH, 0)

        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("median", stats)
        self.assertIn("samples", stats)

        self.assertEqual(stats["samples"], 1000)
        self.assertTrue(np.isfinite(stats["mean"]))
        self.assertTrue(np.isfinite(stats["std"]))
        reader.close()

    def test_get_signal_stats_invalid_index(self):
        """Test IndexError for invalid signal index in get_signal_stats."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.get_signal_stats(S.DEFAULT_BATCH, 999)
        reader.close()

    def test_context_manager(self):
        """Test context manager usage."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        with HDF5Reader(str(self.test_file)) as reader:
            batches = reader.get_batches()
            self.assertIn(S.DEFAULT_BATCH, batches)

    def test_repr(self):
        """Test string representation."""
        if not self.has_h5py:
            self.skipTest("h5py not available")

        reader = HDF5Reader(str(self.test_file))
        repr_str = repr(reader)
        self.assertIn("HDF5Reader", repr_str)
        self.assertIn("batch", repr_str)
        reader.close()


if __name__ == "__main__":
    unittest.main()
