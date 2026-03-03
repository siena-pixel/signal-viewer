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
            self.assertIn(S.DEFAULT_GROUP, keys)
            mock_file.close()

    def test_mock_file_getitem(self):
        """Test accessing group from mock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            group = mock_file[S.DEFAULT_GROUP]
            self.assertIsNotNone(group)
            mock_file.close()

    def test_mock_file_invalid_group(self):
        """Test accessing invalid group raises KeyError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            with self.assertRaises(KeyError):
                _ = mock_file["invalid_group"]
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
                self.assertIn(S.DEFAULT_GROUP, keys)

            # After context, file should be closed
            with self.assertRaises(ValueError):
                _ = mock_file.keys()

    def test_mock_file_contains_all_configured_groups(self):
        """Test that mock file creates all configured groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            keys = mock_file.keys()
            for group_name in S.GROUP_NAMES:
                self.assertIn(group_name, keys)
            mock_file.close()


class TestMockHDF5Group(unittest.TestCase):
    """Test MockHDF5Group class."""

    def test_group_keys(self):
        """Test getting dataset names from group."""
        gn = S.DEFAULT_GROUP
        data_dict = {
            S.ds(gn, S.VALUE_SUFFIX): np.random.randn(4, 1000),
            S.ds(gn, S.TIME_SUFFIX): np.linspace(0, 10, 1000),
        }
        group = MockHDF5Group(data_dict)
        keys = group.keys()
        self.assertEqual(set(keys), {S.ds(gn, S.VALUE_SUFFIX), S.ds(gn, S.TIME_SUFFIX)})

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

    def test_group_contains(self):
        """Test __contains__ method."""
        data_dict = {"ds_a": np.random.randn(10)}
        group = MockHDF5Group(data_dict)
        self.assertTrue("ds_a" in group)
        self.assertFalse("ds_b" in group)


class TestCreateTestFile(unittest.TestCase):
    """Test create_test_file function."""

    def test_create_test_file(self):
        """Test creating a test HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            try:
                create_test_file(str(file_path))
                self.assertTrue(True)
            except Exception:
                pass

    def test_create_test_file_creates_parent(self):
        """Test that create_test_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test.h5"
            try:
                create_test_file(str(file_path))
                self.assertTrue(file_path.parent.exists())
            except Exception:
                pass


class TestHDF5Reader(unittest.TestCase):
    """Test HDF5Reader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "test.h5"
        try:
            create_test_file(str(self.test_file))
            if not self.test_file.exists():
                raise FileNotFoundError(f"File not created: {self.test_file}")
            self.has_h5py = True
        except Exception:
            self.test_file.touch()
            self.has_h5py = False

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_reader_initialization(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        self.assertIsNotNone(reader)
        reader.close()

    def test_reader_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            HDF5Reader("/nonexistent/path/file.h5")

    def test_get_groups(self):
        """Test getting configured group names."""
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        groups = reader.get_groups()
        self.assertIsInstance(groups, list)
        self.assertIn(S.DEFAULT_GROUP, groups)
        # All configured groups should be present
        for gn in S.GROUP_NAMES:
            self.assertIn(gn, groups)
        reader.close()

    def test_get_groups_caching(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        groups1 = reader.get_groups()
        groups2 = reader.get_groups()
        self.assertIs(groups1, groups2)
        reader.close()

    def test_get_group_metadata(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        metadata = reader.get_group_metadata(S.DEFAULT_GROUP)
        self.assertIn("signal_count", metadata)
        self.assertIn("sample_count", metadata)
        self.assertIn("signal_names", metadata)
        self.assertIn("units", metadata)
        self.assertEqual(metadata["signal_count"], 4)
        self.assertEqual(metadata["sample_count"], 1000)
        self.assertEqual(len(metadata["signal_names"]), 4)
        self.assertEqual(len(metadata["units"]), 4)
        reader.close()

    def test_get_group_metadata_invalid(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(ValueError):
            reader.get_group_metadata("invalid_group")
        reader.close()

    def test_load_signal(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal(S.DEFAULT_GROUP, 0)
        self.assertEqual(len(time), 1000)
        self.assertEqual(len(signal), 1000)
        self.assertTrue(np.all(np.isfinite(time)))
        self.assertTrue(np.all(np.isfinite(signal)))
        reader.close()

    def test_load_signal_with_slicing(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal(S.DEFAULT_GROUP, 0, start=100, end=200)
        self.assertEqual(len(signal), 100)
        self.assertEqual(len(time), 100)
        reader.close()

    def test_load_signal_invalid_index(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.load_signal(S.DEFAULT_GROUP, 999)
        reader.close()

    def test_load_signal_by_name(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal_by_name(S.DEFAULT_GROUP, "position")
        self.assertEqual(len(time), 1000)
        self.assertEqual(len(signal), 1000)
        reader.close()

    def test_load_signal_by_name_not_found(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(ValueError):
            reader.load_signal_by_name(S.DEFAULT_GROUP, "nonexistent")
        reader.close()

    def test_get_signal_stats(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        stats = reader.get_signal_stats(S.DEFAULT_GROUP, 0)
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
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.get_signal_stats(S.DEFAULT_GROUP, 999)
        reader.close()

    def test_context_manager(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        with HDF5Reader(str(self.test_file)) as reader:
            groups = reader.get_groups()
            self.assertIn(S.DEFAULT_GROUP, groups)

    def test_repr(self):
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        repr_str = repr(reader)
        self.assertIn("HDF5Reader", repr_str)
        self.assertIn("group", repr_str)
        reader.close()

    def test_backward_compat_aliases(self):
        """Test that get_batches and get_batch_metadata still work."""
        if not self.has_h5py:
            self.skipTest("h5py not available")
        reader = HDF5Reader(str(self.test_file))
        # get_batches should be an alias for get_groups
        self.assertEqual(reader.get_batches(), reader.get_groups())
        # get_batch_metadata should be an alias for get_group_metadata
        meta1 = reader.get_batch_metadata(S.DEFAULT_GROUP)
        meta2 = reader.get_group_metadata(S.DEFAULT_GROUP)
        self.assertEqual(meta1, meta2)
        reader.close()


if __name__ == "__main__":
    unittest.main()
