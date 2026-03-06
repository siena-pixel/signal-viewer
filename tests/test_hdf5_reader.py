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


# ---------------------------------------------------------------------------
# MockHDF5File tests
# ---------------------------------------------------------------------------

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

    def test_mock_type_a_has_nsample(self):
        """Test that first group (Type A) has _SAM dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            group = mock_file[S.GROUP_NAMES[0]]
            ds_sam = S.ds(S.GROUP_NAMES[0], S.NSAMPLE_SUFFIX)
            self.assertIn(ds_sam, group)
            mock_file.close()

    def test_mock_type_a_no_error_datasets(self):
        """Test that first group (Type A) has no _ERR/_SQI/_TLS datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            group = mock_file[S.GROUP_NAMES[0]]
            ds_err = S.ds(S.GROUP_NAMES[0], S.ERROR_SUFFIX)
            ds_sqi = S.ds(S.GROUP_NAMES[0], S.SQI_SUFFIX)
            ds_tls = S.ds(S.GROUP_NAMES[0], S.TLS_SUFFIX)
            self.assertNotIn(ds_err, group)
            self.assertNotIn(ds_sqi, group)
            self.assertNotIn(ds_tls, group)
            mock_file.close()

    def test_mock_type_b_has_nsample(self):
        """Test that second group (Type B) also has _SAM dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            group = mock_file[S.GROUP_NAMES[1]]
            ds_sam = S.ds(S.GROUP_NAMES[1], S.NSAMPLE_SUFFIX)
            self.assertIn(ds_sam, group)
            mock_file.close()

    def test_mock_type_b_has_error_datasets(self):
        """Test that second group (Type B) has _ERR, _SQI, _TLS datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            group = mock_file[S.GROUP_NAMES[1]]
            ds_err = S.ds(S.GROUP_NAMES[1], S.ERROR_SUFFIX)
            ds_sqi = S.ds(S.GROUP_NAMES[1], S.SQI_SUFFIX)
            ds_tls = S.ds(S.GROUP_NAMES[1], S.TLS_SUFFIX)
            self.assertIn(ds_err, group)
            self.assertIn(ds_sqi, group)
            self.assertIn(ds_tls, group)
            mock_file.close()


# ---------------------------------------------------------------------------
# MockHDF5Group tests
# ---------------------------------------------------------------------------

class TestMockHDF5Group(unittest.TestCase):
    """Test MockHDF5Group class."""

    def test_group_keys(self):
        """Test getting dataset names from group."""
        gn = S.DEFAULT_GROUP
        data_dict = {
            S.ds(gn, S.VALUE_SUFFIX): np.random.randn(4, 1000),
            S.ds(gn, S.TIME_SUFFIX): np.array([0.0, 0.5, 1.0, 2.0]),
            S.ds(gn, S.SAMPLING_FREQ_SUFFIX): np.array([100.0, 200.0, 100.0, 500.0]),
        }
        group = MockHDF5Group(data_dict)
        keys = group.keys()
        self.assertEqual(
            set(keys),
            {
                S.ds(gn, S.VALUE_SUFFIX),
                S.ds(gn, S.TIME_SUFFIX),
                S.ds(gn, S.SAMPLING_FREQ_SUFFIX),
            },
        )

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


# ---------------------------------------------------------------------------
# create_test_file tests
# ---------------------------------------------------------------------------

class TestCreateTestFile(unittest.TestCase):
    """Test create_test_file function."""

    def test_create_test_file(self):
        """Test creating a test HDF5 file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            create_test_file(str(file_path))
            self.assertTrue(file_path.exists())

    def test_create_test_file_creates_parent(self):
        """Test that create_test_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "subdir" / "test.h5"
            create_test_file(str(file_path))
            self.assertTrue(file_path.parent.exists())
            self.assertTrue(file_path.exists())


# ---------------------------------------------------------------------------
# HDF5Reader tests
# ---------------------------------------------------------------------------

class TestHDF5Reader(unittest.TestCase):
    """Test HDF5Reader class (uses MockHDF5File when h5py is unavailable)."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.test_file = Path(self.tmpdir) / "test.h5"
        create_test_file(str(self.test_file))

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_reader_initialization(self):
        reader = HDF5Reader(str(self.test_file))
        self.assertIsNotNone(reader)
        reader.close()

    def test_reader_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            HDF5Reader("/nonexistent/path/file.h5")

    def test_get_groups(self):
        """Test getting configured group names."""
        reader = HDF5Reader(str(self.test_file))
        groups = reader.get_groups()
        self.assertIsInstance(groups, list)
        self.assertIn(S.DEFAULT_GROUP, groups)
        for gn in S.GROUP_NAMES:
            self.assertIn(gn, groups)
        reader.close()

    def test_get_groups_caching(self):
        reader = HDF5Reader(str(self.test_file))
        groups1 = reader.get_groups()
        groups2 = reader.get_groups()
        self.assertIs(groups1, groups2)
        reader.close()

    def test_get_group_metadata(self):
        reader = HDF5Reader(str(self.test_file))
        metadata = reader.get_group_metadata(S.DEFAULT_GROUP)
        self.assertIn("signal_count", metadata)
        self.assertIn("sample_count", metadata)
        self.assertIn("signal_names", metadata)
        self.assertIn("units", metadata)
        self.assertIn("batch_type", metadata)
        self.assertIn("n_samples", metadata)
        self.assertEqual(metadata["signal_count"], 4)
        self.assertEqual(metadata["sample_count"], 1000)
        self.assertEqual(len(metadata["signal_names"]), 4)
        self.assertEqual(len(metadata["units"]), 4)
        reader.close()

    def test_get_group_metadata_type_a(self):
        """Test Type A group has batch_type A and per-signal n_samples."""
        reader = HDF5Reader(str(self.test_file))
        meta = reader.get_group_metadata(S.GROUP_NAMES[0])
        self.assertEqual(meta["batch_type"], "A")
        self.assertIsNotNone(meta["n_samples"])
        self.assertEqual(len(meta["n_samples"]), 4)
        self.assertEqual(meta["n_samples"], [800, 600, 900, 750])
        # Type A should not have sqi/tls
        self.assertNotIn("sqi", meta)
        self.assertNotIn("tls", meta)
        reader.close()

    def test_get_group_metadata_type_b(self):
        """Test Type B group has batch_type B, n_samples, and sqi/tls."""
        reader = HDF5Reader(str(self.test_file))
        meta = reader.get_group_metadata(S.GROUP_NAMES[1])
        self.assertEqual(meta["batch_type"], "B")
        # Type B also has _SAM
        self.assertIsNotNone(meta["n_samples"])
        self.assertEqual(len(meta["n_samples"]), 4)
        self.assertEqual(meta["n_samples"], [800, 600, 900, 750])
        # Type B has sqi and tls
        self.assertIn("sqi", meta)
        self.assertIn("tls", meta)
        self.assertEqual(len(meta["sqi"]), 4)
        self.assertEqual(len(meta["tls"]), 4)
        reader.close()

    def test_get_group_metadata_invalid(self):
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(ValueError):
            reader.get_group_metadata("invalid_group")
        reader.close()

    def test_load_signal_type_a(self):
        """Test Type A: signal truncated to n_sample[idx]."""
        reader = HDF5Reader(str(self.test_file))
        group_a = S.GROUP_NAMES[0]

        # Signal 0: n_sample=800, Signal 1: n_sample=600
        time0, sig0 = reader.load_signal(group_a, 0)
        self.assertEqual(len(sig0), 800)
        self.assertEqual(len(time0), 800)

        time1, sig1 = reader.load_signal(group_a, 1)
        self.assertEqual(len(sig1), 600)
        self.assertEqual(len(time1), 600)
        reader.close()

    def test_load_signal_type_b(self):
        """Test Type B: signal also truncated to n_sample[idx] (both types have _SAM)."""
        reader = HDF5Reader(str(self.test_file))
        group_b = S.GROUP_NAMES[1]

        # Type B also uses _SAM: signal 0 has n_sample=800
        time, signal = reader.load_signal(group_b, 0)
        self.assertEqual(len(signal), 800)
        self.assertEqual(len(time), 800)
        reader.close()

    def test_time_construction(self):
        """Test time = t0 + arange(n) * (1000/fs) for both types."""
        reader = HDF5Reader(str(self.test_file))

        # Type A, signal 0: t0=1700000000000.0 (epoch ms), fs=100.0, n=800
        time, _ = reader.load_signal(S.GROUP_NAMES[0], 0)
        expected = 1700000000000.0 + np.arange(800, dtype=np.float64) * (1000.0 / 100.0)
        np.testing.assert_allclose(time, expected, rtol=1e-12)

        # Type A, signal 1: t0=1700000000000.0, fs=200.0, n=600
        time, _ = reader.load_signal(S.GROUP_NAMES[0], 1)
        expected = 1700000000000.0 + np.arange(600, dtype=np.float64) * (1000.0 / 200.0)
        np.testing.assert_allclose(time, expected, rtol=1e-12)

        # Type B, signal 3: t0=1700000000000.0, fs=500.0, n=750
        time, _ = reader.load_signal(S.GROUP_NAMES[1], 3)
        expected = 1700000000000.0 + np.arange(750, dtype=np.float64) * (1000.0 / 500.0)
        np.testing.assert_allclose(time, expected, rtol=1e-12)

        reader.close()

    def test_load_signal(self):
        """Test basic signal loading returns finite arrays."""
        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal(S.DEFAULT_GROUP, 0)
        self.assertGreater(len(time), 0)
        self.assertEqual(len(time), len(signal))
        self.assertTrue(np.all(np.isfinite(time)))
        self.assertTrue(np.all(np.isfinite(signal)))
        reader.close()

    def test_load_signal_invalid_index(self):
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.load_signal(S.DEFAULT_GROUP, 999)
        reader.close()

    def test_load_signal_by_name(self):
        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal_by_name(S.DEFAULT_GROUP, "position")
        self.assertGreater(len(time), 0)
        self.assertEqual(len(time), len(signal))
        reader.close()

    def test_load_signal_by_name_not_found(self):
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(ValueError):
            reader.load_signal_by_name(S.DEFAULT_GROUP, "nonexistent")
        reader.close()

    def test_get_signal_stats(self):
        reader = HDF5Reader(str(self.test_file))
        stats = reader.get_signal_stats(S.DEFAULT_GROUP, 0)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("median", stats)
        self.assertIn("samples", stats)
        # Type A signal 0 → 800 valid samples
        self.assertEqual(stats["samples"], 800)
        self.assertTrue(np.isfinite(stats["mean"]))
        self.assertTrue(np.isfinite(stats["std"]))
        reader.close()

    def test_get_signal_stats_type_b(self):
        """Test stats for Type B also uses _SAM truncation."""
        reader = HDF5Reader(str(self.test_file))
        stats = reader.get_signal_stats(S.GROUP_NAMES[1], 0)
        # Type B also has _SAM: signal 0 → 800 valid samples
        self.assertEqual(stats["samples"], 800)
        reader.close()

    def test_get_signal_stats_invalid_index(self):
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.get_signal_stats(S.DEFAULT_GROUP, 999)
        reader.close()

    def test_context_manager(self):
        with HDF5Reader(str(self.test_file)) as reader:
            groups = reader.get_groups()
            self.assertIn(S.DEFAULT_GROUP, groups)

    def test_repr(self):
        reader = HDF5Reader(str(self.test_file))
        repr_str = repr(reader)
        self.assertIn("HDF5Reader", repr_str)
        self.assertIn("group", repr_str)
        reader.close()

    def test_backward_compat_aliases(self):
        """Test that get_batches and get_batch_metadata still work."""
        reader = HDF5Reader(str(self.test_file))
        self.assertEqual(reader.get_batches(), reader.get_groups())
        meta1 = reader.get_batch_metadata(S.DEFAULT_GROUP)
        meta2 = reader.get_group_metadata(S.DEFAULT_GROUP)
        self.assertEqual(meta1, meta2)
        reader.close()


if __name__ == "__main__":
    unittest.main()
