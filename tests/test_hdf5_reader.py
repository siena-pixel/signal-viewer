"""Unit tests for HDF5 reader module."""

import unittest
import tempfile
import shutil
from unittest.mock import patch
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
            self.assertIn(S.default_group(), keys)
            mock_file.close()

    def test_mock_file_getitem(self):
        """Test accessing group from mock file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            group = mock_file[S.default_group()]
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
                self.assertIn(S.default_group(), keys)

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
            for group_name in S.group_names():
                self.assertIn(group_name, keys)
            mock_file.close()

    def test_mock_type_a_has_nsample(self):
        """Test that first group (Type A) has NSAMPLE dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            gn = S.group_names()[0]
            group = mock_file[gn]
            ds_sam = S.ds(gn, 'NSAMPLE')
            self.assertIn(ds_sam, group)
            mock_file.close()

    def test_mock_type_a_no_error_datasets(self):
        """Test that first group (Type A) has no ERROR/SQI/TLS datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            gn = S.group_names()[0]
            group = mock_file[gn]
            self.assertFalse(S.has_ds(gn, 'ERROR'))
            self.assertFalse(S.has_ds(gn, 'SQI'))
            self.assertFalse(S.has_ds(gn, 'TLS'))
            mock_file.close()

    def test_mock_type_b_has_nsample(self):
        """Test that second group (Type B) also has NSAMPLE dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            gn = S.group_names()[1]
            group = mock_file[gn]
            ds_sam = S.ds(gn, 'NSAMPLE')
            self.assertIn(ds_sam, group)
            mock_file.close()

    def test_mock_type_b_has_error_datasets(self):
        """Test that second group (Type B) has ERROR, SQI, TLS datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.h5"
            file_path.touch()

            mock_file = MockHDF5File(str(file_path), mode="r")
            gn = S.group_names()[1]
            group = mock_file[gn]
            ds_err = S.ds(gn, 'ERROR')
            ds_sqi = S.ds(gn, 'SQI')
            ds_tls = S.ds(gn, 'TLS')
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
        gn = S.default_group()
        data_dict = {
            S.ds(gn, 'VALUE'): np.random.randn(4, 1000),
            S.ds(gn, 'TIME'): np.array([0.0, 0.5, 1.0, 2.0]),
            S.ds(gn, 'SAMPLING_FREQ'): np.array([100.0, 200.0, 100.0, 500.0]),
        }
        group = MockHDF5Group(data_dict)
        keys = group.keys()
        self.assertEqual(
            set(keys),
            {
                S.ds(gn, 'VALUE'),
                S.ds(gn, 'TIME'),
                S.ds(gn, 'SAMPLING_FREQ'),
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
        self.assertIn(S.default_group(), groups)
        for gn in S.group_names():
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
        metadata = reader.get_group_metadata(S.default_group())
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
        meta = reader.get_group_metadata(S.group_names()[0])
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
        meta = reader.get_group_metadata(S.group_names()[1])
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
        group_a = S.group_names()[0]

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
        group_b = S.group_names()[1]

        # Type B also uses _SAM: signal 0 has n_sample=800
        time, signal = reader.load_signal(group_b, 0)
        self.assertEqual(len(signal), 800)
        self.assertEqual(len(time), 800)
        reader.close()

    def test_time_construction(self):
        """Test time = t0 + arange(n) * (1000/fs) for both types."""
        reader = HDF5Reader(str(self.test_file))

        # Type A, signal 0: t0=1700000000000.0 (epoch ms), fs=100.0, n=800
        time, _ = reader.load_signal(S.group_names()[0], 0)
        expected = 1700000000000.0 + np.arange(800, dtype=np.float64) * (1000.0 / 100.0)
        np.testing.assert_allclose(time, expected, rtol=1e-12)

        # Type A, signal 1: t0=1700000000000.0, fs=200.0, n=600
        time, _ = reader.load_signal(S.group_names()[0], 1)
        expected = 1700000000000.0 + np.arange(600, dtype=np.float64) * (1000.0 / 200.0)
        np.testing.assert_allclose(time, expected, rtol=1e-12)

        # Type B, signal 3: t0=1700000000000.0, fs=500.0, n=750
        time, _ = reader.load_signal(S.group_names()[1], 3)
        expected = 1700000000000.0 + np.arange(750, dtype=np.float64) * (1000.0 / 500.0)
        np.testing.assert_allclose(time, expected, rtol=1e-12)

        reader.close()

    def test_load_signal(self):
        """Test basic signal loading returns finite arrays."""
        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal(S.default_group(), 0)
        self.assertGreater(len(time), 0)
        self.assertEqual(len(time), len(signal))
        self.assertTrue(np.all(np.isfinite(time)))
        self.assertTrue(np.all(np.isfinite(signal)))
        reader.close()

    def test_load_signal_invalid_index(self):
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.load_signal(S.default_group(), 999)
        reader.close()

    def test_load_signal_by_name(self):
        reader = HDF5Reader(str(self.test_file))
        time, signal = reader.load_signal_by_name(S.default_group(), "position")
        self.assertGreater(len(time), 0)
        self.assertEqual(len(time), len(signal))
        reader.close()

    def test_load_signal_by_name_not_found(self):
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(ValueError):
            reader.load_signal_by_name(S.default_group(), "nonexistent")
        reader.close()

    def test_get_signal_stats(self):
        reader = HDF5Reader(str(self.test_file))
        stats = reader.get_signal_stats(S.default_group(), 0)
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
        stats = reader.get_signal_stats(S.group_names()[1], 0)
        # Type B also has _SAM: signal 0 → 800 valid samples
        self.assertEqual(stats["samples"], 800)
        reader.close()

    def test_get_signal_stats_invalid_index(self):
        reader = HDF5Reader(str(self.test_file))
        with self.assertRaises(IndexError):
            reader.get_signal_stats(S.default_group(), 999)
        reader.close()

    def test_context_manager(self):
        with HDF5Reader(str(self.test_file)) as reader:
            groups = reader.get_groups()
            self.assertIn(S.default_group(), groups)

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
        meta1 = reader.get_batch_metadata(S.default_group())
        meta2 = reader.get_group_metadata(S.default_group())
        self.assertEqual(meta1, meta2)
        reader.close()


# ---------------------------------------------------------------------------
# Fallback / optional dataset tests
# ---------------------------------------------------------------------------

class _CustomMockFile:
    """Minimal mock HDF5 file with caller-supplied group data."""

    def __init__(self, groups_dict):
        self._groups = groups_dict
        self._closed = False

    def keys(self):
        if self._closed:
            raise ValueError("closed")
        return list(self._groups.keys())

    def __getitem__(self, key):
        if self._closed:
            raise ValueError("closed")
        if key not in self._groups:
            raise KeyError(key)
        return MockHDF5Group(self._groups[key])

    def __contains__(self, key):
        return key in self._groups

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _make_reader_with_mock(groups_dict, tmpdir):
    """Create an HDF5Reader whose _open_file returns a _CustomMockFile."""
    fp = Path(tmpdir) / "custom.h5"
    fp.touch()
    reader = object.__new__(HDF5Reader)
    reader.file_path = fp
    reader._lock = __import__('threading').Lock()
    reader._file_handle = None
    reader._groups_cache = None
    reader._metadata_cache = {}
    reader._open_file = lambda: _CustomMockFile(groups_dict)
    return reader


class TestOptionalNsample(unittest.TestCase):
    """NSAMPLE missing → full sample count used."""

    def test_no_nsample_uses_full_length(self):
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(2, 500).astype(np.float64),
            ds_map.get('TIME', '_T'): np.array([0.0, 0.0]),
            ds_map.get('SAMPLING_FREQ', '_F'): np.array([100.0, 100.0]),
            ds_map.get('NAMES', '_N'): np.array(["sig_a", "sig_b"], dtype=object),
            ds_map.get('UNITS', '_U'): np.array(["V", "A"], dtype=object),
            # NSAMPLE intentionally omitted
        }
        # Remove placeholder keys that don't match actual ds_map entries
        datasets = {k: v for k, v in datasets.items()
                    if k in [ds_map.get(key) for key in ds_map]}

        # Rebuild with only VALUE, TIME, SAMPLING_FREQ, NAMES, UNITS (no NSAMPLE)
        datasets = {}
        datasets[ds_map['VALUE']] = rng.randn(2, 500).astype(np.float64)
        if 'TIME' in ds_map:
            datasets[ds_map['TIME']] = np.array([0.0, 0.0])
        if 'SAMPLING_FREQ' in ds_map:
            datasets[ds_map['SAMPLING_FREQ']] = np.array([100.0, 100.0])
        if 'NAMES' in ds_map:
            datasets[ds_map['NAMES']] = np.array(["sig_a", "sig_b"], dtype=object)
        if 'UNITS' in ds_map:
            datasets[ds_map['UNITS']] = np.array(["V", "A"], dtype=object)

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            meta = reader.get_group_metadata(gn)
            self.assertEqual(meta["n_samples"], [500, 500])
            time, sig = reader.load_signal(gn, 0)
            self.assertEqual(len(sig), 500)
            self.assertEqual(len(time), 500)


class TestOptionalTime(unittest.TestCase):
    """TIME missing → t0 = 0.0."""

    def test_no_time_defaults_to_zero(self):
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(2, 100).astype(np.float64),
            # TIME intentionally omitted
        }
        if 'SAMPLING_FREQ' in ds_map:
            datasets[ds_map['SAMPLING_FREQ']] = np.array([50.0, 50.0])
        if 'NAMES' in ds_map:
            datasets[ds_map['NAMES']] = np.array(["a", "b"], dtype=object)

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            time, sig = reader.load_signal(gn, 0)
            # t0 = 0.0, fs = 50 → step = 20 ms
            expected = np.arange(100, dtype=np.float64) * (1000.0 / 50.0)
            np.testing.assert_allclose(time, expected, rtol=1e-12)


class TestOptionalSamplingFreq(unittest.TestCase):
    """SAMPLING_FREQ missing → parse from VALUE name or default 1.0."""

    def test_freq_from_value_name_suffix(self):
        """VALUE name ending in _0050 → 50 Hz."""
        rng = np.random.RandomState(99)

        # Use a custom config where VALUE name has freq suffix
        custom_ds = {
            'VALUE': 'MY_GRP_V_0050',
            'NAMES': 'MY_GRP_N',
            # No SAMPLING_FREQ, no TIME, no NSAMPLE
        }
        datasets = {
            'MY_GRP_V_0050': rng.randn(1, 200).astype(np.float64),
            'MY_GRP_N': np.array(["test_sig"], dtype=object),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(S.GROUP_DS_NAMES, {'MY_GRP': custom_ds}):
                reader = _make_reader_with_mock({'MY_GRP': datasets}, tmpdir)
                time, sig = reader.load_signal('MY_GRP', 0)
                # fs = 50 → step = 20 ms, t0 = 0
                expected = np.arange(200, dtype=np.float64) * (1000.0 / 50.0)
                np.testing.assert_allclose(time, expected, rtol=1e-12)

    def test_freq_default_when_no_suffix(self):
        """VALUE name without freq suffix → default 1.0 Hz."""
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(1, 50).astype(np.float64),
            # No SAMPLING_FREQ, no TIME
        }
        if 'NAMES' in ds_map:
            datasets[ds_map['NAMES']] = np.array(["x"], dtype=object)

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            time, sig = reader.load_signal(gn, 0)
            # fs = 1.0 → step = 1000 ms, t0 = 0
            expected = np.arange(50, dtype=np.float64) * 1000.0
            np.testing.assert_allclose(time, expected, rtol=1e-12)


class TestRequiredNames(unittest.TestCase):
    """NAMES is required — missing or blank entries make group invalid."""

    def test_missing_names_excluded_from_groups(self):
        """Group without NAMES dataset is excluded from get_groups()."""
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(3, 100).astype(np.float64),
            # NAMES intentionally omitted
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            groups = reader.get_groups()
            self.assertNotIn(gn, groups)

    def test_missing_names_raises_in_metadata(self):
        """get_group_metadata raises ValueError when NAMES is absent."""
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(2, 100).astype(np.float64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            with self.assertRaises(ValueError):
                reader.get_group_metadata(gn)

    def test_blank_names_excluded_from_groups(self):
        """Group with blank entries in NAMES is excluded from get_groups()."""
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(3, 100).astype(np.float64),
        }
        if 'NAMES' in ds_map:
            datasets[ds_map['NAMES']] = np.array(
                ["position", "", "voltage"], dtype=object
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            groups = reader.get_groups()
            self.assertNotIn(gn, groups)

    def test_blank_names_raises_in_metadata(self):
        """get_group_metadata raises ValueError when NAMES has blanks."""
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(2, 100).astype(np.float64),
        }
        if 'NAMES' in ds_map:
            datasets[ds_map['NAMES']] = np.array(
                ["position", "  "], dtype=object
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            with self.assertRaises(ValueError):
                reader.get_group_metadata(gn)

    def test_bytes_names_decoded_and_stripped(self):
        """Bytes NAMES entries are decoded to str and stripped."""
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(2, 100).astype(np.float64),
        }
        if 'NAMES' in ds_map:
            datasets[ds_map['NAMES']] = np.array(
                [b"  pressure  ", b"voltage"], dtype=object
            )
        if 'SAMPLING_FREQ' in ds_map:
            datasets[ds_map['SAMPLING_FREQ']] = np.array([10.0, 10.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            meta = reader.get_group_metadata(gn)
            self.assertEqual(meta["signal_names"][0], "pressure")
            self.assertEqual(meta["signal_names"][1], "voltage")


class TestEmptyGroupHandling(unittest.TestCase):
    """Empty groups are excluded from get_groups()."""

    def test_empty_value_dataset_excluded(self):
        """Group with VALUE shape (0, N) is excluded."""
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: np.empty((0, 100), dtype=np.float64),
        }
        if 'NAMES' in ds_map:
            datasets[ds_map['NAMES']] = np.array([], dtype=object)

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            groups = reader.get_groups()
            self.assertNotIn(gn, groups)

    def test_zero_samples_excluded(self):
        """Group with VALUE shape (M, 0) is excluded."""
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: np.empty((4, 0), dtype=np.float64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            groups = reader.get_groups()
            self.assertNotIn(gn, groups)

    def test_missing_value_dataset_excluded(self):
        """Group present but no VALUE dataset → excluded."""
        gn = S.group_names()[0]
        # Group exists but has no VALUE dataset
        datasets = {"some_other_ds": np.array([1, 2, 3])}

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            groups = reader.get_groups()
            self.assertNotIn(gn, groups)

    def test_valid_group_alongside_empty(self):
        """One valid group, one empty → only valid returned."""
        names = S.group_names()
        if len(names) < 2:
            self.skipTest("Need at least 2 configured groups")

        gn_valid = names[0]
        gn_empty = names[1]
        ds_valid = S.GROUP_DS_NAMES[gn_valid]
        ds_empty = S.GROUP_DS_NAMES[gn_empty]

        rng = np.random.RandomState(99)
        groups = {
            gn_valid: {
                ds_valid['VALUE']: rng.randn(2, 100).astype(np.float64),
            },
            gn_empty: {
                ds_empty['VALUE']: np.empty((0, 100), dtype=np.float64),
            },
        }
        if 'NAMES' in ds_valid:
            groups[gn_valid][ds_valid['NAMES']] = np.array(["a", "b"], dtype=object)

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock(groups, tmpdir)
            result = reader.get_groups()
            self.assertIn(gn_valid, result)
            self.assertNotIn(gn_empty, result)


class TestMinimalGroup(unittest.TestCase):
    """Group with VALUE + NAMES only — all optional datasets missing."""

    def test_minimal_group_metadata(self):
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(3, 200).astype(np.float64),
            ds_map['NAMES']: np.array(["sig_a", "sig_b", "sig_c"], dtype=object),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            meta = reader.get_group_metadata(gn)
            self.assertEqual(meta["signal_count"], 3)
            self.assertEqual(meta["sample_count"], 200)
            self.assertEqual(meta["signal_names"],
                             ["sig_a", "sig_b", "sig_c"])
            self.assertEqual(meta["units"], ["", "", ""])
            self.assertEqual(meta["n_samples"], [200, 200, 200])

    def test_minimal_group_load_signal(self):
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(2, 150).astype(np.float64),
            ds_map['NAMES']: np.array(["x", "y"], dtype=object),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            time, sig = reader.load_signal(gn, 0)
            self.assertEqual(len(sig), 150)
            self.assertEqual(len(time), 150)
            # t0=0, fs=1.0 (default) → step = 1000 ms
            expected = np.arange(150, dtype=np.float64) * 1000.0
            np.testing.assert_allclose(time, expected, rtol=1e-12)

    def test_minimal_group_stats(self):
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(2, 150).astype(np.float64),
            ds_map['NAMES']: np.array(["x", "y"], dtype=object),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            stats = reader.get_signal_stats(gn, 0)
            self.assertEqual(stats["samples"], 150)
            self.assertTrue(np.isfinite(stats["mean"]))


class TestTimeFallback(unittest.TestCase):
    """Per-group TIME_FALLBACK — borrow t0 from same-named signal in another group."""

    def test_fallback_provides_time(self):
        """Signal in group without TIME gets t0 from its TIME_FALLBACK group."""
        rng = np.random.RandomState(99)
        names = S.group_names()
        if len(names) < 2:
            self.skipTest("Need at least 2 configured groups")

        gn_fb = names[0]   # fallback group (has TIME)
        gn_target = names[1]  # target group (no TIME, TIME_FALLBACK → gn_fb)
        ds_fb = S.GROUP_DS_NAMES[gn_fb]
        ds_target = S.GROUP_DS_NAMES[gn_target]

        groups = {
            gn_fb: {
                ds_fb['VALUE']: rng.randn(2, 100).astype(np.float64),
                ds_fb['NAMES']: np.array(["alpha", "beta"], dtype=object),
                ds_fb['TIME']: np.array([1000.0, 2000.0]),
                ds_fb['SAMPLING_FREQ']: np.array([100.0, 100.0]),
            },
            gn_target: {
                ds_target['VALUE']: rng.randn(2, 80).astype(np.float64),
                ds_target['NAMES']: np.array(["alpha", "beta"], dtype=object),
                ds_target['SAMPLING_FREQ']: np.array([50.0, 50.0]),
                # TIME intentionally omitted
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock(groups, tmpdir)
            # Inject TIME_FALLBACK into the target group's config
            with patch.dict(S.GROUP_DS_NAMES[gn_target],
                            {'TIME_FALLBACK': gn_fb}):
                time, sig = reader.load_signal(gn_target, 0)
                # t0 should come from gn_fb's TIME for "alpha" → 1000.0
                expected = 1000.0 + np.arange(80, dtype=np.float64) * (1000.0 / 50.0)
                np.testing.assert_allclose(time, expected, rtol=1e-12)

    def test_fallback_name_not_found_defaults_zero(self):
        """If signal name not in fallback group, t0 = 0.0."""
        rng = np.random.RandomState(99)
        names = S.group_names()
        if len(names) < 2:
            self.skipTest("Need at least 2 configured groups")

        gn_fb = names[0]
        gn_target = names[1]
        ds_fb = S.GROUP_DS_NAMES[gn_fb]
        ds_target = S.GROUP_DS_NAMES[gn_target]

        groups = {
            gn_fb: {
                ds_fb['VALUE']: rng.randn(1, 100).astype(np.float64),
                ds_fb['NAMES']: np.array(["alpha"], dtype=object),
                ds_fb['TIME']: np.array([5000.0]),
                ds_fb['SAMPLING_FREQ']: np.array([100.0]),
            },
            gn_target: {
                ds_target['VALUE']: rng.randn(1, 60).astype(np.float64),
                ds_target['NAMES']: np.array(["gamma"], dtype=object),
                ds_target['SAMPLING_FREQ']: np.array([25.0]),
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock(groups, tmpdir)
            with patch.dict(S.GROUP_DS_NAMES[gn_target],
                            {'TIME_FALLBACK': gn_fb}):
                time, sig = reader.load_signal(gn_target, 0)
                # "gamma" not in fallback → t0 = 0.0
                expected = np.arange(60, dtype=np.float64) * (1000.0 / 25.0)
                np.testing.assert_allclose(time, expected, rtol=1e-12)

    def test_no_fallback_configured(self):
        """When TIME_FALLBACK is absent from group config, t0 = 0.0."""
        rng = np.random.RandomState(99)
        gn = S.group_names()[0]
        ds_map = S.GROUP_DS_NAMES[gn]

        datasets = {
            ds_map['VALUE']: rng.randn(1, 50).astype(np.float64),
            ds_map['NAMES']: np.array(["x"], dtype=object),
            ds_map['SAMPLING_FREQ']: np.array([10.0]),
            # TIME intentionally omitted; no TIME_FALLBACK in default config
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            reader = _make_reader_with_mock({gn: datasets}, tmpdir)
            time, sig = reader.load_signal(gn, 0)
            expected = np.arange(50, dtype=np.float64) * (1000.0 / 10.0)
            np.testing.assert_allclose(time, expected, rtol=1e-12)

    def test_each_group_can_have_different_fallback(self):
        """Two groups with different TIME_FALLBACK targets."""
        rng = np.random.RandomState(99)
        names = S.group_names()
        if len(names) < 2:
            self.skipTest("Need at least 2 configured groups")

        gn_a = names[0]
        gn_b = names[1]
        ds_a = S.GROUP_DS_NAMES[gn_a]
        ds_b = S.GROUP_DS_NAMES[gn_b]

        groups = {
            gn_a: {
                ds_a['VALUE']: rng.randn(1, 40).astype(np.float64),
                ds_a['NAMES']: np.array(["sig1"], dtype=object),
                ds_a['TIME']: np.array([100.0]),
                ds_a['SAMPLING_FREQ']: np.array([10.0]),
            },
            gn_b: {
                ds_b['VALUE']: rng.randn(1, 40).astype(np.float64),
                ds_b['NAMES']: np.array(["sig1"], dtype=object),
                ds_b['TIME']: np.array([200.0]),
                ds_b['SAMPLING_FREQ']: np.array([10.0]),
            },
        }

        # Custom groups that use each other as fallback
        custom_c = {
            'VALUE': 'C_V', 'NAMES': 'C_N', 'SAMPLING_FREQ': 'C_F',
            'TIME_FALLBACK': gn_a,
        }
        custom_d = {
            'VALUE': 'D_V', 'NAMES': 'D_N', 'SAMPLING_FREQ': 'D_F',
            'TIME_FALLBACK': gn_b,
        }
        groups['GRP_C'] = {
            'C_V': rng.randn(1, 30).astype(np.float64),
            'C_N': np.array(["sig1"], dtype=object),
            'C_F': np.array([20.0]),
        }
        groups['GRP_D'] = {
            'D_V': rng.randn(1, 30).astype(np.float64),
            'D_N': np.array(["sig1"], dtype=object),
            'D_F': np.array([20.0]),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(S.GROUP_DS_NAMES,
                            {'GRP_C': custom_c, 'GRP_D': custom_d}):
                reader = _make_reader_with_mock(groups, tmpdir)

                # GRP_C falls back to gn_a → t0 = 100.0
                time_c, _ = reader.load_signal('GRP_C', 0)
                expected_c = 100.0 + np.arange(30, dtype=np.float64) * (1000.0 / 20.0)
                np.testing.assert_allclose(time_c, expected_c, rtol=1e-12)

                # GRP_D falls back to gn_b → t0 = 200.0
                reader._metadata_cache.clear()
                time_d, _ = reader.load_signal('GRP_D', 0)
                expected_d = 200.0 + np.arange(30, dtype=np.float64) * (1000.0 / 20.0)
                np.testing.assert_allclose(time_d, expected_d, rtol=1e-12)


class TestParseFreqFromName(unittest.TestCase):
    """Test HDF5Schema.parse_freq_from_name()."""

    def test_standard_suffix(self):
        self.assertEqual(S.parse_freq_from_name('GROUP_T1_V_0050'), 50.0)

    def test_large_freq(self):
        self.assertEqual(S.parse_freq_from_name('GRP_V_10000'), 10000.0)

    def test_no_suffix(self):
        self.assertIsNone(S.parse_freq_from_name('GROUP_T0_V'))

    def test_trailing_text(self):
        self.assertIsNone(S.parse_freq_from_name('GROUP_V_abc'))

    def test_zero_padded(self):
        self.assertEqual(S.parse_freq_from_name('SIG_V_0001'), 1.0)


if __name__ == "__main__":
    unittest.main()
