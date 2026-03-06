"""
Integration tests for Engineering Time Series Signal Viewer.

Tests the Tornado web server, API endpoints, and data handling using
tornado.testing.AsyncHTTPTestCase. Includes:
  - Server startup and page rendering
  - API endpoints for metadata cascading (serials, steps, files, batches)
  - Signal loading and downsampling
  - Analysis endpoints (stats, correlation, trend, etc.)
  - Error handling (404, 400 responses)
  - Cache statistics endpoint

When h5py is not installed, tests use MockHDF5File for full coverage.
"""

import base64
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict
from urllib.parse import quote

import numpy as np
import tornado.testing
import tornado.ioloop

from signal_viewer import config
from signal_viewer.config import HDF5Schema as S
from signal_viewer.core.hdf5_reader import create_test_file
from signal_viewer.server.app import make_app


class IntegrationTestBase(tornado.testing.AsyncHTTPTestCase):
    """Base class for integration tests with test data setup."""

    def setUp(self):
        """Set up test server and temporary data directory."""
        # Save original config
        self.original_data_root = config.DATA_ROOT

        # Create temporary data root
        self.test_data_dir = tempfile.mkdtemp()

        # Structure: serial / folder_1 / folder_2 / file.h5
        self.serials = ["SN001", "SN002"]
        self.folder1s = ["p001_test"]
        self.folder2s = ["run_1", "run_2"]
        self.file_paths = {}

        for serial in self.serials:
            self.file_paths[serial] = {}
            for f1 in self.folder1s:
                self.file_paths[serial][f1] = {}
                for f2 in self.folder2s:
                    dir_path = Path(self.test_data_dir) / serial / f1 / f2
                    dir_path.mkdir(parents=True)
                    file_path = dir_path / "data.h5"
                    create_test_file(str(file_path))
                    self.file_paths[serial][f1][f2] = str(file_path)

        # Configure app to use temp directory BEFORE parent setUp creates app
        config.DATA_ROOT = Path(self.test_data_dir)

        # Call parent setUp (starts server)
        try:
            super().setUp()
        except Exception:
            # Restore config on error
            config.DATA_ROOT = self.original_data_root
            raise

    def tearDown(self):
        """Clean up test server and temporary data."""
        super().tearDown()

        # Restore original config
        config.DATA_ROOT = self.original_data_root

        # Clean up temp directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def get_app(self):
        """Create and return the Tornado application for testing."""
        return make_app()

    def encode_file_path(self, file_path: str) -> str:
        """Encode file path as base64 for API calls."""
        return base64.urlsafe_b64encode(file_path.encode()).decode()

    def get_json(self, path: str, **kwargs) -> Dict:
        """Make GET request and parse JSON response."""
        response = self.fetch(path, **kwargs)
        return json.loads(response.body.decode())

    def post_json(self, path: str, data: Dict, **kwargs) -> Dict:
        """Make POST request with JSON body and parse response."""
        kwargs['body'] = json.dumps(data)
        kwargs['method'] = 'POST'
        response = self.fetch(path, **kwargs)
        return json.loads(response.body.decode())


class TestPageHandlers(IntegrationTestBase):
    """Test page rendering endpoints."""

    def test_root_page(self):
        """Test GET / returns viewer page (home)."""
        response = self.fetch('/')
        self.assertEqual(response.code, 200)
        self.assertIn('text/html', response.headers.get('Content-Type', ''))

    def test_analysis_page(self):
        """Test GET /analysis returns analysis page."""
        response = self.fetch('/analysis')
        self.assertEqual(response.code, 200)
        self.assertIn('text/html', response.headers.get('Content-Type', ''))

    def test_comparison_page(self):
        """Test GET /comparison returns comparison page."""
        response = self.fetch('/comparison')
        self.assertEqual(response.code, 200)
        self.assertIn('text/html', response.headers.get('Content-Type', ''))

    def test_docs_page(self):
        """Test GET /docs returns documentation page."""
        response = self.fetch('/docs')
        self.assertEqual(response.code, 200)
        self.assertIn('text/html', response.headers.get('Content-Type', ''))


class TestSerialsCascading(IntegrationTestBase):
    """Test cascading API endpoints for metadata."""

    def test_get_serials(self):
        """Test GET /api/serials returns serial numbers."""
        response = self.get_json('/api/serials')

        self.assertIn('serials', response)
        self.assertEqual(len(response['serials']), 2)
        self.assertIn('SN001', response['serials'])
        self.assertIn('SN002', response['serials'])

    def test_get_steps_for_serial(self):
        """Test GET /api/serials/{serial}/steps returns folder_1/folder_2 steps."""
        response = self.get_json('/api/serials/SN001/steps')

        self.assertIn('steps', response)
        # 1 folder_1 × 2 folder_2s = 2 step combinations
        self.assertEqual(len(response['steps']), 2)
        self.assertIn('p001_test/run_1', response['steps'])
        self.assertIn('p001_test/run_2', response['steps'])

    def test_get_steps_for_missing_serial(self):
        """Test GET /api/serials/{serial}/steps with missing serial returns 404."""
        response = self.fetch('/api/serials/MISSING/steps')
        self.assertEqual(response.code, 404)

    def test_get_files_for_step(self):
        """Test GET /api/serials/{serial}/steps/{step}/files returns files."""
        step = quote('p001_test/run_1', safe='')
        response = self.get_json(f'/api/serials/SN001/steps/{step}/files')

        self.assertIn('files', response)
        self.assertEqual(len(response['files']), 1)
        # Files are returned as dicts with metadata
        self.assertIsInstance(response['files'][0], dict)
        self.assertIn('path', response['files'][0])

    def test_get_files_for_missing_step(self):
        """Test GET /api/serials/{serial}/steps/{step}/files with invalid step."""
        response = self.fetch('/api/serials/SN001/steps/missing_step/files')
        self.assertEqual(response.code, 404)


class TestBatchesAndSignals(IntegrationTestBase):
    """Test batch and signal loading endpoints."""

    def test_get_batches(self):
        """Test GET /api/files/{encoded_path}/batches returns batches."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        response = self.get_json(f'/api/files/{encoded}/batches')

        self.assertIn('batches', response)
        self.assertIn(S.DEFAULT_GROUP, response['batches'])

    def test_get_batch_metadata(self):
        """Test GET /api/files/{encoded}/batches/{batch}/meta returns metadata."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        response = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/meta'
        )

        self.assertIn('signal_names', response)
        self.assertIn('units', response)
        self.assertEqual(len(response['signal_names']), 4)
        self.assertEqual(len(response['units']), 4)

    def test_get_signal(self):
        """Test GET /api/files/{encoded}/batches/{batch}/signals/{idx} loads signal."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        response = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
        )

        self.assertIn('time', response)
        self.assertIn('values', response)
        self.assertIn('name', response)
        self.assertIn('units', response)
        self.assertIn('samples', response)
        # DEFAULT_GROUP (GROUP_T0) is Type A: signal 0 has n_samples=800
        self.assertEqual(response['samples'], 800)
        self.assertEqual(len(response['time']), 800)
        self.assertEqual(len(response['values']), 800)

    def test_get_signal_with_downsampling(self):
        """Test signal downsampling with downsample query parameter."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        response = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0?downsample=100'
        )

        self.assertIn('time', response)
        self.assertIn('values', response)
        # Downsampled to ~100 points
        self.assertLess(len(response['values']), 200)

    def test_get_signal_with_time_window(self):
        """Test signal loading with t_min/t_max time window parameters."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        # First load full signal to get time extent
        full = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
        )
        t_start = full['t_start']
        t_end = full['t_end']
        total = full['total_samples']

        # Request a ~50% window in the middle
        mid = (t_start + t_end) / 2
        quarter = (t_end - t_start) / 4
        t_min = mid - quarter
        t_max = mid + quarter

        windowed = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
            f'?t_min={t_min}&t_max={t_max}'
        )

        self.assertIn('time', windowed)
        self.assertIn('values', windowed)
        self.assertTrue(windowed['windowed'])
        # Windowed result should have fewer samples than full signal
        self.assertLess(len(windowed['values']), total)
        # total_samples should still report the FULL signal length
        self.assertEqual(windowed['total_samples'], total)

    def test_get_signal_windowed_with_downsampling(self):
        """Test time-windowed signal with additional downsampling."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        full = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
        )
        t_start = full['t_start']
        t_end = full['t_end']

        windowed = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
            f'?t_min={t_start}&t_max={t_end}&downsample=50'
        )

        self.assertIn('time', windowed)
        self.assertIn('values', windowed)
        self.assertEqual(len(windowed['time']), len(windowed['values']))
        # Should be downsampled to roughly 50 points
        self.assertLessEqual(len(windowed['values']), 100)

    def test_get_signal_invalid_index(self):
        """Test GET /api/files/{encoded}/batches/{batch}/signals/{idx} with bad index."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        response = self.fetch(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/999'
        )
        self.assertEqual(response.code, 404)

    def test_get_signal_invalid_file_path(self):
        """Test /api/files/{encoded}/batches with invalid encoded path."""
        bad_encoded = base64.urlsafe_b64encode(b'/nonexistent/path.h5').decode()
        response = self.fetch(f'/api/files/{bad_encoded}/batches')
        self.assertEqual(response.code, 400)


class TestAnalysisEndpoints(IntegrationTestBase):
    """Test analysis API endpoints."""

    def setUp(self):
        """Set up test data and get file info."""
        super().setUp()
        self.file_path = self.file_paths['SN001']['p001_test']['run_1']

    def test_stats_analysis(self):
        """Test POST /api/analysis/stats computes signal statistics."""
        payload = {
            'file_path': self.file_path,
            'batch': S.DEFAULT_GROUP,
            'signal_idx': 0,
        }

        response = self.post_json('/api/analysis/stats', payload)

        expected_stats = [
            'count',
            'mean',
            'std',
            'min',
            'max',
            'median',
        ]
        for stat in expected_stats:
            self.assertIn(stat, response)

    def test_trend_analysis(self):
        """Test POST /api/analysis/trend fits polynomial trend."""
        payload = {
            'file_path': self.file_path,
            'batch': S.DEFAULT_GROUP,
            'signal_idx': 0,
            'degree': 1,
        }

        response = self.post_json('/api/analysis/trend', payload)

        self.assertIn('coefficients', response)
        self.assertIn('fitted', response)
        self.assertIn('residuals', response)
        self.assertIn('time', response)
        # DEFAULT_GROUP (GROUP_T0) is Type A: signal 0 has n_sample=800
        self.assertEqual(len(response['fitted']), 800)
        self.assertEqual(len(response['residuals']), 800)

    def test_correlation_analysis(self):
        """Test POST /api/analysis/correlation computes cross-correlation."""
        payload = {
            'file_path_a': self.file_path,
            'batch_a': S.DEFAULT_GROUP,
            'signal_idx_a': 0,
            'file_path_b': self.file_path,
            'batch_b': S.DEFAULT_GROUP,
            'signal_idx_b': 1,
        }

        response = self.post_json('/api/analysis/correlation', payload)

        self.assertIn('lags', response)
        self.assertIn('correlation', response)
        self.assertIn('max_lag', response)
        self.assertEqual(
            len(response['lags']), len(response['correlation'])
        )

    def test_missing_required_fields(self):
        """Test analysis endpoints return 400 for missing required fields."""
        payload = {
            'batch': S.DEFAULT_GROUP,
            'signal_idx': 0,
        }

        response = self.fetch(
            '/api/analysis/stats',
            body=json.dumps(payload),
            method='POST',
        )
        self.assertEqual(response.code, 400)


class TestCacheAndRescan(IntegrationTestBase):
    """Test cache and rescan endpoints."""

    def test_cache_stats(self):
        """Test GET /api/cache/stats returns cache statistics."""
        # Load a signal to populate cache
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)
        self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
        )

        response = self.get_json('/api/cache/stats')

        self.assertIn('hits', response)
        self.assertIn('misses', response)
        self.assertIn('entries', response)
        self.assertIn('memory_used', response)
        self.assertIn('memory_budget', response)

    def test_rescan_endpoint(self):
        """Test POST /api/rescan re-scans filesystem."""
        payload = {}
        response = self.post_json('/api/rescan', payload)

        self.assertIn('status', response)
        self.assertEqual(response['status'], 'success')
        self.assertIn('serial_count', response)
        self.assertIn('total_steps', response)
        self.assertEqual(response['serial_count'], 2)
        # 2 serials × 1 folder_1 × 2 folder_2s = 4 total steps
        self.assertEqual(response['total_steps'], 4)


class TestErrorHandling(IntegrationTestBase):
    """Test error handling and edge cases."""

    def test_cors_headers(self):
        """Test CORS headers are set in API responses."""
        response = self.fetch('/api/serials')

        self.assertEqual(
            response.headers.get('Access-Control-Allow-Origin'), '*'
        )
        self.assertIn(
            'GET',
            response.headers.get('Access-Control-Allow-Methods'),
        )

    def test_options_request(self):
        """Test OPTIONS requests for CORS preflight."""
        response = self.fetch('/api/serials', method='OPTIONS')
        self.assertEqual(response.code, 204)

    def test_invalid_json_in_post(self):
        """Test POST endpoint with invalid JSON."""
        response = self.fetch(
            '/api/analysis/stats',
            body='invalid json {',
            method='POST',
        )
        self.assertEqual(response.code, 500)

    def test_signal_caching(self):
        """Test that signals are cached after first load."""
        file_path = self.file_paths['SN001']['p001_test']['run_1']
        encoded = self.encode_file_path(file_path)

        # First load
        response1 = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
        )

        # Get cache stats
        stats1 = self.get_json('/api/cache/stats')
        initial_entries = stats1['entries']

        # Second load (should hit cache)
        response2 = self.get_json(
            f'/api/files/{encoded}/batches/{S.DEFAULT_GROUP}/signals/0'
        )

        stats2 = self.get_json('/api/cache/stats')

        # Cache should still have same number of entries
        self.assertEqual(stats2['entries'], initial_entries)
        # Data should be identical
        self.assertEqual(response1['values'], response2['values'])


if __name__ == '__main__':
    unittest.main()
