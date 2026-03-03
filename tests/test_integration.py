"""
Integration tests for Engineering Time Series Signal Viewer.

Tests the Tornado web server, API endpoints, and data handling using
tornado.testing.AsyncHTTPTestCase. Includes:
  - Server startup and page rendering
  - API endpoints for metadata cascading (serials, steps, files, batches)
  - Signal loading and downsampling
  - Analysis endpoints (FFT, PSD, filter, anomaly, stats, etc.)
  - Error handling (404, 400 responses)
  - Cache statistics endpoint

NOTE: These tests require h5py to be installed. If h5py is not available,
tests will be skipped. Install with: pip install h5py
"""

import base64
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Dict

import numpy as np
import tornado.testing
import tornado.ioloop

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from signal_viewer import config
from signal_viewer.core.hdf5_reader import create_test_file
from signal_viewer.server.app import make_app


@unittest.skipIf(not HAS_H5PY, "h5py not installed - integration tests require h5py")
class IntegrationTestBase(tornado.testing.AsyncHTTPTestCase):
    """Base class for integration tests with test data setup."""

    def setUp(self):
        """Set up test server and temporary data directory."""
        # Save original config
        self.original_data_root = config.DATA_ROOT

        # Create temporary data root
        self.test_data_dir = tempfile.mkdtemp()

        # Create test data structure: SN###/step_N/file.h5
        self.serials = ["SN001", "SN002"]
        self.steps = [1, 2]
        self.file_paths = {}

        for serial in self.serials:
            serial_dir = Path(self.test_data_dir) / serial
            serial_dir.mkdir(parents=True)
            self.file_paths[serial] = {}

            for step in self.steps:
                step_dir = serial_dir / f"step_{step}"
                step_dir.mkdir(parents=True)

                # Create test HDF5 file
                file_path = step_dir / "data.h5"
                create_test_file(str(file_path))
                self.file_paths[serial][step] = str(file_path)

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

    def test_anomaly_page(self):
        """Test GET /anomaly returns anomaly page."""
        response = self.fetch('/anomaly')
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
        """Test GET /api/serials/{serial}/steps returns steps."""
        response = self.get_json('/api/serials/SN001/steps')

        self.assertIn('steps', response)
        self.assertEqual(len(response['steps']), 2)
        self.assertIn(1, response['steps'])
        self.assertIn(2, response['steps'])

    def test_get_steps_for_missing_serial(self):
        """Test GET /api/serials/{serial}/steps with missing serial returns 404."""
        response = self.fetch('/api/serials/MISSING/steps')
        self.assertEqual(response.code, 404)

    def test_get_files_for_step(self):
        """Test GET /api/serials/{serial}/steps/{step}/files returns files."""
        response = self.get_json('/api/serials/SN001/steps/1/files')

        self.assertIn('files', response)
        self.assertEqual(len(response['files']), 1)
        # Files are returned as dicts with metadata
        self.assertIsInstance(response['files'][0], dict)
        self.assertIn('path', response['files'][0])

    def test_get_files_for_missing_step(self):
        """Test GET /api/serials/{serial}/steps/{step}/files with invalid step."""
        response = self.fetch('/api/serials/SN001/steps/999/files')
        self.assertEqual(response.code, 404)


class TestBatchesAndSignals(IntegrationTestBase):
    """Test batch and signal loading endpoints."""

    def test_get_batches(self):
        """Test GET /api/files/{encoded_path}/batches returns batches."""
        file_path = self.file_paths['SN001'][1]
        encoded = self.encode_file_path(file_path)

        response = self.get_json(f'/api/files/{encoded}/batches')

        self.assertIn('batches', response)
        self.assertEqual(len(response['batches']), 1)
        self.assertIn('batch_001', response['batches'])

    def test_get_batch_metadata(self):
        """Test GET /api/files/{encoded}/batches/{batch}/meta returns metadata."""
        file_path = self.file_paths['SN001'][1]
        encoded = self.encode_file_path(file_path)

        response = self.get_json(
            f'/api/files/{encoded}/batches/batch_001/meta'
        )

        self.assertIn('signal_names', response)
        self.assertIn('units', response)
        self.assertEqual(len(response['signal_names']), 4)
        self.assertEqual(len(response['units']), 4)

    def test_get_signal(self):
        """Test GET /api/files/{encoded}/batches/{batch}/signals/{idx} loads signal."""
        file_path = self.file_paths['SN001'][1]
        encoded = self.encode_file_path(file_path)

        response = self.get_json(
            f'/api/files/{encoded}/batches/batch_001/signals/0'
        )

        self.assertIn('time', response)
        self.assertIn('values', response)
        self.assertIn('name', response)
        self.assertIn('units', response)
        self.assertIn('samples', response)
        self.assertEqual(response['samples'], 1000)
        self.assertEqual(len(response['time']), 1000)
        self.assertEqual(len(response['values']), 1000)

    def test_get_signal_with_downsampling(self):
        """Test signal downsampling with downsample query parameter."""
        file_path = self.file_paths['SN001'][1]
        encoded = self.encode_file_path(file_path)

        response = self.get_json(
            f'/api/files/{encoded}/batches/batch_001/signals/0?downsample=100'
        )

        self.assertIn('time', response)
        self.assertIn('values', response)
        # Downsampled to ~100 points
        self.assertLess(len(response['values']), 200)

    def test_get_signal_invalid_index(self):
        """Test GET /api/files/{encoded}/batches/{batch}/signals/{idx} with bad index."""
        file_path = self.file_paths['SN001'][1]
        encoded = self.encode_file_path(file_path)

        response = self.fetch(
            f'/api/files/{encoded}/batches/batch_001/signals/999'
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
        self.file_path = self.file_paths['SN001'][1]

    def test_fft_analysis(self):
        """Test POST /api/analysis/fft computes FFT."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'sampling_rate': 1000.0,
        }

        response = self.post_json('/api/analysis/fft', payload)

        self.assertIn('frequencies', response)
        self.assertIn('magnitude', response)
        self.assertGreater(len(response['frequencies']), 0)
        self.assertEqual(
            len(response['frequencies']), len(response['magnitude'])
        )

    def test_psd_analysis(self):
        """Test POST /api/analysis/psd computes Power Spectral Density."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'sampling_rate': 1000.0,
            'nperseg': 1024,
            'window': 'hann',
        }

        response = self.post_json('/api/analysis/psd', payload)

        self.assertIn('frequencies', response)
        self.assertIn('psd', response)
        self.assertIn('dominant_peaks', response)
        self.assertGreater(len(response['frequencies']), 0)

    def test_lowpass_filter(self):
        """Test POST /api/analysis/filter applies lowpass filter."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'type': 'lowpass',
            'cutoff': 100.0,
            'sampling_rate': 1000.0,
            'order': 4,
        }

        response = self.post_json('/api/analysis/filter', payload)

        self.assertIn('time', response)
        self.assertIn('filtered', response)
        self.assertIn('original', response)
        self.assertEqual(response['filter_type'], 'lowpass')
        self.assertEqual(len(response['filtered']), len(response['original']))

    def test_highpass_filter(self):
        """Test POST /api/analysis/filter applies highpass filter."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'type': 'highpass',
            'cutoff': 50.0,
            'sampling_rate': 1000.0,
            'order': 4,
        }

        response = self.post_json('/api/analysis/filter', payload)

        self.assertEqual(response['filter_type'], 'highpass')
        self.assertIn('filtered', response)

    def test_bandpass_filter(self):
        """Test POST /api/analysis/filter applies bandpass filter."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'type': 'bandpass',
            'low_cutoff': 50.0,
            'high_cutoff': 150.0,
            'sampling_rate': 1000.0,
            'order': 4,
        }

        response = self.post_json('/api/analysis/filter', payload)

        self.assertEqual(response['filter_type'], 'bandpass')
        self.assertIn('filtered', response)

    def test_anomaly_zscore(self):
        """Test POST /api/analysis/anomaly detects anomalies with z-score."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'method': 'zscore',
            'threshold': 3.0,
        }

        response = self.post_json('/api/analysis/anomaly', payload)

        self.assertIn('anomaly_indices', response)
        self.assertIn('scores', response)
        self.assertIn('count', response)
        self.assertIsInstance(response['anomaly_indices'], list)
        self.assertIsInstance(response['scores'], list)

    def test_anomaly_mad(self):
        """Test POST /api/analysis/anomaly detects anomalies with MAD."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'method': 'mad',
            'threshold': 3.0,
        }

        response = self.post_json('/api/analysis/anomaly', payload)

        self.assertIn('count', response)
        self.assertGreaterEqual(response['count'], 0)

    def test_anomaly_iqr(self):
        """Test POST /api/analysis/anomaly detects anomalies with IQR."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
            'signal_idx': 0,
            'method': 'iqr',
            'threshold': 1.5,
        }

        response = self.post_json('/api/analysis/anomaly', payload)

        self.assertIn('count', response)

    def test_stats_analysis(self):
        """Test POST /api/analysis/stats computes signal statistics."""
        payload = {
            'file_path': self.file_path,
            'batch': 'batch_001',
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
            'batch': 'batch_001',
            'signal_idx': 0,
            'degree': 1,
        }

        response = self.post_json('/api/analysis/trend', payload)

        self.assertIn('coefficients', response)
        self.assertIn('fitted', response)
        self.assertIn('residuals', response)
        self.assertIn('time', response)
        self.assertEqual(len(response['fitted']), 1000)
        self.assertEqual(len(response['residuals']), 1000)

    def test_correlation_analysis(self):
        """Test POST /api/analysis/correlation computes cross-correlation."""
        payload = {
            'file_path_a': self.file_path,
            'batch_a': 'batch_001',
            'signal_idx_a': 0,
            'file_path_b': self.file_path,
            'batch_b': 'batch_001',
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
            'batch': 'batch_001',
            'signal_idx': 0,
        }

        response = self.fetch(
            '/api/analysis/fft',
            body=json.dumps(payload),
            method='POST',
        )
        self.assertEqual(response.code, 400)


class TestCacheAndRescan(IntegrationTestBase):
    """Test cache and rescan endpoints."""

    def test_cache_stats(self):
        """Test GET /api/cache/stats returns cache statistics."""
        # Load a signal to populate cache
        file_path = self.file_paths['SN001'][1]
        encoded = self.encode_file_path(file_path)
        self.get_json(
            f'/api/files/{encoded}/batches/batch_001/signals/0'
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
            '/api/analysis/fft',
            body='invalid json {',
            method='POST',
        )
        self.assertEqual(response.code, 500)

    def test_signal_caching(self):
        """Test that signals are cached after first load."""
        file_path = self.file_paths['SN001'][1]
        encoded = self.encode_file_path(file_path)

        # First load
        response1 = self.get_json(
            f'/api/files/{encoded}/batches/batch_001/signals/0'
        )

        # Get cache stats
        stats1 = self.get_json('/api/cache/stats')
        initial_entries = stats1['entries']

        # Second load (should hit cache)
        response2 = self.get_json(
            f'/api/files/{encoded}/batches/batch_001/signals/0'
        )

        stats2 = self.get_json('/api/cache/stats')

        # Cache should still have same number of entries
        self.assertEqual(stats2['entries'], initial_entries)
        # Data should be identical
        self.assertEqual(response1['values'], response2['values'])


if __name__ == '__main__':
    unittest.main()
