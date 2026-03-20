"""
Integration tests for the Signal Viewer web application.

Tests the Tornado web server, API endpoints, and data handling using
``tornado.testing.AsyncHTTPTestCase``.  Covers:

- Server startup and page rendering
- Cascading metadata API (serials → steps → files → batches)
- Signal loading, windowed queries, and downsampling
- Analysis endpoints (stats, correlation, trend)
- Error handling (404, 400, invalid JSON)
- Cache statistics and filesystem rescan

When tornado is not installed (e.g. system Python on macOS) the entire
module is skipped gracefully.
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

try:
    import tornado.testing
    import tornado.ioloop
    _HAS_TORNADO = True
except ModuleNotFoundError:
    _HAS_TORNADO = False

from signal_viewer import config
from signal_viewer.config import HDF5Schema as S
from signal_viewer.core.hdf5_reader import create_test_file

if _HAS_TORNADO:
    from signal_viewer.server.app import make_app


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

_BASE = tornado.testing.AsyncHTTPTestCase if _HAS_TORNADO else unittest.TestCase


@unittest.skipUnless(_HAS_TORNADO, "tornado is not installed")
class IntegrationTestBase(_BASE):
    """Shared fixture: temp data directory + running Tornado app.

    Creates two serial numbers (SN001, SN002) each with one procedure
    folder and two run folders containing a test HDF5 file.
    """

    # -- Setup / teardown ---------------------------------------------------

    # Root label used by all integration tests
    ROOT_LABEL = 'Default'

    def setUp(self):
        self._original_data_root = config.DATA_ROOT
        self._original_data_roots = config.DATA_ROOTS.copy()

        self.test_data_dir = tempfile.mkdtemp()
        self.serials = ["SN001", "SN002"]
        self.folder1s = ["p001_test"]
        self.folder2s = ["run_1", "run_2"]
        self.file_paths: Dict[str, Dict[str, Dict[str, str]]] = {}

        for serial in self.serials:
            self.file_paths[serial] = {}
            for f1 in self.folder1s:
                self.file_paths[serial][f1] = {}
                for f2 in self.folder2s:
                    dir_path = Path(self.test_data_dir) / serial / f1 / f2
                    dir_path.mkdir(parents=True)
                    fp = dir_path / "data.h5"
                    create_test_file(str(fp))
                    self.file_paths[serial][f1][f2] = str(fp)

        config.DATA_ROOT = Path(self.test_data_dir)
        config.DATA_ROOTS = {self.ROOT_LABEL: Path(self.test_data_dir)}
        try:
            super().setUp()
        except Exception:
            config.DATA_ROOT = self._original_data_root
            config.DATA_ROOTS = self._original_data_roots
            raise

    def tearDown(self):
        super().tearDown()
        config.DATA_ROOT = self._original_data_root
        config.DATA_ROOTS = self._original_data_roots
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def _root_url(self, path: str) -> str:
        """Prefix an API path with the root label segment."""
        return f"/api/roots/{quote(self.ROOT_LABEL, safe='')}{path}"

    def get_app(self):
        return make_app()

    # -- Convenience helpers ------------------------------------------------

    def encode_file_path(self, file_path: str) -> str:
        """Base64-encode a file path for use in API URLs."""
        return base64.urlsafe_b64encode(file_path.encode()).decode()

    def get_json(self, path: str, **kwargs) -> Dict:
        """GET *path*, parse the JSON body, and return it."""
        return json.loads(self.fetch(path, **kwargs).body.decode())

    def post_json(self, path: str, data: Dict, **kwargs) -> Dict:
        """POST JSON *data* to *path*, parse the response, and return it."""
        kwargs["body"] = json.dumps(data)
        kwargs["method"] = "POST"
        return json.loads(self.fetch(path, **kwargs).body.decode())


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

class TestPageHandlers(IntegrationTestBase):
    """Verify that every HTML page renders without error."""

    def test_root_page(self):
        resp = self.fetch("/")
        self.assertEqual(resp.code, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))

    def test_analysis_page(self):
        resp = self.fetch("/analysis")
        self.assertEqual(resp.code, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))

    def test_comparison_page(self):
        resp = self.fetch("/comparison")
        self.assertEqual(resp.code, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))

    def test_docs_page(self):
        resp = self.fetch("/docs")
        self.assertEqual(resp.code, 200)
        self.assertIn("text/html", resp.headers.get("Content-Type", ""))


# ---------------------------------------------------------------------------
# Metadata cascading API
# ---------------------------------------------------------------------------

class TestSerialsCascading(IntegrationTestBase):
    """Test the root → serials → steps → files metadata cascade."""

    def test_get_roots(self):
        data = self.get_json("/api/roots")
        self.assertIn("roots", data)
        self.assertEqual(len(data["roots"]), 1)
        self.assertIn(self.ROOT_LABEL, data["roots"])

    def test_get_serials(self):
        data = self.get_json(self._root_url("/serials"))
        self.assertIn("serials", data)
        self.assertEqual(len(data["serials"]), 2)
        self.assertIn("SN001", data["serials"])
        self.assertIn("SN002", data["serials"])

    def test_get_serials_missing_root_returns_404(self):
        self.assertEqual(self.fetch("/api/roots/MISSING/serials").code, 404)

    def test_get_steps_for_serial(self):
        data = self.get_json(self._root_url("/serials/SN001/steps"))
        self.assertIn("steps", data)
        self.assertEqual(len(data["steps"]), 2)
        self.assertIn("p001_test/run_1", data["steps"])
        self.assertIn("p001_test/run_2", data["steps"])

    def test_get_steps_missing_serial_returns_404(self):
        self.assertEqual(self.fetch(self._root_url("/serials/MISSING/steps")).code, 404)

    def test_get_files_for_step(self):
        step = quote("p001_test/run_1", safe="")
        data = self.get_json(self._root_url(f"/serials/SN001/steps/{step}/files"))
        self.assertIn("files", data)
        self.assertEqual(len(data["files"]), 1)
        self.assertIsInstance(data["files"][0], dict)
        self.assertIn("path", data["files"][0])

    def test_get_files_missing_step_returns_404(self):
        self.assertEqual(
            self.fetch(self._root_url("/serials/SN001/steps/missing_step/files")).code,
            404,
        )


# ---------------------------------------------------------------------------
# Batches and signals
# ---------------------------------------------------------------------------

class TestBatchesAndSignals(IntegrationTestBase):
    """Test batch listing, signal loading, windowing, and downsampling."""

    def _encoded(self) -> str:
        return self.encode_file_path(
            self.file_paths["SN001"]["p001_test"]["run_1"]
        )

    # -- Batches / metadata -------------------------------------------------

    def test_get_batches(self):
        data = self.get_json(f"/api/files/{self._encoded()}/batches")
        self.assertIn("batches", data)
        self.assertIn(S.default_group(), data["batches"])

    def test_get_batch_metadata(self):
        data = self.get_json(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}/meta"
        )
        self.assertIn("signal_names", data)
        self.assertIn("units", data)
        self.assertEqual(len(data["signal_names"]), 4)
        self.assertEqual(len(data["units"]), 4)

    # -- Signal loading -----------------------------------------------------

    def test_get_signal(self):
        """Full signal load — Type A signal 0 has 800 samples."""
        data = self.get_json(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}/signals/0"
        )
        for key in ("time", "values", "name", "units", "samples"):
            self.assertIn(key, data)
        self.assertEqual(data["samples"], 800)
        self.assertEqual(len(data["time"]), 800)
        self.assertEqual(len(data["values"]), 800)

    def test_get_signal_with_downsampling(self):
        data = self.get_json(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}"
            f"/signals/0?downsample=100"
        )
        self.assertIn("values", data)
        self.assertLess(len(data["values"]), 200)

    def test_get_signal_with_time_window(self):
        """Request a ~50 % window in the middle of the signal."""
        full = self.get_json(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}/signals/0"
        )
        mid = (full["t_start"] + full["t_end"]) / 2
        quarter = (full["t_end"] - full["t_start"]) / 4

        windowed = self.get_json(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}/signals/0"
            f"?t_min={mid - quarter}&t_max={mid + quarter}"
        )
        self.assertTrue(windowed["windowed"])
        self.assertLess(len(windowed["values"]), full["total_samples"])
        self.assertEqual(windowed["total_samples"], full["total_samples"])

    def test_get_signal_windowed_with_downsampling(self):
        """Combine time window with downsample=50."""
        full = self.get_json(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}/signals/0"
        )
        windowed = self.get_json(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}/signals/0"
            f"?t_min={full['t_start']}&t_max={full['t_end']}&downsample=50"
        )
        self.assertEqual(len(windowed["time"]), len(windowed["values"]))
        self.assertLessEqual(len(windowed["values"]), 100)

    # -- Error cases --------------------------------------------------------

    def test_get_signal_invalid_index_returns_404(self):
        resp = self.fetch(
            f"/api/files/{self._encoded()}/batches/{S.default_group()}/signals/999"
        )
        self.assertEqual(resp.code, 404)

    def test_get_signal_invalid_file_path_returns_400(self):
        bad = base64.urlsafe_b64encode(b"/nonexistent/path.h5").decode()
        self.assertEqual(self.fetch(f"/api/files/{bad}/batches").code, 400)


# ---------------------------------------------------------------------------
# Analysis endpoints
# ---------------------------------------------------------------------------

class TestAnalysisEndpoints(IntegrationTestBase):
    """Test stats, trend, and correlation analysis."""

    def setUp(self):
        super().setUp()
        self.file_path = self.file_paths["SN001"]["p001_test"]["run_1"]

    def test_stats_analysis(self):
        data = self.post_json("/api/analysis/stats", {
            "file_path": self.file_path,
            "batch": S.default_group(),
            "signal_idx": 0,
        })
        for key in ("count", "mean", "std", "min", "max", "median"):
            self.assertIn(key, data)

    def test_trend_analysis(self):
        data = self.post_json("/api/analysis/trend", {
            "file_path": self.file_path,
            "batch": S.default_group(),
            "signal_idx": 0,
            "degree": 1,
        })
        for key in ("coefficients", "fitted", "residuals", "time"):
            self.assertIn(key, data)
        self.assertEqual(len(data["fitted"]), 800)
        self.assertEqual(len(data["residuals"]), 800)

    def test_correlation_analysis(self):
        data = self.post_json("/api/analysis/correlation", {
            "file_path_a": self.file_path,
            "batch_a": S.default_group(),
            "signal_idx_a": 0,
            "file_path_b": self.file_path,
            "batch_b": S.default_group(),
            "signal_idx_b": 1,
        })
        for key in ("lags", "correlation", "max_lag"):
            self.assertIn(key, data)
        self.assertEqual(len(data["lags"]), len(data["correlation"]))

    def test_missing_required_fields_returns_400(self):
        resp = self.fetch(
            "/api/analysis/stats",
            body=json.dumps({"batch": S.default_group(), "signal_idx": 0}),
            method="POST",
        )
        self.assertEqual(resp.code, 400)


# ---------------------------------------------------------------------------
# Cache and rescan
# ---------------------------------------------------------------------------

class TestCacheAndRescan(IntegrationTestBase):
    """Test the cache stats and filesystem rescan endpoints."""

    def test_cache_stats(self):
        enc = self.encode_file_path(
            self.file_paths["SN001"]["p001_test"]["run_1"]
        )
        self.get_json(f"/api/files/{enc}/batches/{S.default_group()}/signals/0")

        stats = self.get_json("/api/cache/stats")
        for key in ("hits", "misses", "entries", "memory_used", "memory_budget"):
            self.assertIn(key, stats)

    def test_rescan_endpoint(self):
        data = self.post_json("/api/rescan", {})
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["serial_count"], 2)
        # 2 serials × 1 folder_1 × 2 folder_2s = 4 steps
        self.assertEqual(data["total_steps"], 4)


# ---------------------------------------------------------------------------
# Error handling and edge cases
# ---------------------------------------------------------------------------

class TestErrorHandling(IntegrationTestBase):
    """CORS headers, OPTIONS preflight, invalid JSON, caching behaviour."""

    def test_cors_headers(self):
        resp = self.fetch("/api/roots")
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "*")
        self.assertIn("GET", resp.headers.get("Access-Control-Allow-Methods"))

    def test_options_preflight(self):
        self.assertEqual(self.fetch("/api/roots", method="OPTIONS").code, 204)

    def test_invalid_json_returns_500(self):
        resp = self.fetch(
            "/api/analysis/stats", body="invalid json {", method="POST"
        )
        self.assertEqual(resp.code, 500)

    def test_signal_caching(self):
        """Second GET for the same signal should hit the cache."""
        enc = self.encode_file_path(
            self.file_paths["SN001"]["p001_test"]["run_1"]
        )
        url = f"/api/files/{enc}/batches/{S.default_group()}/signals/0"

        resp1 = self.get_json(url)
        entries_after_first = self.get_json("/api/cache/stats")["entries"]

        resp2 = self.get_json(url)
        entries_after_second = self.get_json("/api/cache/stats")["entries"]

        self.assertEqual(entries_after_second, entries_after_first)
        self.assertEqual(resp1["values"], resp2["values"])


if __name__ == "__main__":
    unittest.main()
