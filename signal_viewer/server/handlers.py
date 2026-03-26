"""
Tornado Request Handlers for Engineering Time Series Signal Viewer API

Includes:
  - PageHandler: Renders Jinja2 templates for HTML pages
  - API Handlers: Return JSON responses for data operations
  - Error handling and numpy array serialization
"""

import base64
import concurrent.futures
import json
import logging
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import tornado.ioloop
import tornado.web
import tornado.iostream

# Shared thread pool for CPU-heavy handlers (correlation, statistics, etc.)
_COMPUTE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2)

from signal_viewer import config
from signal_viewer.core.hdf5_reader import HDF5Reader
from signal_viewer.processing import (
    resampling,
    statistics,
    correlation,
    trend,
)

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy types to native Python types."""

    def default(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class BaseHandler(tornado.web.RequestHandler):
    """
    Base handler with common functionality.

    Features:
      - JSON serialization with numpy support
      - CORS headers
      - Error handling
      - Access to shared app context (index, cache, readers)
    """

    def set_default_headers(self):
        """Set default headers for CORS and JSON."""
        self.set_header("Content-Type", "application/json")
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.set_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization",
        )

    def options(self, *args, **kwargs):
        """Handle CORS preflight requests."""
        self.set_status(204)
        self.finish()

    def write_json(self, data: dict):
        """
        Write response as JSON with numpy type conversion.

        Args:
            data: Dictionary to serialize
        """
        self.write(json.dumps(data, cls=NumpyEncoder))

    def write_error(self, status_code: int, **kwargs):
        """
        Handle error responses with JSON formatting.

        Args:
            status_code: HTTP status code
        """
        self.set_status(status_code)
        error_message = self._reason

        if status_code == 404:
            error_message = "Not found"
        elif status_code == 500:
            error_message = "Internal server error"
            if self.application.settings.get("debug"):
                # Include traceback in debug mode
                exc_info = kwargs.get("exc_info")
                if exc_info:
                    error_message = "".join(
                        traceback.format_exception(*exc_info)
                    )

        self.write_json({"error": error_message, "status": status_code})

    def get_metadata_index(self, root_label: str):
        """
        Look up the MetadataIndex for a given root label.

        Args:
            root_label: URL-decoded root label (key in DATA_ROOTS)

        Returns:
            MetadataIndex instance

        Raises:
            tornado.web.HTTPError: 404 if root label not found, 503 if index not initialised
        """
        indices = getattr(self.application, 'metadata_indices', {})
        if root_label not in indices:
            raise tornado.web.HTTPError(404, f"Root not found: {root_label}")
        idx = indices[root_label]
        if idx is None:
            raise tornado.web.HTTPError(503, "Index not initialized for this root")
        return idx

    def decode_file_path(self, encoded_path: str) -> str:
        """
        Decode base64-encoded file path from URL and validate it stays within
        any configured DATA_ROOT.

        Args:
            encoded_path: Base64-encoded file path

        Returns:
            Decoded file path

        Raises:
            ValueError: If decoding fails or path is outside all data roots
        """
        try:
            # Re-add padding stripped by JS encodePath
            padded = encoded_path + "=" * (-len(encoded_path) % 4)
            decoded = base64.urlsafe_b64decode(padded)
            path = decoded.decode("utf-8")

            # Validate path stays within ANY configured data root
            resolved = Path(path).resolve()
            for root_path in config.DATA_ROOTS.values():
                data_root = Path(root_path).resolve()
                if str(resolved).startswith(str(data_root) + "/") or resolved == data_root:
                    return path

            raise ValueError("Path outside data root")
        except Exception as e:
            raise ValueError(f"Failed to decode file path: {e}")

    def get_hdf5_reader(self, file_path: str) -> HDF5Reader:
        """
        Get or create HDF5Reader for a file (lazy loading).

        Readers are cached in app.hdf5_readers dict.

        Args:
            file_path: Path to HDF5 file

        Returns:
            HDF5Reader instance

        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file is not a valid HDF5 file
        """
        if file_path not in self.application.hdf5_readers:
            try:
                reader = HDF5Reader(file_path)
                self.application.hdf5_readers[file_path] = reader
            except (FileNotFoundError, ValueError) as e:
                raise tornado.web.HTTPError(
                    400, f"Invalid HDF5 file: {e}"
                )

        return self.application.hdf5_readers[file_path]


class PageHandler(BaseHandler):
    """
    Renders HTML pages using Jinja2 templates.

    Template name is passed via the route's third-argument dict and received
    through Tornado's ``initialize()`` method.
    """

    def initialize(self, template: str = ""):
        """
        Store template name from route configuration.

        Args:
            template: Template filename (e.g., 'viewer.html')
        """
        self.template = template

    def get(self):
        """Render the template assigned to this route."""
        if not self.template:
            self.set_status(400)
            self.write_json({"error": "Template not specified"})
            return

        try:
            jinja_env = self.application.jinja_env
            tmpl = jinja_env.get_template(self.template)
            html = tmpl.render(
                app_title=config.APP_TITLE,
                debug=config.DEBUG,
            )
            self.set_header("Content-Type", "text/html")
            self.write(html)
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in PageHandler: {e}\n{traceback.format_exc()}")
            self.set_status(500)
            self.write_json({"error": f"Template error: {e}"})


class RootsHandler(BaseHandler):
    """GET /api/roots → List configured data root labels."""

    def get(self):
        """Return list of root labels (keys from DATA_ROOTS)."""
        try:
            roots = list(getattr(self.application, 'metadata_indices', {}).keys())
            self.write_json({"roots": roots})
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in RootsHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in RootsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class SerialsHandler(BaseHandler):
    """GET /api/roots/{root}/serials → List serial numbers for a root."""

    def get(self, root_label: str):
        """Return list of serial numbers for the given root."""
        try:
            idx = self.get_metadata_index(root_label)
            serials = idx.get_serial_numbers()
            self.write_json({"serials": serials})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in SerialsHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in SerialsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class StepsHandler(BaseHandler):
    """GET /api/roots/{root}/serials/{serial}/steps → List steps for a serial."""

    def get(self, root_label: str, serial_num: str):
        """
        Get steps (folder_1/folder_2 keys) for a serial within a root.

        Args:
            root_label: root label (key in DATA_ROOTS)
            serial_num: serial identifier (e.g., "SN001")
        """
        try:
            idx = self.get_metadata_index(root_label)
            steps = idx.get_steps(serial_num)
            self.write_json({"steps": steps})
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(404, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in StepsHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in StepsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class FilesHandler(BaseHandler):
    """GET /api/roots/{root}/serials/{serial}/steps/{step}/files → List files."""

    def get(self, root_label: str, serial_num: str, step: str):
        """
        Get files for a serial and step within a root.

        Args:
            root_label: root label (key in DATA_ROOTS)
            serial_num: serial identifier
            step: step key (folder_1/folder_2)
        """
        try:
            idx = self.get_metadata_index(root_label)
            files = idx.get_files(serial_num, step)
            self.write_json({"files": files})
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(404, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in FilesHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in FilesHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class BatchesHandler(BaseHandler):
    """GET /api/files/{encoded_path}/batches → List configured groups in a file."""

    def get(self, encoded_path: str):
        """
        Get configured HDF5 groups present in the file.

        Args:
            encoded_path: Base64-encoded file path
        """
        try:
            file_path = self.decode_file_path(encoded_path)
            reader = self.get_hdf5_reader(file_path)

            batches = reader.get_groups()
            self.write_json({"batches": batches})
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(400, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in BatchesHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in BatchesHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class BatchMetaHandler(BaseHandler):
    """GET /api/files/{encoded_path}/batches/{batch}/meta → Batch metadata."""

    def get(self, encoded_path: str, batch: str):
        """
        Get metadata for a batch.

        Args:
            encoded_path: Base64-encoded file path
            batch: Batch name
        """
        try:
            file_path = self.decode_file_path(encoded_path)
            reader = self.get_hdf5_reader(file_path)

            metadata = reader.get_batch_metadata(batch)
            self.write_json(metadata)
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(404, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in BatchMetaHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in BatchMetaHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class SignalHandler(BaseHandler):
    """GET /api/files/{encoded_path}/batches/{batch}/signals/{idx} → Signal data.

    Supports adaptive-zoom by accepting an optional time window:
      ?downsample=N           target point count (default 5000, max 10000)
      &t_min=<epoch_ms>       left edge of visible window
      &t_max=<epoch_ms>       right edge of visible window

    When t_min / t_max are given, the server slices the full (cached) signal
    to that window and downsamples only the slice — giving the browser
    pixel-level detail at any zoom level while never exceeding N points.

    The first request (no t_min/t_max) additionally returns ``total_samples``,
    ``t_start``, and ``t_end`` so the frontend knows the full signal extent.
    """

    def get(self, encoded_path: str, batch: str, signal_idx: str):
        try:
            file_path = self.decode_file_path(encoded_path)
            reader = self.get_hdf5_reader(file_path)
            idx = int(signal_idx)

            # ── Load full signal (from cache or HDF5) ────────────────────
            cache_key = f"{file_path}::{batch}::{idx}"
            cached = self.application.signal_cache.get(cache_key)
            if cached is not None:
                time_full, signal_full = cached
            else:
                time_full, signal_full = reader.load_signal(batch, idx)
                self.application.signal_cache.put(
                    cache_key, time_full, signal_full
                )

            total_samples = len(signal_full)
            t_start = float(time_full[0])
            t_end = float(time_full[-1])

            # ── Optional time-window slice ────────────────────────────────
            t_min_param = self.get_argument("t_min", None)
            t_max_param = self.get_argument("t_max", None)
            is_windowed = False

            if t_min_param is not None and t_max_param is not None:
                try:
                    t_min = float(t_min_param)
                    t_max = float(t_max_param)
                    if t_min < t_max:
                        # Binary-search for slice boundaries
                        i_lo = int(np.searchsorted(time_full, t_min, side="left"))
                        i_hi = int(np.searchsorted(time_full, t_max, side="right"))
                        # Expand context on each side by ~2% of the slice length
                        # (minimum 2 samples) so the LTTB-downsampled result has
                        # adequate density at the visible edges.
                        slice_len = max(i_hi - i_lo, 1)
                        expand = max(2, int(slice_len * 0.02))
                        i_lo = max(0, i_lo - expand)
                        i_hi = min(total_samples, i_hi + expand)
                        time_full = time_full[i_lo:i_hi]
                        signal_full = signal_full[i_lo:i_hi]
                        is_windowed = True
                except (ValueError, TypeError):
                    pass

            # ── Downsample ────────────────────────────────────────────────
            downsample_param = self.get_argument("downsample", None)
            if downsample_param:
                try:
                    target_points = int(downsample_param)
                    target_points = min(
                        max(target_points, 10), config.MAX_DOWNSAMPLE_POINTS
                    )
                    if len(signal_full) > target_points:
                        time_full, signal_full = (
                            resampling.minmax_lttb_downsample(
                                time_full, signal_full, target_points
                            )
                        )
                except (ValueError, TypeError):
                    pass

            # ── Metadata ──────────────────────────────────────────────────
            batch_meta = reader.get_batch_metadata(batch)
            signal_name = batch_meta["signal_names"][idx]
            units = batch_meta["units"][idx]

            out_samples = len(signal_full)
            result = {
                "time": time_full,
                "values": signal_full,
                "name": signal_name,
                "units": units,
                "samples": out_samples,
                "total_samples": total_samples,
                "t_start": t_start,
                "t_end": t_end,
                "windowed": is_windowed,
            }

            if config.VERBOSE:
                logger.info(
                    "SignalHandler %s/%d: total=%d out=%d windowed=%s",
                    batch, idx, total_samples, out_samples, is_windowed,
                )

            self.write_json(result)
        except tornado.web.HTTPError:
            raise
        except (ValueError, IndexError) as e:
            raise tornado.web.HTTPError(404, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(
                    f"Error in SignalHandler: {e}\n{traceback.format_exc()}"
                )
            else:
                logger.error(f"Error in SignalHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))




class StatsHandler(BaseHandler):
    """POST /api/analysis/stats → Compute signal statistics.

    Returns descriptive stats, pre-computed rainflow visualisation data
    (histogram, exceedance curves, percentile table) and an optional
    downsampled copy of the signal values for the value histogram.

    Heavy computation is offloaded to a thread-pool executor so the
    Tornado event loop stays responsive.
    """

    async def post(self):
        """
        Compute descriptive statistics and rainflow cycle counting for a signal.

        Request body:
        {
          "file_path": "/path/to/file.h5",
          "batch": "batch_001",
          "signal_idx": 0,
          "rainflow_bins": 10,
          "include_signal": 5000       ← optional: downsample target
        }

        When *include_signal* is given the response includes a ``values``
        key with the downsampled signal, removing the need for a second
        GET request from the frontend.
        """
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path")
            batch = data.get("batch")
            signal_idx = int(data.get("signal_idx", 0))
            rainflow_bins = int(data.get("rainflow_bins", 10))
            include_signal = data.get("include_signal")  # int or None

            if not file_path or not batch:
                raise tornado.web.HTTPError(400, "Missing required fields")

            if rainflow_bins < 1:
                raise tornado.web.HTTPError(400, "rainflow_bins must be >= 1")

            reader = self.get_hdf5_reader(file_path)
            time_arr, signal = reader.load_signal(batch, signal_idx)

            # Populate the signal cache so any subsequent GET is a hit
            cache_key = f"{file_path}::{batch}::{signal_idx}"
            self.application.signal_cache.put(cache_key, time_arr, signal)

            # Offload heavy computation to thread pool
            loop = tornado.ioloop.IOLoop.current()
            result = await loop.run_in_executor(
                _COMPUTE_EXECUTOR,
                self._compute_stats,
                signal,
                rainflow_bins,
                int(include_signal) if include_signal else 0,
                time_arr,
            )

            self.write_json(result)
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in StatsHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in StatsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))

    @staticmethod
    def _compute_stats(signal, rainflow_bins, downsample_target, time_arr):
        """Pure-computation helper executed in a thread."""
        stats = statistics.compute_descriptive_stats(signal)
        rf = statistics.compute_rainflow(signal, n_bins=rainflow_bins)

        # ── Pre-compute visualisation data so the frontend doesn't ──
        # ── need the full ranges/cycle_maxs/cycle_mins arrays.     ──
        rainflow_vis = StatsHandler._build_rainflow_vis(rf)

        result = {**stats, "rainflow": rainflow_vis}

        # Optionally include downsampled signal values
        if downsample_target and downsample_target > 0:
            target = min(max(downsample_target, 10), config.MAX_DOWNSAMPLE_POINTS)
            if len(signal) > target:
                _, ds_signal = resampling.lttb_downsample(
                    time_arr, signal, target
                )
                result["values"] = ds_signal.tolist()
            else:
                result["values"] = signal.tolist()

        return result

    @staticmethod
    def _build_rainflow_vis(rf):
        """Convert raw rainflow output into compact visualisation payload.

        Returns a dict with histogram (counts, bin_edges), percentile
        table, exceedance curves (downsampled to ≤500 pts), and summary
        scalars.  The bulky ranges/cycle_maxs/cycle_mins arrays are NOT
        included — saving potentially megabytes of JSON.
        """
        vis = {
            "counts": rf["counts"],
            "bin_edges": rf["bin_edges"],
            "total_cycles": rf["total_cycles"],
            "total_half_cycles": rf["total_half_cycles"],
        }

        ranges = rf.get("ranges", [])
        cycle_maxs = rf.get("cycle_maxs", [])
        cycle_mins = rf.get("cycle_mins", [])

        # Max range
        vis["maxRange"] = max(ranges) if ranges else None

        # Percentile table (every 10th percentile)
        if ranges:
            sorted_r = sorted(ranges)
            n = len(sorted_r)
            ptable = []
            for p in range(10, 101, 10):
                idx = min(int(p / 100 * n), n - 1)
                val = sorted_r[idx]
                # Count cycles with range >= val
                # (binary search on sorted array)
                lo, hi = 0, n
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sorted_r[mid] < val:
                        lo = mid + 1
                    else:
                        hi = mid
                count = n - lo
                ptable.append({"percentile": p, "value": val, "count": count})
            vis["percentiles"] = ptable
        else:
            vis["percentiles"] = []

        # Exceedance curves (downsampled to ≤500 points)
        MAX_PTS = 500
        vis["rangeExceedance"] = StatsHandler._exceedance_desc(ranges, MAX_PTS)
        vis["maxExceedance"] = StatsHandler._exceedance_desc(cycle_maxs, MAX_PTS)
        vis["minExceedance"] = StatsHandler._exceedance_asc(cycle_mins, MAX_PTS)

        return vis

    @staticmethod
    def _exceedance_desc(values, max_pts):
        """Build descending exceedance curve: cycles with value ≥ threshold."""
        if not values:
            return None
        sorted_v = sorted(values, reverse=True)
        n = len(sorted_v)
        thresholds, counts = [], []
        prev = None
        for i, v in enumerate(sorted_v):
            if v != prev:
                thresholds.append(v)
                counts.append(i + 1)
                prev = v
        if counts and counts[-1] != n:
            thresholds.append(sorted_v[-1])
            counts.append(n)
        return StatsHandler._downsample_curve(counts, thresholds, max_pts)

    @staticmethod
    def _exceedance_asc(values, max_pts):
        """Build ascending exceedance curve: cycles with value ≤ threshold."""
        if not values:
            return None
        sorted_v = sorted(values)
        n = len(sorted_v)
        thresholds, counts = [], []
        prev = None
        for i, v in enumerate(sorted_v):
            if v != prev:
                thresholds.append(v)
                counts.append(i + 1)
                prev = v
        if counts and counts[-1] != n:
            thresholds.append(sorted_v[-1])
            counts.append(n)
        return StatsHandler._downsample_curve(counts, thresholds, max_pts)

    @staticmethod
    def _downsample_curve(x_arr, y_arr, max_pts):
        """Log-spaced downsample of an exceedance curve."""
        n = len(x_arr)
        if n <= max_pts:
            return {"x": x_arr, "y": y_arr}
        import math
        indices = set()
        indices.add(0)
        indices.add(n - 1)
        log_max = math.log(n)
        for i in range(1, max_pts - 1):
            log_idx = log_max * i / (max_pts - 1)
            idx = min(round(math.exp(log_idx)) - 1, n - 1)
            indices.add(idx)
        sorted_idx = sorted(indices)
        return {
            "x": [x_arr[i] for i in sorted_idx],
            "y": [y_arr[i] for i in sorted_idx],
        }


class CorrelationHandler(BaseHandler):
    """POST /api/analysis/correlation → Compute cross-correlation.

    Heavy computation is offloaded to a thread-pool executor so that
    Tornado's event loop stays responsive for other requests.
    """

    # Cap the number of samples fed into np.correlate to keep the
    # computation tractable (~10 K points → finishes in < 1 s).
    MAX_CORR_SAMPLES = 10_000

    async def post(self):
        """
        Compute cross-correlation between two signals.

        Request body:
        {
          "file_path_a": "/path/to/file1.h5",
          "batch_a": "batch_001",
          "signal_idx_a": 0,
          "file_path_b": "/path/to/file2.h5",
          "batch_b": "batch_001",
          "signal_idx_b": 1
        }
        """
        try:
            data = json.loads(self.request.body)
            file_path_a = data.get("file_path_a")
            batch_a = data.get("batch_a")
            signal_idx_a = int(data.get("signal_idx_a", 0))
            file_path_b = data.get("file_path_b")
            batch_b = data.get("batch_b")
            signal_idx_b = int(data.get("signal_idx_b", 0))

            if not all([file_path_a, batch_a, file_path_b, batch_b]):
                raise tornado.web.HTTPError(400, "Missing required fields")

            reader_a = self.get_hdf5_reader(file_path_a)
            time_a, signal_a = reader_a.load_signal(batch_a, signal_idx_a)

            reader_b = self.get_hdf5_reader(file_path_b)
            time_b, signal_b = reader_b.load_signal(batch_b, signal_idx_b)

            # Ensure same length (truncate longer signal)
            min_len = min(len(signal_a), len(signal_b))
            signal_a = signal_a[:min_len]
            signal_b = signal_b[:min_len]

            # Down-sample large signals to keep correlation tractable.
            # LTTB would be ideal but a simple uniform stride is sufficient
            # for a correlation overview.
            cap = self.MAX_CORR_SAMPLES
            if min_len > cap:
                stride = max(1, min_len // cap)
                signal_a = signal_a[::stride].copy()
                signal_b = signal_b[::stride].copy()

            # Offload the CPU-heavy computation to a thread so we don't
            # block the Tornado event loop (which would freeze all other
            # HTTP handlers until this returns).
            loop = tornado.ioloop.IOLoop.current()
            result = await loop.run_in_executor(
                _COMPUTE_EXECUTOR,
                self._compute_correlation,
                signal_a,
                signal_b,
            )

            self.write_json(result)
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in CorrelationHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in CorrelationHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))

    @staticmethod
    def _compute_correlation(signal_a, signal_b):
        """Pure-computation helper executed in a thread."""
        lags, corr = correlation.cross_correlate(
            signal_a, signal_b, mode="same"
        )
        max_lag = correlation.find_lag(signal_a, signal_b)
        return {
            "lags": lags,
            "correlation": corr,
            "max_lag": max_lag,
        }


class TrendHandler(BaseHandler):
    """POST /api/analysis/trend → Analyze signal trend."""

    def post(self):
        """
        Fit polynomial trend to a signal.

        Request body:
        {
          "file_path": "/path/to/file.h5",
          "batch": "batch_001",
          "signal_idx": 0,
          "degree": 1
        }
        """
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path")
            batch = data.get("batch")
            signal_idx = int(data.get("signal_idx", 0))
            degree = int(data.get("degree", 1))

            if not file_path or not batch:
                raise tornado.web.HTTPError(400, "Missing required fields")

            reader = self.get_hdf5_reader(file_path)
            time, signal = reader.load_signal(batch, signal_idx)

            # Fit polynomial trend
            coeffs, fitted, residuals = trend.fit_polynomial(
                time, signal, degree
            )

            self.write_json(
                {
                    "coefficients": coeffs,
                    "fitted": fitted,
                    "residuals": residuals,
                    "time": time,
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in TrendHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in TrendHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class CacheStatsHandler(BaseHandler):
    """GET /api/cache/stats → Get cache statistics."""

    def get(self):
        """Return cache statistics."""
        try:
            stats = self.application.signal_cache.stats()
            self.write_json(stats)
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in CacheStatsHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in CacheStatsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class RescanHandler(BaseHandler):
    """POST /api/rescan → Re-scan filesystem for new files."""

    def post(self):
        """
        Re-scan all data root directories to detect new or removed files.

        Returns metadata about the rescan operation.
        """
        try:
            indices = getattr(self.application, 'metadata_indices', {})
            if not indices:
                raise tornado.web.HTTPError(503, "No indices initialized")

            total_serials = 0
            total_steps = 0
            for idx in indices.values():
                idx.rescan()
                serials = idx.get_serial_numbers()
                total_serials += len(serials)
                total_steps += sum(len(idx.get_steps(s)) for s in serials)

            self.write_json(
                {
                    "status": "success",
                    "serial_count": total_serials,
                    "total_steps": total_steps,
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in RescanHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in RescanHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


# ======================================================================
# Favourites / Comments / Lists API
# ======================================================================

class FavouritesHandler(BaseHandler):
    """GET/POST /api/favourites/{encoded_path} → Check or toggle favourite."""

    def get(self, encoded_path: str):
        """Check whether a file is favourited by the current user."""
        try:
            file_path = self.decode_file_path(encoded_path)
            db = self.application.database
            self.write_json({"favourite": db.is_favourite(file_path)})
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(400, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in FavouritesHandler GET: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in FavouritesHandler GET: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def post(self, encoded_path: str):
        """Toggle favourite on/off. Body: {"active": true/false}."""
        try:
            file_path = self.decode_file_path(encoded_path)
            data = json.loads(self.request.body)
            active = bool(data.get("active", True))
            db = self.application.database
            new_state = db.set_favourite(file_path, active)
            self.write_json({"favourite": new_state})
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(400, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in FavouritesHandler POST: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in FavouritesHandler POST: {e}")
            raise tornado.web.HTTPError(500, str(e))


class CommentsHandler(BaseHandler):
    """GET/POST /api/comments/{encoded_path} → Read or write comments for a file."""

    def get(self, encoded_path: str):
        """Return private + public comments for a file."""
        try:
            file_path = self.decode_file_path(encoded_path)
            db = self.application.database
            self.write_json(db.get_comments(file_path))
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(400, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in CommentsHandler GET: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in CommentsHandler GET: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def post(self, encoded_path: str):
        """Create or update a comment.

        Body: {
          "content": "...",
          "is_public": false,        // optional, default false (private)
          "comment_id": null         // optional, for updating existing public comment
        }
        """
        try:
            file_path = self.decode_file_path(encoded_path)
            data = json.loads(self.request.body)
            content = data.get("content", "")
            is_public = bool(data.get("is_public", False))
            comment_id = data.get("comment_id")
            db = self.application.database

            if is_public:
                cid = db.save_public_comment(
                    file_path, content,
                    comment_id=int(comment_id) if comment_id else None,
                )
            else:
                cid = db.save_private_comment(file_path, content)

            self.write_json({"id": cid})
        except tornado.web.HTTPError:
            raise
        except ValueError as e:
            raise tornado.web.HTTPError(400, str(e))
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in CommentsHandler POST: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in CommentsHandler POST: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def delete(self, encoded_path: str):
        """Delete a comment by id. Query param: ?id=<comment_id>."""
        try:
            comment_id = self.get_argument("id", None)
            if not comment_id:
                raise tornado.web.HTTPError(400, "Missing comment id")
            db = self.application.database
            ok = db.delete_comment(int(comment_id))
            self.write_json({"deleted": ok})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in CommentsHandler DELETE: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in CommentsHandler DELETE: {e}")
            raise tornado.web.HTTPError(500, str(e))


class ListsHandler(BaseHandler):
    """GET/POST/PATCH/DELETE /api/lists → Manage user lists."""

    def get(self):
        """Return all lists visible to the current user, including current_user."""
        try:
            db = self.application.database
            current_user = db.get_current_user()
            # Optional: filter lists that contain a specific file
            file_filter = self.get_argument("file", None)
            if file_filter:
                self.write_json({
                    "lists": db.get_lists_for_file(file_filter),
                    "current_user": current_user,
                })
            else:
                self.write_json({
                    "lists": db.get_lists(),
                    "current_user": current_user,
                })
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ListsHandler GET: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ListsHandler GET: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def post(self):
        """Create a new list. Body: {"name": "...", "is_public": false}."""
        try:
            data = json.loads(self.request.body)
            name = data.get("name", "").strip()
            if not name:
                raise tornado.web.HTTPError(400, "List name is required")
            is_public = bool(data.get("is_public", False))
            db = self.application.database
            lid = db.create_list(name, is_public=is_public)
            if lid is None:
                raise tornado.web.HTTPError(409, "List name already exists")
            self.write_json({"id": lid, "name": name})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ListsHandler POST: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ListsHandler POST: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def patch(self):
        """Toggle a list's public/private state. Body: {"list_id": N, "is_public": bool}."""
        try:
            data = json.loads(self.request.body)
            list_id = data.get("list_id")
            is_public = data.get("is_public")
            if list_id is None or is_public is None:
                raise tornado.web.HTTPError(400, "list_id and is_public are required")
            db = self.application.database
            ok = db.update_list_public(int(list_id), bool(is_public))
            self.write_json({"updated": ok})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ListsHandler PATCH: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ListsHandler PATCH: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def delete(self):
        """Delete a list. Query param: ?id=<list_id>."""
        try:
            list_id = self.get_argument("id", None)
            if not list_id:
                raise tornado.web.HTTPError(400, "Missing list id")
            db = self.application.database
            ok = db.delete_list(int(list_id))
            self.write_json({"deleted": ok})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ListsHandler DELETE: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ListsHandler DELETE: {e}")
            raise tornado.web.HTTPError(500, str(e))


class ListFilesHandler(BaseHandler):
    """GET/POST/DELETE /api/lists/{list_id}/files → Manage files in a list."""

    def get(self, list_id: str):
        """Return files in a list."""
        try:
            db = self.application.database
            files = db.get_list_files(int(list_id))
            self.write_json({"files": files})
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ListFilesHandler GET: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ListFilesHandler GET: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def post(self, list_id: str):
        """Add a file to a list. Body: {"file_path": "..."}."""
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path", "")
            if not file_path:
                raise tornado.web.HTTPError(400, "file_path is required")
            db = self.application.database
            ok = db.add_file_to_list(int(list_id), file_path)
            self.write_json({"added": ok})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ListFilesHandler POST: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ListFilesHandler POST: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def delete(self, list_id: str):
        """Remove a file from a list. Query param: ?file=<encoded_path>."""
        try:
            file_path = self.get_argument("file", None)
            if not file_path:
                raise tornado.web.HTTPError(400, "Missing file path")
            db = self.application.database
            ok = db.remove_file_from_list(int(list_id), file_path)
            self.write_json({"removed": ok})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ListFilesHandler DELETE: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ListFilesHandler DELETE: {e}")
            raise tornado.web.HTTPError(500, str(e))


class FavouritePathsHandler(BaseHandler):
    """GET /api/favourites → Return all favourite file paths for the current user."""

    def get(self):
        try:
            db = self.application.database
            self.write_json({"paths": db.get_favourite_paths()})
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in FavouritePathsHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in FavouritePathsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class FileTreeHandler(BaseHandler):
    """GET /api/file-tree?filter=favs|list:N → Filtered cascade tree.

    Returns { tree: { rootLabel: { serial: { step: [{path, filename, size}] } } } }
    containing only entries that lead to at least one matching file.
    """

    def get(self):
        try:
            filter_val = self.get_argument("filter", "all")
            db = self.application.database

            # Determine the set of allowed file paths
            if filter_val == "favs":
                allowed = set(db.get_favourite_paths())
            elif filter_val.startswith("list:"):
                list_id = int(filter_val.split(":")[1])
                allowed = set(db.get_list_files(list_id))
            else:
                self.write_json({"tree": {}})
                return

            if not allowed:
                self.write_json({"tree": {}})
                return

            # Walk all roots/serials/steps/files and keep only matching
            indices = getattr(self.application, 'metadata_indices', {})
            tree = {}
            for root_label, idx in indices.items():
                root_serials = {}
                for serial in idx.get_serial_numbers():
                    serial_steps = {}
                    try:
                        steps = idx.get_steps(serial)
                    except ValueError:
                        continue
                    for step in steps:
                        try:
                            files = idx.get_files(serial, step)
                        except ValueError:
                            continue
                        matched = [f for f in files if f["path"] in allowed]
                        if matched:
                            serial_steps[step] = matched
                    if serial_steps:
                        root_serials[serial] = serial_steps
                if root_serials:
                    tree[root_label] = root_serials
            self.write_json({"tree": tree})
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in FileTreeHandler: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in FileTreeHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class ResolvePathHandler(BaseHandler):
    """GET /api/resolve-path?path=...  → Resolve one file path.
       POST /api/resolve-path (body: {"paths": [...]}) → Batch resolve."""

    def _build_lookup(self):
        """Build a {file_path → {root, serial, step, file}} lookup from all indices."""
        indices = getattr(self.application, 'metadata_indices', {})
        lookup = {}
        for root_label, idx in indices.items():
            for serial in idx.get_serial_numbers():
                try:
                    steps = idx.get_steps(serial)
                except ValueError:
                    continue
                for step in steps:
                    try:
                        files = idx.get_files(serial, step)
                    except ValueError:
                        continue
                    for f in files:
                        lookup[f["path"]] = {
                            "root": root_label,
                            "serial": serial,
                            "step": step,
                            "file": f,
                        }
        return lookup

    def get(self):
        try:
            file_path = self.get_argument("path", "")
            if not file_path:
                raise tornado.web.HTTPError(400, "Missing 'path' parameter")
            lookup = self._build_lookup()
            result = lookup.get(file_path)
            if not result:
                raise tornado.web.HTTPError(404, "File path not found in any index")
            self.write_json(result)
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ResolvePathHandler GET: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ResolvePathHandler GET: {e}")
            raise tornado.web.HTTPError(500, str(e))

    def post(self):
        """Batch resolve. Body: {"paths": ["...", "..."]}
        Returns: {"resolved": { path: {root, serial, step, file} | null, ... }}"""
        try:
            data = json.loads(self.request.body)
            paths = data.get("paths", [])
            if not paths:
                self.write_json({"resolved": {}})
                return
            lookup = self._build_lookup()
            resolved = {}
            for p in paths:
                resolved[p] = lookup.get(p)
            self.write_json({"resolved": resolved})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            if config.VERBOSE:
                logger.error(f"Error in ResolvePathHandler POST: {e}\n{traceback.format_exc()}")
            else:
                logger.error(f"Error in ResolvePathHandler POST: {e}")
            raise tornado.web.HTTPError(500, str(e))
