"""
Tornado Request Handlers for Engineering Time Series Signal Viewer API

Includes:
  - PageHandler: Renders Jinja2 templates for HTML pages
  - API Handlers: Return JSON responses for data operations
  - Error handling and numpy array serialization
"""

import base64
import json
import logging
import traceback
from typing import Optional

import numpy as np
import tornado.web
import tornado.iostream

from signal_viewer import config
from signal_viewer.core.hdf5_reader import HDF5Reader
from signal_viewer.processing import (
    resampling,
    spectral,
    filtering,
    statistics,
    anomaly,
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
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
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

    def decode_file_path(self, encoded_path: str) -> str:
        """
        Decode base64-encoded file path from URL.

        Args:
            encoded_path: Base64-encoded file path

        Returns:
            Decoded file path

        Raises:
            ValueError: If decoding fails
        """
        try:
            # Re-add padding stripped by JS encodePath
            padded = encoded_path + "=" * (-len(encoded_path) % 4)
            decoded = base64.urlsafe_b64decode(padded)
            return decoded.decode("utf-8")
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
            self.set_status(500)
            self.write_json({"error": f"Template error: {e}"})


class SerialsHandler(BaseHandler):
    """GET /api/serials → List all serial numbers."""

    def get(self):
        """Return list of serial numbers."""
        try:
            if self.application.metadata_index is None:
                self.write_json({"serials": [], "error": "Index not initialized"})
                return

            serials = self.application.metadata_index.get_serial_numbers()
            self.write_json({"serials": serials})
        except Exception as e:
            logger.error(f"Error in SerialsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class StepsHandler(BaseHandler):
    """GET /api/serials/{serial_num}/steps → List steps for a serial number."""

    def get(self, serial_num: str):
        """
        Get steps for a serial number.

        Args:
            serial_num: Serial number identifier
        """
        try:
            if self.application.metadata_index is None:
                raise tornado.web.HTTPError(503, "Index not initialized")

            steps = self.application.metadata_index.get_steps(serial_num)
            self.write_json({"steps": steps})
        except ValueError as e:
            raise tornado.web.HTTPError(404, str(e))
        except Exception as e:
            logger.error(f"Error in StepsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class FilesHandler(BaseHandler):
    """GET /api/serials/{serial_num}/steps/{step}/files → List files in a step."""

    def get(self, serial_num: str, step: str):
        """
        Get files for a serial number and step.

        Args:
            serial_num: Serial number identifier
            step: Step number (as string)
        """
        try:
            if self.application.metadata_index is None:
                raise tornado.web.HTTPError(503, "Index not initialized")

            step_int = int(step)
            files = self.application.metadata_index.get_files(
                serial_num, step_int
            )
            self.write_json({"files": files})
        except ValueError as e:
            raise tornado.web.HTTPError(404, str(e))
        except Exception as e:
            logger.error(f"Error in FilesHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class BatchesHandler(BaseHandler):
    """GET /api/files/{encoded_path}/batches → List batches in a file."""

    def get(self, encoded_path: str):
        """
        Get batches in an HDF5 file.

        Args:
            encoded_path: Base64-encoded file path
        """
        try:
            file_path = self.decode_file_path(encoded_path)
            reader = self.get_hdf5_reader(file_path)

            batches = reader.get_batches()
            self.write_json({"batches": batches})
        except tornado.web.HTTPError:
            raise
        except Exception as e:
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
            logger.error(f"Error in BatchMetaHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class SignalHandler(BaseHandler):
    """GET /api/files/{encoded_path}/batches/{batch}/signals/{idx} → Signal data."""

    def get(self, encoded_path: str, batch: str, signal_idx: str):
        """
        Load a signal with optional downsampling.

        Query parameters:
          - downsample: Target number of points (default: 2000, max: 10000)

        Args:
            encoded_path: Base64-encoded file path
            batch: Batch name
            signal_idx: Signal index (0-based)
        """
        try:
            file_path = self.decode_file_path(encoded_path)
            reader = self.get_hdf5_reader(file_path)
            idx = int(signal_idx)

            # Get cache key
            cache_key = f"{file_path}::{batch}::{idx}"

            # Check cache first
            cached = self.application.signal_cache.get(cache_key)
            if cached is not None:
                time, signal = cached
            else:
                # Load from file
                time, signal = reader.load_signal(batch, idx)
                # Cache the full signal
                self.application.signal_cache.put(cache_key, time, signal)

            # Apply downsampling if requested
            downsample_param = self.get_argument("downsample", None)
            if downsample_param:
                try:
                    target_points = int(downsample_param)
                    # Clamp to bounds
                    target_points = min(
                        max(target_points, 10), config.MAX_DOWNSAMPLE_POINTS
                    )
                    time, signal = resampling.lttb_downsample(
                        time, signal, target_points
                    )
                except (ValueError, TypeError):
                    pass

            # Get signal metadata
            batch_meta = reader.get_batch_metadata(batch)
            signal_name = batch_meta["signal_names"][idx]
            units = batch_meta["units"][idx]

            self.write_json(
                {
                    "time": time,
                    "values": signal,
                    "name": signal_name,
                    "units": units,
                    "samples": len(signal),
                }
            )
        except tornado.web.HTTPError:
            raise
        except (ValueError, IndexError) as e:
            raise tornado.web.HTTPError(404, str(e))
        except Exception as e:
            logger.error(f"Error in SignalHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class FFTHandler(BaseHandler):
    """POST /api/analysis/fft → Compute FFT."""

    def post(self):
        """
        Compute FFT of a signal.

        Request body:
        {
          "file_path": "/path/to/file.h5",
          "batch": "batch_001",
          "signal_idx": 0,
          "sampling_rate": 1000.0
        }
        """
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path")
            batch = data.get("batch")
            signal_idx = int(data.get("signal_idx", 0))
            sampling_rate = float(data.get("sampling_rate", 1000.0))

            if not file_path or not batch:
                raise tornado.web.HTTPError(400, "Missing required fields")

            reader = self.get_hdf5_reader(file_path)
            time, signal = reader.load_signal(batch, signal_idx)

            # Compute FFT
            frequencies, magnitude_db = spectral.compute_fft(
                signal, sampling_rate
            )

            self.write_json(
                {
                    "frequencies": frequencies,
                    "magnitude": magnitude_db,
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            logger.error(f"Error in FFTHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class PSDHandler(BaseHandler):
    """POST /api/analysis/psd → Compute Power Spectral Density."""

    def post(self):
        """
        Compute PSD using Welch's method.

        Request body:
        {
          "file_path": "/path/to/file.h5",
          "batch": "batch_001",
          "signal_idx": 0,
          "sampling_rate": 1000.0,
          "nperseg": 1024,
          "window": "hann"
        }
        """
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path")
            batch = data.get("batch")
            signal_idx = int(data.get("signal_idx", 0))
            sampling_rate = float(data.get("sampling_rate", 1000.0))
            nperseg = int(data.get("nperseg", 1024))
            window = data.get("window", config.DEFAULT_FFT_WINDOW)

            if not file_path or not batch:
                raise tornado.web.HTTPError(400, "Missing required fields")

            reader = self.get_hdf5_reader(file_path)
            time, signal = reader.load_signal(batch, signal_idx)

            # Compute PSD
            frequencies, psd = spectral.compute_psd_welch(
                signal, sampling_rate, nperseg=nperseg, window=window
            )

            # Find dominant frequencies
            peaks = spectral.find_dominant_frequencies(frequencies, psd, n_peaks=5)

            self.write_json(
                {
                    "frequencies": frequencies,
                    "psd": psd,
                    "dominant_peaks": peaks,
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            logger.error(f"Error in PSDHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class FilterHandler(BaseHandler):
    """POST /api/analysis/filter → Apply digital filter."""

    def post(self):
        """
        Apply Butterworth filter to a signal.

        Request body:
        {
          "file_path": "/path/to/file.h5",
          "batch": "batch_001",
          "signal_idx": 0,
          "type": "lowpass",
          "cutoff": 100.0,
          "low_cutoff": 50.0,
          "high_cutoff": 200.0,
          "sampling_rate": 1000.0,
          "order": 4
        }

        filter_type options: "lowpass", "highpass", "bandpass"
        """
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path")
            batch = data.get("batch")
            signal_idx = int(data.get("signal_idx", 0))
            filter_type = data.get("type", "lowpass")
            sampling_rate = float(data.get("sampling_rate", 1000.0))
            order = int(data.get("order", config.DEFAULT_FILTER_ORDER))

            if not file_path or not batch:
                raise tornado.web.HTTPError(400, "Missing required fields")

            reader = self.get_hdf5_reader(file_path)
            time, signal = reader.load_signal(batch, signal_idx)

            # Apply filter based on type
            if filter_type == "lowpass":
                cutoff = data.get("cutoff")
                if cutoff is None:
                    raise tornado.web.HTTPError(400, "Missing 'cutoff' parameter for lowpass filter")
                filtered = filtering.butterworth_lowpass(
                    signal, sampling_rate, float(cutoff), order
                )
            elif filter_type == "highpass":
                cutoff = data.get("cutoff")
                if cutoff is None:
                    raise tornado.web.HTTPError(400, "Missing 'cutoff' parameter for highpass filter")
                filtered = filtering.butterworth_highpass(
                    signal, sampling_rate, float(cutoff), order
                )
            elif filter_type == "bandpass":
                low_cutoff = data.get("low_cutoff")
                high_cutoff = data.get("high_cutoff")
                if low_cutoff is None or high_cutoff is None:
                    raise tornado.web.HTTPError(400, "Missing 'low_cutoff' or 'high_cutoff' parameter for bandpass filter")
                filtered = filtering.butterworth_bandpass(
                    signal, sampling_rate, float(low_cutoff), float(high_cutoff), order
                )
            else:
                raise tornado.web.HTTPError(400, f"Unknown filter type: {filter_type}")

            self.write_json(
                {
                    "time": time,
                    "filtered": filtered,
                    "original": signal,
                    "filter_type": filter_type,
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            logger.error(f"Error in FilterHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class AnomalyHandler(BaseHandler):
    """POST /api/analysis/anomaly → Detect anomalies."""

    def post(self):
        """
        Detect anomalies in a signal.

        Request body:
        {
          "file_path": "/path/to/file.h5",
          "batch": "batch_001",
          "signal_idx": 0,
          "method": "zscore",
          "threshold": 3.0
        }

        method options: "zscore", "mad", "derivative", "iqr", "rolling"
        """
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path")
            batch = data.get("batch")
            signal_idx = int(data.get("signal_idx", 0))
            method = data.get("method", "zscore")
            threshold = float(data.get("threshold", 3.0))

            if not file_path or not batch:
                raise tornado.web.HTTPError(400, "Missing required fields")

            reader = self.get_hdf5_reader(file_path)
            time, signal = reader.load_signal(batch, signal_idx)

            # Detect anomalies based on method
            if method == "zscore":
                anomaly_indices, scores = anomaly.zscore_anomaly(
                    signal, threshold
                )
            elif method == "mad":
                anomaly_indices, scores = anomaly.mad_anomaly(signal, threshold)
            elif method == "derivative":
                anomaly_indices, scores = anomaly.derivative_anomaly(
                    signal, threshold
                )
            elif method == "iqr":
                anomaly_indices, scores = anomaly.iqr_anomaly(
                    signal, threshold
                )
            elif method == "rolling":
                window_size = int(data.get("window_size", 100))
                anomaly_indices, scores = anomaly.rolling_anomaly(
                    signal, window_size, threshold
                )
            else:
                raise tornado.web.HTTPError(400, f"Unknown method: {method}")

            self.write_json(
                {
                    "anomaly_indices": anomaly_indices,
                    "scores": scores,
                    "count": len(anomaly_indices),
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            logger.error(f"Error in AnomalyHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class StatsHandler(BaseHandler):
    """POST /api/analysis/stats → Compute signal statistics."""

    def post(self):
        """
        Compute descriptive statistics and rainflow cycle counting for a signal.

        Request body:
        {
          "file_path": "/path/to/file.h5",
          "batch": "batch_001",
          "signal_idx": 0,
          "rainflow_bins": 10
        }

        Optional parameters:
          - rainflow_bins: Number of bins for rainflow histogram (default: 10)
        """
        try:
            data = json.loads(self.request.body)
            file_path = data.get("file_path")
            batch = data.get("batch")
            signal_idx = int(data.get("signal_idx", 0))
            rainflow_bins = int(data.get("rainflow_bins", 10))

            if not file_path or not batch:
                raise tornado.web.HTTPError(400, "Missing required fields")

            reader = self.get_hdf5_reader(file_path)
            time, signal = reader.load_signal(batch, signal_idx)

            # Compute descriptive statistics
            stats = statistics.compute_descriptive_stats(signal)

            # Compute rainflow cycle counting
            rainflow_data = statistics.compute_rainflow(signal, n_bins=rainflow_bins)

            # Combine results
            result = {
                **stats,
                "rainflow": rainflow_data,
            }

            self.write_json(result)
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            logger.error(f"Error in StatsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class CorrelationHandler(BaseHandler):
    """POST /api/analysis/correlation → Compute cross-correlation."""

    def post(self):
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

            # Compute cross-correlation
            lags, corr = correlation.cross_correlate(
                signal_a, signal_b, mode="same"
            )

            # Find lag at maximum correlation
            max_lag = correlation.find_lag(signal_a, signal_b)

            self.write_json(
                {
                    "lags": lags,
                    "correlation": corr,
                    "max_lag": max_lag,
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            logger.error(f"Error in CorrelationHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


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
            logger.error(f"Error in CacheStatsHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))


class RescanHandler(BaseHandler):
    """POST /api/rescan → Re-scan filesystem for new files."""

    def post(self):
        """
        Re-scan the data root directory to detect new or removed files.

        Returns metadata about the rescan operation.
        """
        try:
            if self.application.metadata_index is None:
                raise tornado.web.HTTPError(503, "Index not initialized")

            # Re-scan filesystem
            self.application.metadata_index.rescan()

            # Get updated statistics
            serials = self.application.metadata_index.get_serial_numbers()
            total_steps = sum(
                len(self.application.metadata_index.get_steps(s))
                for s in serials
            )

            self.write_json(
                {
                    "status": "success",
                    "serial_count": len(serials),
                    "total_steps": total_steps,
                }
            )
        except tornado.web.HTTPError:
            raise
        except Exception as e:
            logger.error(f"Error in RescanHandler: {e}")
            raise tornado.web.HTTPError(500, str(e))
