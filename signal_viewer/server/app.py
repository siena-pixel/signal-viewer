"""
Tornado Web Server for Engineering Time Series Signal Viewer

Main application entry point. Configures routes, initializes shared context
(MetadataIndex, SignalCache, HDF5Readers), and starts the server.
"""

import logging
import os
from pathlib import Path
from typing import Dict

import tornado.web
from jinja2 import Environment, FileSystemLoader

from signal_viewer import config
from signal_viewer.core.metadata_index import MetadataIndex
from signal_viewer.core.signal_cache import SignalCache
from signal_viewer.core.hdf5_reader import HDF5Reader
from signal_viewer.server.handlers import (
    PageHandler,
    SerialsHandler,
    StepsHandler,
    FilesHandler,
    BatchesHandler,
    BatchMetaHandler,
    SignalHandler,
    FFTHandler,
    PSDHandler,
    FilterHandler,
    AnomalyHandler,
    StatsHandler,
    CorrelationHandler,
    TrendHandler,
    CacheStatsHandler,
    RescanHandler,
)

logger = logging.getLogger(__name__)


class Application(tornado.web.Application):
    """
    Custom Tornado Application with shared context for handlers.

    Stores:
      - metadata_index: MetadataIndex instance for filesystem scanning
      - signal_cache: SignalCache for caching loaded signals
      - hdf5_readers: Dict of {file_path -> HDF5Reader} for open file handles
    """

    def __init__(self, *args, **kwargs):
        """Initialize application with shared context."""
        super().__init__(*args, **kwargs)

        # Initialize metadata index (scans DATA_ROOT)
        try:
            self.metadata_index = MetadataIndex(str(config.DATA_ROOT))
            logger.info(f"MetadataIndex initialized: {self.metadata_index}")
        except ValueError as e:
            logger.warning(f"Failed to initialize MetadataIndex: {e}")
            self.metadata_index = None

        # Initialize signal cache
        self.signal_cache = SignalCache(max_memory_bytes=config.CACHE_MAX_MEMORY_BYTES)
        logger.info(
            f"SignalCache initialized: {config.CACHE_MAX_MEMORY_MB} MB budget"
        )

        # Dict to hold open HDF5Readers (lazy loading)
        self.hdf5_readers: Dict[str, HDF5Reader] = {}


def make_app():
    """
    Create and configure Tornado application with all routes and handlers.

    Returns:
        tornado.web.Application instance
    """
    # Setup Jinja2 template loader
    template_path = Path(__file__).parent.parent / "templates"
    jinja_env = Environment(
        loader=FileSystemLoader(str(template_path)), autoescape=True
    )

    # Setup static files
    static_path = Path(__file__).parent.parent.parent / "static"

    # Define URL routes
    routes = [
        # Page routes (render HTML templates)
        (r"/", PageHandler, {"template": "viewer.html"}),
        (r"/analysis", PageHandler, {"template": "analysis.html"}),
        (r"/comparison", PageHandler, {"template": "comparison.html"}),
        (r"/docs", PageHandler, {"template": "documentation.html"}),
        # API routes
        (r"/api/serials", SerialsHandler),
        (r"/api/serials/([^/]+)/steps", StepsHandler),
        (r"/api/serials/([^/]+)/steps/(\d+)/files", FilesHandler),
        (r"/api/files/([^/]+)/batches", BatchesHandler),
        (r"/api/files/([^/]+)/batches/([^/]+)/meta", BatchMetaHandler),
        (
            r"/api/files/([^/]+)/batches/([^/]+)/signals/(\d+)",
            SignalHandler,
        ),
        (r"/api/analysis/fft", FFTHandler),
        (r"/api/analysis/psd", PSDHandler),
        (r"/api/analysis/filter", FilterHandler),
        (r"/api/analysis/anomaly", AnomalyHandler),
        (r"/api/analysis/stats", StatsHandler),
        (r"/api/analysis/correlation", CorrelationHandler),
        (r"/api/analysis/trend", TrendHandler),
        (r"/api/cache/stats", CacheStatsHandler),
        (r"/api/rescan", RescanHandler),
    ]

    # Create application with settings
    app = Application(
        routes,
        template_loader=tornado.template.Loader(str(template_path)),
        static_path=str(static_path),
        debug=config.DEBUG,
        autoreload=config.DEBUG,
    )

    # Attach Jinja2 environment to app
    app.jinja_env = jinja_env

    return app


def main():
    """
    Start the Tornado web server.

    Binds to HOST:PORT and runs the IOLoop indefinitely.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = make_app()
    app.listen(config.PORT, config.HOST)

    logger.info(f"Starting server on {config.HOST}:{config.PORT}")
    logger.info(f"Data root: {config.DATA_ROOT}")
    logger.info(f"Debug mode: {config.DEBUG}")
    logger.info(f"Cache budget: {config.CACHE_MAX_MEMORY_MB} MB")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")


if __name__ == "__main__":
    main()
