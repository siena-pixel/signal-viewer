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
from signal_viewer.core.database import Database
from signal_viewer.server.handlers import (
    PageHandler,
    RootsHandler,
    SerialsHandler,
    StepsHandler,
    FilesHandler,
    BatchesHandler,
    BatchMetaHandler,
    SignalHandler,
    StatsHandler,
    CorrelationHandler,
    TrendHandler,
    CacheStatsHandler,
    RescanHandler,
    FavouritesHandler,
    FavouritePathsHandler,
    FileTreeHandler,
    ResolvePathHandler,
    CommentsHandler,
    ListsHandler,
    ListFilesHandler,
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

        # Initialize metadata indices — one MetadataIndex per configured root
        self.metadata_indices: Dict[str, MetadataIndex] = {}
        for label, root_path in config.DATA_ROOTS.items():
            try:
                idx = MetadataIndex(str(root_path))
                self.metadata_indices[label] = idx
                logger.info(f"MetadataIndex initialized: {label} → {idx}")
            except ValueError as e:
                logger.warning(f"Failed to initialize MetadataIndex for '{label}': {e}")

        # Backward-compat alias (first index, used by legacy code paths)
        first_label = next(iter(self.metadata_indices), None)
        self.metadata_index = self.metadata_indices.get(first_label)

        # Initialize signal cache
        self.signal_cache = SignalCache(max_memory_bytes=config.CACHE_MAX_MEMORY_BYTES)
        logger.info(
            f"SignalCache initialized: {config.CACHE_MAX_MEMORY_MB} MB budget"
        )

        # Dict to hold open HDF5Readers (lazy loading)
        self.hdf5_readers: Dict[str, HDF5Reader] = {}

        # SQLite database for favourites, comments, lists
        self.database = Database(str(config.DATABASE_PATH))
        logger.info(f"Database initialized: {config.DATABASE_PATH}")


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
        # API routes — root-aware cascade
        (r"/api/roots", RootsHandler),
        (r"/api/roots/([^/]+)/serials", SerialsHandler),
        (r"/api/roots/([^/]+)/serials/([^/]+)/steps", StepsHandler),
        (r"/api/roots/([^/]+)/serials/([^/]+)/steps/([^/]+)/files", FilesHandler),
        (r"/api/files/([^/]+)/batches", BatchesHandler),
        (r"/api/files/([^/]+)/batches/([^/]+)/meta", BatchMetaHandler),
        (
            r"/api/files/([^/]+)/batches/([^/]+)/signals/(\d+)",
            SignalHandler,
        ),
        (r"/api/analysis/stats", StatsHandler),
        (r"/api/analysis/correlation", CorrelationHandler),
        (r"/api/analysis/trend", TrendHandler),
        (r"/api/cache/stats", CacheStatsHandler),
        (r"/api/rescan", RescanHandler),
        # Favourites / Comments / Lists
        (r"/api/favourites", FavouritePathsHandler),
        (r"/api/favourites/([^/]+)", FavouritesHandler),
        (r"/api/comments/([^/]+)", CommentsHandler),
        (r"/api/lists", ListsHandler),
        (r"/api/lists/(\d+)/files", ListFilesHandler),
        (r"/api/file-tree", FileTreeHandler),
        (r"/api/resolve-path", ResolvePathHandler),
        # Notes & Lists pages
        (r"/comments", PageHandler, {"template": "comments.html"}),
        (r"/lists", PageHandler, {"template": "lists.html"}),
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
    for label, root_path in config.DATA_ROOTS.items():
        logger.info(f"Data root: {label} → {root_path}")
    logger.info(f"Debug mode: {config.DEBUG}")
    logger.info(f"Cache budget: {config.CACHE_MAX_MEMORY_MB} MB")

    try:
        tornado.ioloop.IOLoop.current().start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")


if __name__ == "__main__":
    main()
