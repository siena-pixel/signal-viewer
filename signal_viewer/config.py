"""Application configuration."""
import os
from pathlib import Path

# Project root: the directory containing this package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data root: defaults to <project>/data, overridable via env var
DATA_ROOT = Path(os.environ.get('SIGNAL_VIEWER_DATA_ROOT',
                                str(PROJECT_ROOT / 'data')))

# Server
HOST = os.environ.get('SIGNAL_VIEWER_HOST', '127.0.0.1')
PORT = int(os.environ.get('SIGNAL_VIEWER_PORT', '8050'))
DEBUG = os.environ.get('SIGNAL_VIEWER_DEBUG', 'true').lower() == 'true'

# Cache
CACHE_MAX_MEMORY_MB = int(os.environ.get('SIGNAL_VIEWER_CACHE_MB', '500'))
CACHE_MAX_MEMORY_BYTES = CACHE_MAX_MEMORY_MB * 1024 * 1024

# Processing defaults
DEFAULT_DOWNSAMPLE_POINTS = 2000
MAX_DOWNSAMPLE_POINTS = 10000
DEFAULT_FFT_WINDOW = 'hann'
DEFAULT_FILTER_ORDER = 4

# UI
APP_TITLE = 'Engineering Signal Viewer & Analyzer'

# ---------------------------------------------------------------------------
# HDF5 Schema Configuration
# ---------------------------------------------------------------------------
# Change these values to match your HDF5 file structure.
# All readers, indexers, and data generators use this single source of truth.

class HDF5Schema:
    """
    Defines the HDF5 file structure: dataset names, directory conventions,
    and naming patterns.  Every module that touches HDF5 data imports from here.
    """

    # -- Dataset names inside each batch group --------------------------------
    VALUES   = 'value'               # float64 matrix [num_signals, num_samples]
    TIME     = 'time'                # float64 array  [num_samples]
    POSITIONS = 'corrected_positions' # float64 array  [num_samples]
    UNITS    = 'units'               # string array   [num_signals]
    NAMES    = 'name'                # string array   [num_signals]

    # -- Filesystem conventions -----------------------------------------------
    STEP_PREFIX    = 'step_'         # directory prefix, e.g. "step_1", "step_2"
    STEP_REGEX     = r'step_(\d+)'   # regex to extract the step number
    FILE_EXTENSION = '*.h5'          # glob pattern for HDF5 files
    FILE_SUFFIX    = '.h5'           # file suffix for generated files

    # -- Batch naming ---------------------------------------------------------
    BATCH_FORMAT   = 'batch_{:03d}'  # Python format string for batch groups
    DEFAULT_BATCH  = 'batch_001'     # default batch name for tests / mocks
