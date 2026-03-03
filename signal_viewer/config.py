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

    Folder layout:  root / serial / folder_1 / folder_2 / file.h5
      - serial:   top-level serial number directory (e.g. SN001, SN002)
      - folder_1: pXXX_label  (e.g. p001_motor_test)
      - folder_2: any subfolder name (e.g. run_nominal, warmup)

    HDF5 internal layout:
      /GROUP_NAME/
          GROUP_NAME_V   float64 [num_signals, num_samples]  (values)
          GROUP_NAME_T   float64 [num_samples]               (time)
          GROUP_NAME_P   float64 [num_samples]               (positions)
          GROUP_NAME_N   str     [num_signals]               (signal names)
          GROUP_NAME_U   str     [num_signals]               (units)
    """

    # -- Filesystem conventions -----------------------------------------------
    FOLDER1_REGEX  = r'p(\d{3})_.+'  # regex for folder_1 (e.g. p001_motor)
    FILE_EXTENSION = '*.h5'          # glob pattern for HDF5 files
    FILE_SUFFIX    = '.h5'           # file suffix for generated files

    # -- HDF5 groups of interest ----------------------------------------------
    GROUP_NAMES = ['GROUP_T0', 'GROUP_T1']  # top-level groups to read

    # -- Dataset suffix convention (appended to group name) -------------------
    VALUE_SUFFIX    = '_V'           # float64 matrix [num_signals, num_samples]
    TIME_SUFFIX     = '_T'           # float64 array  [num_samples]
    POSITION_SUFFIX = '_P'           # float64 array  [num_samples]
    NAMES_SUFFIX    = '_N'           # string array   [num_signals]
    UNITS_SUFFIX    = '_U'           # string array   [num_signals]

    # -- Default group for tests / mocks --------------------------------------
    DEFAULT_GROUP   = 'GROUP_T0'

    @classmethod
    def ds(cls, group: str, suffix: str) -> str:
        """Build a dataset name: GROUP_NAME + suffix."""
        return group + suffix
