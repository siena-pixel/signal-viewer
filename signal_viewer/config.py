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

# Verbose mode: when True, log detailed tracebacks for every error to the CLI
VERBOSE = os.environ.get('SIGNAL_VIEWER_VERBOSE', 'false').lower() == 'true'

# Cache
CACHE_MAX_MEMORY_MB = int(os.environ.get('SIGNAL_VIEWER_CACHE_MB', '500'))
CACHE_MAX_MEMORY_BYTES = CACHE_MAX_MEMORY_MB * 1024 * 1024

# Processing defaults
DEFAULT_DOWNSAMPLE_POINTS = 5000
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
      Each group can have its own dataset names, configured via GROUP_DS_NAMES.
      Common (required) dataset keys: VALUE, TIME, SAMPLING_FREQ, NSAMPLE, NAMES, UNITS
      Type B additional (optional) keys: ERROR, SQI, TLS

    Batch types:
      - Type A: only the common datasets (no ERROR key or ERROR maps to None)
      - Type B: common + ERROR, SQI, TLS
      Both types have NSAMPLE (valid sample count per signal).

    Time construction (both types):
      time[i] = TIM[i] + arange(length) * (1000.0 / FRE[i])
      where TIM is epoch time in milliseconds and FRE is in Hz.
    """

    # -- Filesystem conventions -----------------------------------------------
    SERIAL_PREFIX  = 'SN'            # only serial dirs starting with this prefix
    FOLDER1_REGEX  = r'p(\d{3})_.+'  # regex for folder_1 (e.g. p001_motor)
    FILE_EXTENSION = '*.h5'          # glob pattern for HDF5 files
    FILE_SUFFIX    = '.h5'           # file suffix for generated files

    # -- Per-group dataset name mapping ---------------------------------------
    # Keys: VALUE, TIME, SAMPLING_FREQ, NSAMPLE, NAMES, UNITS  (required)
    #        ERROR, SQI, TLS  (optional — present in Type B groups only)
    #
    # Each group maps these logical keys to the actual HDF5 dataset name.
    # Group names are derived automatically: list(GROUP_DS_NAMES.keys()).
    GROUP_DS_NAMES = {
        'GROUP_T0': {
            'VALUE':         'GROUP_T0_V',
            'TIME':          'GROUP_T0_TIM',
            'SAMPLING_FREQ': 'GROUP_T0_FRE',
            'NSAMPLE':       'GROUP_T0_SAM',
            'NAMES':         'GROUP_T0_N',
            'UNITS':         'GROUP_T0_UNI',
            # Type A — no ERROR/SQI/TLS
        },
        'GROUP_T1': {
            'VALUE':         'GROUP_T1_V',
            'TIME':          'GROUP_T1_TIM',
            'SAMPLING_FREQ': 'GROUP_T1_FRE',
            'NSAMPLE':       'GROUP_T1_SAM',
            'NAMES':         'GROUP_T1_N',
            'UNITS':         'GROUP_T1_UNI',
            # Type B — additional datasets
            'ERROR':         'GROUP_T1_ERR',
            'SQI':           'GROUP_T1_SQI',
            'TLS':           'GROUP_T1_TLS',
        },
    }

    # -- Derived helpers (read-only) ------------------------------------------

    @classmethod
    def group_names(cls):
        """Return list of configured group names."""
        return list(cls.GROUP_DS_NAMES.keys())

    @classmethod
    def ds(cls, group, key):
        """Look up the actual HDF5 dataset name for a group and logical key."""
        return cls.GROUP_DS_NAMES[group][key]

    @classmethod
    def has_ds(cls, group, key):
        """Check whether a logical key is configured for a group."""
        return key in cls.GROUP_DS_NAMES.get(group, {})

    @classmethod
    def default_group(cls):
        """Return the first configured group name (used in tests/mocks)."""
        return cls.group_names()[0]
