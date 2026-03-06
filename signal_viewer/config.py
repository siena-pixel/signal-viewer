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

    HDF5 internal layout (per group):
      /GROUP_NAME/
          GROUP_NAME_V    float64 [n_signals, max(n_samples)]  signal values
          GROUP_NAME_TIM  float64 [n_signals]  epoch time (ms) of first sample
          GROUP_NAME_FRE  float64 [n_signals]  sampling frequency (Hz)
          GROUP_NAME_SAM  int64   [n_signals]  valid sample count per signal
          GROUP_NAME_N    str     [n_signals]  signal names
          GROUP_NAME_UNI  str     [n_signals]  units

      Type B groups additionally contain:
          GROUP_NAME_ERR  float64 [n_signals, max(n_samples)]  signal errors
          GROUP_NAME_SQI  float64 [n_signals]  signal quality metric
          GROUP_NAME_TLS  float64 [n_signals]  max error gap (seconds)

    Batch types:
      - Type A: only the base datasets above
      - Type B: base datasets + _ERR, _SQI, _TLS
      Both types have _SAM (valid sample count per signal).

    Time construction (both types):
      time[i] = TIM[i] + arange(length) / FRE[i]
      where TIM is epoch time in milliseconds.
    """

    # -- Filesystem conventions -----------------------------------------------
    SERIAL_PREFIX  = 'SN'            # only serial dirs starting with this prefix
    FOLDER1_REGEX  = r'p(\d{3})_.+'  # regex for folder_1 (e.g. p001_motor)
    FILE_EXTENSION = '*.h5'          # glob pattern for HDF5 files
    FILE_SUFFIX    = '.h5'           # file suffix for generated files

    # -- HDF5 groups of interest ----------------------------------------------
    GROUP_NAMES = ['GROUP_T0', 'GROUP_T1']  # top-level groups to read

    # -- Dataset suffix convention (appended to group name) -------------------
    # Common to both Type A and Type B:
    VALUE_SUFFIX         = '_V'      # float64 [n_signals, max(n_samples)]
    TIME_SUFFIX          = '_TIM'    # float64 [n_signals]  epoch ms of first sample
    SAMPLING_FREQ_SUFFIX = '_FRE'    # float64 [n_signals]  sampling frequency (Hz)
    NSAMPLE_SUFFIX       = '_SAM'    # int64   [n_signals]  valid sample count
    NAMES_SUFFIX         = '_N'      # str     [n_signals]  signal names
    UNITS_SUFFIX         = '_UNI'    # str     [n_signals]  units

    # Type B additional datasets:
    ERROR_SUFFIX         = '_ERR'    # float64 [n_signals, max(n_samples)]  errors
    SQI_SUFFIX           = '_SQI'    # float64 [n_signals]  signal quality index
    TLS_SUFFIX           = '_TLS'    # float64 [n_signals]  max error gap (seconds)

    # -- Default group for tests / mocks --------------------------------------
    DEFAULT_GROUP   = 'GROUP_T0'

    @classmethod
    def ds(cls, group: str, suffix: str) -> str:
        """Build a dataset name: GROUP_NAME + suffix."""
        return group + suffix
