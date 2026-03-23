"""Application configuration."""
import os
import re
from pathlib import Path

# Project root: the directory containing this package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data roots: dictionary of { display_label: path }
# ---------------------------------------------------------------------------
# The ROOT dropdown in the UI presents the keys; the corresponding values
# are used as the data directory for each root.
#
# Override via env var SIGNAL_VIEWER_DATA_ROOT (sets the "Default" entry).
# Add additional roots by extending this dict.
DATA_ROOTS = {
    'Default': Path(os.environ.get('SIGNAL_VIEWER_DATA_ROOT',
                                   str(PROJECT_ROOT / 'data'))),
    # 'Lab A': Path('/path/to/lab_a/data'),
    # 'Lab B': Path('/path/to/lab_b/data'),
}

# Backward-compat alias: first root path (used internally by tests/legacy code)
DATA_ROOT = list(DATA_ROOTS.values())[0]

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

      Required dataset keys:
        VALUE  — 2D signal data (M signals × N samples)
        NAMES  — 1D signal names (M,); must be present and non-blank for every
                 signal.  Groups with missing NAMES or any blank entry are
                 excluded (treated as invalid).

      Optional dataset keys (graceful fallback when absent):
        TIME              — epoch ms per signal    [default: see TIME_FALLBACK]
        TIME_FALLBACK     — (config only, not an HDF5 dataset) name of another
                            group to borrow TIME from when TIME is absent.
                            Match is by signal name.  Set per group or omit.
        SAMPLING_FREQ — Hz per signal              [default: parsed from VALUE
                        dataset name suffix, e.g. _0050 → 50 Hz, else 1.0]
        NSAMPLE       — valid sample count         [default: full N from VALUE]
        UNITS         — signal units               [default: "" per signal]

      Type B additional (optional) keys: ERROR, SQI, TLS

    Batch types:
      - Type A: no ERROR key configured → base datasets only
      - Type B: ERROR key configured and present → base + error/quality datasets

    Time construction (both types):
      time[i] = t0 + arange(n) * (1000.0 / fs)
      where t0 comes from TIME (or 0.0), n from NSAMPLE (or N),
      and fs from SAMPLING_FREQ (or parsed from VALUE name, or 1.0).

    Empty groups (no VALUE dataset or VALUE with 0 signals/samples) are
    silently ignored.
    """

    # -- Filesystem conventions -----------------------------------------------
    SERIAL_PREFIX  = 'SN'            # only serial dirs starting with this prefix
    FOLDER1_REGEX  = r'p(\d{3})_.+'  # regex for folder_1 (e.g. p001_motor)
    FILE_EXTENSION = '*.h5'          # glob pattern for HDF5 files
    FILE_SUFFIX    = '.h5'           # file suffix for generated files

    # -- Frequency extraction from VALUE dataset name --------------------------
    # If SAMPLING_FREQ is not configured or absent from the file, the reader
    # tries to extract Hz from the VALUE dataset name.  The trailing digits
    # after the last underscore are interpreted as integer Hz.
    #   e.g.  GROUP_T1_V_0050  →  50 Hz
    FREQ_SUFFIX_REGEX = r'_(\d+)$'

    # -- Per-group dataset name mapping ---------------------------------------
    # Required keys: VALUE              (must be present in HDF5 file)
    #                NAMES              (must be present; all entries non-blank)
    # Optional:      TIME, SAMPLING_FREQ, NSAMPLE, UNITS
    # Type B only:   ERROR, SQI, TLS
    #
    # Each group maps these logical keys to the actual HDF5 dataset name.
    # Group names are derived automatically: list(GROUP_DS_NAMES.keys()).

    # -- TIME fallback --------------------------------------------------------
    # When TIME is missing for a group, the reader tries to borrow the time
    # value from a signal with the same name in the group specified by
    # TIME_FALLBACK (a per-group config key, not an HDF5 dataset).
    # Omit or set to None to disable (defaults to t0 = 0.0).
    GROUP_DS_NAMES = {
        'GROUP_T0': {
            'VALUE':         'GROUP_T0_V',
            'TIME':          'GROUP_T0_TIM',
            'SAMPLING_FREQ': 'GROUP_T0_FRE',
            'NSAMPLE':       'GROUP_T0_SAM',
            'NAMES':         'GROUP_T0_N',
            'UNITS':         'GROUP_T0_UNI',
            # Type A — no ERROR/SQI/TLS
            # 'TIME_FALLBACK': 'GROUP_T1',  # example: borrow TIME from GROUP_T1
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
            # 'TIME_FALLBACK': 'GROUP_T0',  # example: borrow TIME from GROUP_T0
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

    @classmethod
    def parse_freq_from_name(cls, ds_name):
        """Extract sampling frequency (Hz) from a dataset name suffix.

        Matches trailing ``_DDDD`` where DDDD is one or more digits.
        Returns the integer value as a float, or None if no match.

        Examples:
            'GROUP_T1_V_0050' → 50.0
            'GROUP_T0_V'      → None
        """
        m = re.search(cls.FREQ_SUFFIX_REGEX, ds_name)
        return float(int(m.group(1))) if m else None
