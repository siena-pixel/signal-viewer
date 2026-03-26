# Engineering Signal Viewer & Analyzer

A web-based tool for exploring, visualizing, and analyzing time-series engineering data stored in HDF5 format. Built with Python/Tornado on the backend and Plotly.js on the frontend.

```
┌──────────────────────────────────────────────────────────────────────┐
│  Root: [▾]  Serial: [▾]  Step: [▾]  File: [▾]  ♥  File Lists: [▾]  │
├──────────────────────────────────────────────────────────────────────┤
│  Viewer │ Analysis │ Comparison │ Notes │ Lists │ Docs              │
├──────────────────────────────────────────────────────────────────────┤
│ Signals │                                                           │
│ ▸ GRP_T0│  Signal Plot (Plotly scattergl, adaptive zoom)            │
│ ▸ GRP_T1│  ┌──────────────────────────────────────────────┐         │
│         │  │ ╱╲╱╲  ╱╲╱╲         Time: absolute | relative │         │
│         │  │╱  ╲╱  ╲╱  ╲        Layout: overlay | subplots│         │
│         │  └──────────────────────────────────────────────┘         │
│         │  SIGNAL CARDS   [card1] [card2] [card3]                   │
└──────────────────────────────────────────────────────────────────────┘
```

## Features

- **Data Explorer** — cascading type-to-filter dropdowns (Root → Serial → Step → File) with session-state persistence; supports multiple data root directories
- **Interactive Plots** — overlay or subplot layout modes with WebGL-accelerated rendering (Plotly scattergl); absolute and relative (H:MM:SS) time axes
- **Adaptive Zoom** — click-drag to zoom; the viewer re-fetches data at higher resolution via server-side MinMax-LTTB downsampling with binary-search windowing
- **Favourites** — mark files with a heart icon; filter the cascade to show only favourited files
- **File Lists** — create named lists of files; filter the cascade to show only files in a list; cascade dropdowns are pruned to only display entries leading to matching files
- **Notes** — per-file private notes (one per user) and public notes (visible to all users) with author attribution
- **Lists Management** — dedicated page to create, manage, and populate named file collections
- **Statistics** — descriptive stats, value histograms, and ASTM E1049 rainflow cycle counting
- **Trend Analysis** — polynomial fitting, envelope estimation, RMS trend, changepoint detection
- **Correlation** — cross-correlation with lag detection and correlation strength quantification
- **Signal Comparison** — overlay, X-vs-Y scatter, combined histogram, and cross-correlation views for two signals
- **Performance** — in-memory LRU signal cache with configurable budget, two-pass MinMax-LTTB downsampling, binary-search windowed queries
- **SQLite Backend** — favourites, notes, and lists stored in a WAL-mode SQLite database, scoped per OS user
- **REST API** — all data served through JSON endpoints; file paths encoded as base64url in URL segments


## Quick Start

### 1. Clone and set up

```bash
git clone <repo-url>
cd 03_hdf5_signal_viewer
bash create_venv.sh
source venv/bin/activate
```

### 2. Generate sample data (optional)

```bash
python3 generate_dummy_data.py
```

Creates three serial numbers with multiple processing steps and realistic engineering signals (vibration, temperature, pressure, motor current, strain, etc.).

### 3. Start the server

```bash
bash run.sh
# or manually:
source venv/bin/activate
python3 -m signal_viewer.server.app
```

Open **http://127.0.0.1:8050** in your browser.


## Project Structure

```
03_hdf5_signal_viewer/
├── signal_viewer/
│   ├── config.py                  # All configuration (data roots, DB path, HDF5 schema)
│   ├── core/
│   │   ├── database.py            # SQLite backend (favourites, notes, lists)
│   │   ├── hdf5_reader.py         # HDF5 file reader with schema-driven dataset mapping
│   │   ├── metadata_index.py      # Filesystem index (serial → step → file tree)
│   │   └── signal_cache.py        # LRU memory cache with configurable budget
│   ├── processing/
│   │   ├── correlation.py         # Cross-correlation and coherence
│   │   ├── resampling.py          # MinMax-LTTB downsampling
│   │   ├── statistics.py          # Descriptive stats, histograms, rainflow
│   │   └── trend.py               # Polynomial fitting, RMS, changepoints
│   ├── server/
│   │   ├── app.py                 # Tornado application factory and route table
│   │   └── handlers.py            # All API request handlers
│   └── templates/
│       ├── base.html              # Master template (file selector, navbar, sidebar)
│       ├── viewer.html            # Signal plotting with adaptive zoom engine
│       ├── analysis.html          # Statistical analysis page
│       ├── comparison.html        # Two-signal comparison page
│       ├── comments.html          # Notes page (private + public)
│       ├── lists.html             # File list management page
│       └── documentation.html     # In-app documentation
├── static/
│   ├── css/theme.css              # Complete CSS (variables, layout, all components)
│   └── js/
│       ├── app.js                 # Core module (API helpers, GlobalNav, FilterSelect, sidebar)
│       └── vendor/plotly.min.js   # Plotly.js (vendored)
├── tests/                         # 247 unit/integration tests
│   ├── test_database.py           # SQLite backend tests
│   ├── test_hdf5_reader.py        # HDF5 reader tests
│   ├── test_metadata_index.py     # Filesystem index tests
│   ├── test_signal_cache.py       # Cache tests
│   ├── test_resampling.py         # Downsampling tests
│   ├── test_statistics.py         # Statistics/rainflow tests
│   ├── test_trend.py              # Trend analysis tests
│   ├── test_correlation.py        # Correlation tests
│   └── test_integration.py        # Full Tornado server integration tests
├── generate_dummy_data.py         # Sample data generator
├── create_venv.sh                 # Virtual environment setup script
├── run.sh                         # Server launch script
├── requirements.txt               # Python dependencies
└── pyproject.toml                 # Package metadata
```


## Architecture

### Backend

The server is a single-process **Tornado** application. Route handlers in `handlers.py` delegate to:

- **MetadataIndex** — scans data root directories on startup, builds an in-memory `{serial: {step: [file_dicts]}}` tree. Thread-safe with a read-write lock.
- **HDF5Reader** — opens files with `h5py`, reads datasets according to the schema defined in `config.py`. Constructs time vectors from `t0 + arange(n) × (1000/fs)`.
- **SignalCache** — LRU cache keyed by `(file_path, batch, signal_index)`. Stores full signal arrays in memory up to the configured budget. Zoom queries slice cached arrays with `numpy.searchsorted` (binary search).
- **Database** — SQLite with WAL mode and foreign keys. Tables: `favourites`, `comments`, `lists`, `list_files`. All queries scoped by OS username (`getpass.getuser()`).
- **Processing modules** — pure NumPy/SciPy functions for statistics, resampling, correlation, and trend analysis. Stateless — called on demand by handlers.

### Frontend

A single-page-like application using Jinja2 templates with shared state:

- **`app.js`** — core module loaded on every page. Provides `GlobalNav` (cascading dropdown navigation with `FilterSelect` widget), `apiFetch`/`apiPost` helpers, `URLState` persistence, sidebar management, loading overlay, and toast notifications.
- **`FilterSelect`** — custom combobox widget replacing native `<select>` with type-to-filter, keyboard navigation, and a portal dropdown appended to `document.body` with `position: fixed`.
- **Page scripts** — each template's `{% block extra_js %}` contains page-specific logic. Pages register `GlobalNav.onFileSelected` and `GlobalNav.onSelectionChange` callbacks.
- **Plotly.js** — all plotting uses `type: 'linear'` x-axis with custom `tickvals/ticktext` to avoid timezone parsing bugs. Epoch-ms values are passed as plain numbers; Plotly never converts them to date strings.

### Adaptive Zoom Engine (viewer.html)

The zoom system re-fetches data from the server whenever the user zooms in:

1. User drags to zoom → Plotly fires `plotly_relayout` with new x-axis range
2. After a 150ms debounce, the client sends parallel `GET /api/files/{path}/batches/{batch}/signals/{idx}?downsample=8000&t_min=...&t_max=...` requests for each loaded signal
3. The server binary-searches the cached signal array, slices the window, and applies MinMax-LTTB downsampling
4. The client replaces signal data and re-renders via `Plotly.newPlot()` (full WebGL buffer refresh)
5. Echo detection: `_lastPlotRange` comparison suppresses the relayout event that Plotly fires after our own `newPlot()`

In relative time mode, relayout coordinates (seconds) are converted back to epoch-ms before the API call, and the response is converted back to relative seconds for display.


## Configuration

All configuration lives in `signal_viewer/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNAL_VIEWER_DATA_ROOT` | `<project>/data/` | Default root directory for HDF5 files (sets the "Default" entry in `DATA_ROOTS`) |
| `SIGNAL_VIEWER_DB_PATH` | `<project>/signal_viewer.db3` | SQLite database file for favourites, notes, and lists |
| `SIGNAL_VIEWER_HOST` | `127.0.0.1` | Server bind address |
| `SIGNAL_VIEWER_PORT` | `8050` | Server port |
| `SIGNAL_VIEWER_DEBUG` | `true` | Enable debug mode with auto-reload |
| `SIGNAL_VIEWER_VERBOSE` | `false` | Log detailed error tracebacks to the console |
| `SIGNAL_VIEWER_CACHE_MB` | `500` | Signal cache memory budget in MB |

### Multiple data roots

Configure additional data roots directly in `config.py`:

```python
DATA_ROOTS = {
    'Default': Path('/path/to/default/data'),
    'Lab A':   Path('/mnt/lab_a/measurements'),
    'Lab B':   Path('/mnt/lab_b/measurements'),
}
```

Each entry appears in the **Root** dropdown. When only one root is configured, it is auto-selected.

### Database

The SQLite database is created automatically on first run at the path specified by `SIGNAL_VIEWER_DB_PATH`. It uses WAL mode for concurrent read access. The database file and its WAL/SHM journals (`*.db3-wal`, `*.db3-shm`) are excluded from version control via `.gitignore`.

Tables:
- `favourites` — `(user, file_path, created_at)` with unique constraint on `(user, file_path)`
- `comments` — `(user, file_path, content, is_public, created_at, updated_at)`; private notes have unique constraint on `(user, file_path, is_public=0)`
- `lists` — `(user, name, created_at)` with unique constraint on `(user, name)`
- `list_files` — `(list_id, file_path, added_at)` with foreign key to `lists`


## Preparing Your Data

### Folder structure

```
data/                               ← one data root
├── SN001/                          ← serial number (prefix: SN)
│   ├── p001_motor_test/            ← folder_1 (pattern: pNNN_label)
│   │   ├── run_nominal/            ← folder_2 (any name)
│   │   │   └── data.h5
│   │   └── run_overload/
│   │       └── data.h5
│   └── p002_vibration/
│       └── baseline/
│           └── data.h5
├── SN002/
│   └── ...
```

- **Serial directories** must start with `SN` (configurable via `HDF5Schema.SERIAL_PREFIX`)
- **folder_1** must match `pNNN_label` (configurable via `HDF5Schema.FOLDER1_REGEX`)
- **folder_2** can be any name
- **Files** must have `.h5` extension (configurable via `HDF5Schema.FILE_EXTENSION`)

### HDF5 file structure

Each `.h5` file contains one or more **groups**. The mapping between logical dataset roles and actual HDF5 dataset names is defined per group in `config.py` via `GROUP_DS_NAMES`.

**GROUP_T0** (Type A — standard signals):

| Dataset | Example name | Shape | Required | Fallback |
|---------|-------------|-------|----------|----------|
| VALUE | `GROUP_T0_V` | (M, N) float64 | **Yes** | — |
| NAMES | `GROUP_T0_N` | (M,) string | Recommended | Signal_0, Signal_1, … |
| TIME | `GROUP_T0_TIM` | (M,) float64 | No | 0.0 (epoch ms) |
| SAMPLING_FREQ | `GROUP_T0_FRE` | (M,) float64 | No | Parsed from VALUE name suffix (`_0050` → 50 Hz), else 1.0 |
| NSAMPLE | `GROUP_T0_SAM` | (M,) int64 | No | Full N from VALUE shape |
| UNITS | `GROUP_T0_UNI` | (M,) string | No | Empty strings |

**GROUP_T1** (Type B — extended signals): all of the above, plus:

| Dataset | Example name | Shape | Description |
|---------|-------------|-------|-------------|
| ERROR | `GROUP_T1_ERR` | (M, N) float64 | Error values per signal |
| SQI | `GROUP_T1_SQI` | (M,) float64 | Signal Quality Index |
| TLS | `GROUP_T1_TLS` | (M,) float64 | Tolerance values |

Type A vs B is auto-detected by the presence of the ERROR dataset. Groups where VALUE is missing or empty are silently ignored.

**Time construction**: `t0 + arange(n) × (1000 / fs)` where `t0` from TIME (or 0.0), `n` from NSAMPLE (or full N), `fs` from SAMPLING_FREQ (or parsed from VALUE name suffix, or 1.0). All timestamps are epoch milliseconds.

### Creating a compatible file with h5py

```python
import h5py
import numpy as np

with h5py.File('data.h5', 'w') as f:
    grp = f.create_group('GROUP_T0')

    n_signals, n_samples = 4, 10000
    grp.create_dataset('GROUP_T0_V',   data=np.random.randn(n_signals, n_samples))
    grp.create_dataset('GROUP_T0_TIM', data=np.full(n_signals, 1700000000000.0))
    grp.create_dataset('GROUP_T0_FRE', data=np.array([100.0, 200.0, 100.0, 500.0]))
    grp.create_dataset('GROUP_T0_SAM', data=np.array([10000, 10000, 10000, 10000]))
    grp.create_dataset('GROUP_T0_N',   data=['position', 'velocity', 'current', 'voltage'])
    grp.create_dataset('GROUP_T0_UNI', data=['m', 'm/s', 'A', 'V'])
```

### Customizing the schema

Edit `GROUP_DS_NAMES` in `signal_viewer/config.py`:

```python
GROUP_DS_NAMES = {
    'MY_GROUP': {
        'VALUE':         'my_values',
        'TIME':          'my_timestamps',
        'SAMPLING_FREQ': 'my_fs',
        'NSAMPLE':       'my_nsamples',
        'NAMES':         'my_signal_names',
        'UNITS':         'my_units',
        # Add these for Type B:
        # 'ERROR': 'my_errors',
        # 'SQI':   'my_sqi',
        # 'TLS':   'my_tls',
    },
}
```


## API Reference

All endpoints return JSON. File paths in URL segments are base64url-encoded (RFC 4648 §5, no padding).

### Navigation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/roots` | List configured data root labels |
| GET | `/api/roots/{root}/serials` | List serials for a root |
| GET | `/api/roots/{root}/serials/{serial}/steps` | List steps for a serial |
| GET | `/api/roots/{root}/serials/{serial}/steps/{step}/files` | List files (path, filename, size, modified) |

### Signal Data

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/files/{path}/batches` | List batches (groups) in a file |
| GET | `/api/files/{path}/batches/{batch}/meta` | Batch metadata (signal names, units, count) |
| GET | `/api/files/{path}/batches/{batch}/signals/{idx}` | Signal data with optional `?downsample=N&t_min=...&t_max=...` |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/analysis/stats` | Descriptive statistics and rainflow cycle counting |
| POST | `/api/analysis/correlation` | Cross-correlation between two signals |
| POST | `/api/analysis/trend` | Polynomial trend fitting |

### Favourites

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/favourites` | All favourite file paths for the current user |
| GET | `/api/favourites/{path}` | Check if a file is favourited |
| POST | `/api/favourites/{path}` | Set/unset favourite `{"active": true}` |

### Notes (Comments)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/comments/{path}` | Get private and public notes for a file |
| POST | `/api/comments/{path}` | Save a note `{"content": "...", "is_public": true}` |
| DELETE | `/api/comments/{path}?id=N` | Delete a note by ID |

### Lists

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/lists` | All lists for the current user |
| POST | `/api/lists` | Create a list `{"name": "..."}` |
| DELETE | `/api/lists` | Delete a list `{"list_id": N}` |
| GET | `/api/lists/{id}/files` | File paths in a list |
| POST | `/api/lists/{id}/files` | Add a file `{"file_path": "..."}` |
| DELETE | `/api/lists/{id}/files?file=path` | Remove a file from a list |

### Filtering

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/file-tree?filter=favs\|list:N` | Pruned cascade tree containing only entries leading to matching files |

### Cache

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cache/stats` | Cache statistics (entry count, memory usage) |
| POST | `/api/rescan` | Rescan data directories and rebuild the metadata index |


## Running Tests

```bash
source venv/bin/activate
python3 -m unittest discover tests/ -v
```

The test suite includes 247 tests covering: HDF5 reader, metadata indexing, signal cache, resampling, statistics, rainflow, trend analysis, correlation, SQLite database, and full integration tests against the Tornado server. All tests run without requiring real HDF5 data — a built-in mock provides realistic sample signals.


## Performance Tips

- **Cache budget** — increase `SIGNAL_VIEWER_CACHE_MB` if you have spare RAM. The LRU cache keeps full signals in memory so repeated zoom queries skip disk I/O entirely.
- **Adaptive zoom** — resolution is handled automatically. Zooming in requests only the visible window at higher point density (up to 8,000 points per signal).
- **Large files** — signals are truncated to their declared `NSAMPLE` length, so unused buffer space in HDF5 arrays is never loaded.
- **Concurrent access** — the server uses thread-safe file handles and a thread-pool executor for analysis. The SQLite database uses WAL mode for concurrent reads.
- **WebGL rendering** — `scattergl` traces handle tens of thousands of points smoothly. The viewer caps display at 8,000 points per signal via downsampling.


## Troubleshooting

**No files appear in the cascade dropdowns**
Verify that your data directory matches the expected folder structure: `root/SNxxx/pNNN_label/subfolder/file.h5`. Serial directories must start with the configured `SERIAL_PREFIX` (default `SN`), and the first subfolder must match `FOLDER1_REGEX` (default `pNNN_label`). Check that `SIGNAL_VIEWER_DATA_ROOT` points to the correct path. Use `POST /api/rescan` to rebuild the metadata index after adding files.

**Signals load but the plot is empty or shows a flat line**
This typically means the signal's VALUE dataset contains all zeros or NaN. Check the HDF5 file with `h5py` directly. If only the tail of the signal appears flat, this was a known issue with MinMax-LTTB tail underrepresentation — upgrade to the latest version which applies uniform bucket density across the full signal length.

**Zoom echoes or unexpected zoom resets**
The viewer uses deterministic echo detection by comparing the relayout range against `_lastPlotRange`. If you experience zoom instability, ensure you are using the latest `viewer.html` which removes all timing-based guards in favour of range comparison. All built-in Plotly autorange triggers (`doubleClick`, `resetScale2d`, `autoScale2d`) are disabled so that only explicit user zoom/pan events reach the handler.

**Database errors or "database is locked"**
The SQLite database uses WAL mode for concurrent reads. If you see locking errors under heavy multi-user access, increase the busy timeout or consider placing the database file on a local (non-network) filesystem. The database path is configurable via `SIGNAL_VIEWER_DB_PATH`.

**Server fails to start with "Address already in use"**
Another process is using port 8050 (or your configured `SIGNAL_VIEWER_PORT`). Either stop the other process or set a different port via `SIGNAL_VIEWER_PORT=8060 python3 -m signal_viewer.server.app`.

**HDF5 groups not appearing as batches**
Groups are excluded when the VALUE dataset is missing or empty, or when the NAMES dataset is missing or contains blank entries. Check that both datasets exist and that NAMES has a non-empty label for every signal. Enable verbose logging with `SIGNAL_VIEWER_VERBOSE=true` to see detailed skip reasons on the console.

**Slow initial load for large data directories**
The metadata index scans all serial/step/file paths at startup. For very large trees, this can take several seconds. The index is rebuilt lazily — once built, navigation is instant. Use `POST /api/rescan` to refresh the index without restarting the server.


## Requirements

- Python 3.9+
- numpy, scipy, scikit-learn, tornado, h5py, Jinja2, matplotlib

See `requirements.txt` for exact versions. All dependencies are installed automatically by `create_venv.sh`.


## License

MIT License — see LICENSE file for details.
