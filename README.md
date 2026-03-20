# Engineering Signal Viewer & Analyzer

A web-based tool for exploring, visualizing, and analyzing time-series engineering data stored in HDF5 format.

```
┌─────────────────────────────────────────────────────────────┐
│                 SIGNAL VIEWER & ANALYZER                    │
├─────────────────────────────────────────────────────────────┤
│ Serials: [SN001] [SN002] [SN003]                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Signal Plot            │ Statistics  │ Trend Analysis      │
│  ┌──────────────────┐   │ Mean: 5.2   │ Degree 1: 5.1x+0.2 │
│  │ ╱╲╱╲  ╱╲╱╲       │   │ Std:  1.1   │ Fit quality: 0.94  │
│  │╱  ╲╱  ╲╱  ╲      │   │ Min:  2.3   │                    │
│  │                  │   │ Max:  8.1   │ Correlation         │
│  │ Trends│ Corr    │   │ Median: 5.0 │ Cross-correlation   │
│  │ Poly-1│ Cross   │   │             │ Max lag: 42 samples │
│  └──────────────────┘   │ RMS Trend   │ Analysis Tools      │
│                         │ Windows: 5  │ Rainflow cycles     │
│                                       │ Envelope estimation │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Data Explorer** — browse your HDF5 data hierarchy (Serial → Step → File) from the sidebar
- **Interactive Plots** — overlay or subplot layout modes with WebGL-accelerated rendering (Plotly scattergl)
- **Adaptive Zoom** — click-drag to zoom in; the viewer automatically re-fetches data at higher resolution so you get pixel-level detail at any depth
- **Statistics** — descriptive stats, rolling statistics, histograms, and rainflow cycle counting
- **Trend Analysis** — polynomial fitting, envelope estimation, RMS trend, changepoint detection
- **Correlation** — cross-correlation, coherence, and correlation matrices
- **Signal Comparison** — compare signals across different files or groups side-by-side
- **Signal Types** — supports Type A (standard) and Type B (extended with error, SQI, and TLS datasets), auto-detected per group
- **Performance** — in-memory LRU signal cache with configurable budget, two-pass MinMax-LTTB downsampling, binary-search windowed queries
- **REST API** — all data is served through CORS-enabled JSON endpoints, so you can integrate with external tools


## Quick Start

### 1. Clone or download

```bash
git clone https://github.com/yourusername/03_hdf5_signal_viewer.git
cd 03_hdf5_signal_viewer
```

### 2. Set up the Python environment

```bash
bash create_venv.sh
source venv/bin/activate
```

This creates a virtual environment and installs all dependencies.

### 3. Generate sample data (optional)

To try the app right away without your own data:

```bash
python3 generate_dummy_data.py
```

This creates three serial numbers with multiple processing steps and realistic engineering signals (vibration, temperature, pressure, motor current, strain, and more).

### 4. Start the server

```bash
bash run.sh
```

Or manually:

```bash
source venv/bin/activate
python3 -m signal_viewer.server.app
```

### 5. Open your browser

Navigate to **http://127.0.0.1:8050** and start exploring.


## Using the Viewer

### Browsing data

The sidebar lists your serial numbers. Expand a serial to see its processing steps, then select a file. The available signal groups appear as collapsible sections — click one to expand it and see the individual signals.

### Plotting signals

Click a signal name to add it to the plot area. You can overlay up to five signals on a single chart or switch to subplots mode for a stacked view. Each signal card shows the signal name, group, and unit.

### Zooming and navigation

Click and drag on the plot to zoom into a region. The viewer re-fetches only the visible window at higher resolution, so zooming in always reveals more detail. Use the **Reset Zoom** button to return to the full view.

### Analysis tools

The **Analysis** page provides statistical analysis, trend fitting, and correlation tools. Select a file and signal, then choose your analysis type. Results are displayed alongside interactive plots.

### Signal comparison

The **Comparison** page lets you load signals from different files or groups and view them side-by-side for cross-comparison.

### Documentation

The **Docs** page within the app provides an in-app reference for all features and API endpoints.


## Preparing Your Data

### Folder structure

Place your HDF5 files inside the `data/` directory following this layout:

```
data/
├── SN001/                        ← serial number
│   ├── p001_motor_test/          ← processing step (folder_1)
│   │   ├── run_nominal/          ← run folder (folder_2)
│   │   │   └── data.h5
│   │   └── run_overload/
│   │       └── data.h5
│   └── p002_vibration/
│       └── baseline/
│           └── data.h5
├── SN002/
│   └── ...
```

- **Serial directories** must start with `SN` (configurable)
- **folder_1** must match the pattern `pNNN_label` (e.g. `p001_motor_test`)
- **folder_2** can be any name (e.g. `run_nominal`, `warmup`, `cycle_1`)

To store data elsewhere, set the environment variable:

```bash
export SIGNAL_VIEWER_DATA_ROOT=/path/to/your/data
```

### HDF5 file structure

Each `.h5` file contains one or more **groups**. Each group holds a set of time-series signals with associated metadata. The mapping between logical dataset roles and actual HDF5 dataset names is defined in the configuration (`signal_viewer/config.py`) via `GROUP_DS_NAMES`.

The default configuration defines two groups:

**GROUP_T0** (Type A — standard signals):

| Dataset | Example name | Shape | Description |
|---------|-------------|-------|-------------|
| VALUE | `GROUP_T0_V` | (M, N) float64 | Signal values — M signals, N samples each |
| TIME | `GROUP_T0_TIM` | (M,) float64 | Start time per signal (epoch milliseconds) |
| SAMPLING_FREQ | `GROUP_T0_FRE` | (M,) float64 | Sampling frequency per signal (Hz) |
| NSAMPLE | `GROUP_T0_SAM` | (M,) int64 | Valid sample count per signal |
| NAMES | `GROUP_T0_N` | (M,) string | Signal names |
| UNITS | `GROUP_T0_UNI` | (M,) string | Signal units |

**GROUP_T1** (Type B — extended signals): all of the above, plus:

| Dataset | Example name | Shape | Description |
|---------|-------------|-------|-------------|
| ERROR | `GROUP_T1_ERR` | (M, N) float64 | Error values per signal |
| SQI | `GROUP_T1_SQI` | (M,) float64 | Signal Quality Index per signal |
| TLS | `GROUP_T1_TLS` | (M,) float64 | Tolerance values per signal |

The viewer auto-detects Type A vs Type B based on whether the ERROR dataset is present.

**Time construction** (both types): the time vector for signal `i` is computed as `TIME[i] + arange(NSAMPLE[i]) × (1000 / SAMPLING_FREQ[i])`, producing timestamps in epoch milliseconds.

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

To adapt the viewer to your own HDF5 naming convention, edit the `GROUP_DS_NAMES` dictionary in `signal_viewer/config.py`. Each entry maps a group name to a dictionary of logical keys → actual dataset names. You can add more groups, rename datasets, or mark a group as Type B by including an `ERROR` key:

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


## Configuration

Control the server via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNAL_VIEWER_DATA_ROOT` | `<project>/data/` | Root directory for HDF5 files |
| `SIGNAL_VIEWER_HOST` | `127.0.0.1` | Server bind address |
| `SIGNAL_VIEWER_PORT` | `8050` | Server port |
| `SIGNAL_VIEWER_DEBUG` | `true` | Enable debug mode with auto-reload |
| `SIGNAL_VIEWER_VERBOSE` | `false` | Log detailed error tracebacks |
| `SIGNAL_VIEWER_CACHE_MB` | `500` | Signal cache memory budget (MB) |


## Running Tests

```bash
source venv/bin/activate
python3 -m unittest discover tests/ -v
```

The test suite includes 196 tests covering the HDF5 reader, metadata indexing, signal cache, resampling, statistics, trend analysis, correlation, and full integration tests against the Tornado server. All tests run without requiring real HDF5 data — a built-in mock provides realistic sample signals.


## Performance Tips

- **Cache budget** — increase `SIGNAL_VIEWER_CACHE_MB` if you have spare RAM. The LRU cache keeps full signals in memory so repeated zoom queries skip disk I/O entirely.
- **Adaptive zoom** — the viewer handles resolution automatically. When you zoom in, it requests only the visible window at higher point density.
- **Large files** — signals are truncated to their declared `NSAMPLE` length, so unused buffer space in the HDF5 arrays is never loaded.
- **Concurrent access** — the server uses thread-safe file handles and a thread-pool executor for analysis, so multiple browser tabs work without blocking each other.


## Requirements

- Python 3.9+
- numpy, tornado, h5py, Jinja2, plotly

See `requirements.txt` for exact versions. All dependencies are installed automatically by `create_venv.sh`.


## License

MIT License — see LICENSE file for details.
