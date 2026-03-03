# Engineering Signal Viewer & Analyzer

A professional web-based tool for exploring, visualizing, and analyzing time-series engineering data stored in HDF5 format.

```
┌─────────────────────────────────────────────────────────────┐
│                 SIGNAL VIEWER & ANALYZER                    │
├─────────────────────────────────────────────────────────────┤
│ Serials: [SN001] [SN002] [SN003]                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Signal Plot            │ Statistics  │ FFT Analysis         │
│  ┌──────────────────┐   │ Mean: 5.2   │ Freq: 0-500 Hz       │
│  │ ╱╲╱╲  ╱╲╱╲       │   │ Std:  1.1   │ Peak: 120 Hz         │
│  │╱  ╲╱  ╲╱  ╲      │   │ Min:  2.3   │ Mag:  45 dB          │
│  │                  │   │ Max:  8.1   │                      │
│  │ Filters │ Trends │   │ Median: 5.0 │ Anomalies: 3 detected
│  │ Lowpass │ Poly-1 │   │             │                      │
│  └──────────────────┘   │ Correlation │ Analysis Tools       │
│                         │ Cross-corr  │ Welch PSD, Trends    │
│                                       │ IIR Filtering        │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Data Explorer**: Browse hierarchical HDF5 structure (Serial Number → Step → Files)
- **Signal Viewer**: Interactive time-series plots with zoom, pan, and downsampling
- **Spectral Analysis**: FFT and Power Spectral Density (Welch's method)
- **Filtering**: Butterworth lowpass, highpass, and bandpass filters
- **Anomaly Detection**: Multiple methods (Z-score, MAD, IQR, rolling window)
- **Statistical Insights**: Descriptive stats, trends, cross-correlation
- **Performance**: Efficient caching and LTTB downsampling for large datasets
- **Responsive UI**: Works on desktop and tablets
- **CORS-Enabled API**: RESTful endpoints for integration with external tools

## Quick Start

### 1. Clone or Download

```bash
git clone https://github.com/yourusername/03_hdf5_signal_viewer.git
cd 03_hdf5_signal_viewer
```

### 2. Create Python Environment

```bash
bash create_venv.sh
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Add Your Data

Generate realistic dummy data to explore the app immediately:

```bash
python generate_dummy_data.py
```

This creates 3 serial numbers, multiple processing steps, and batches with 16 different
engineering signals (vibration, temperature, pressure, motor current, strain, etc.) totalling
50k-150k samples each.

Or place your own HDF5 data inside `data/`:

```
03_hdf5_signal_viewer/
├── data/
│   ├── SN001/
│   │   ├── step_1/
│   │   │   └── SN001_2024-01-15_abc123.h5
│   │   ├── step_2/ ...
│   ├── SN002/ ...
```

Override the data location with an environment variable if your data lives elsewhere:

```bash
export SIGNAL_VIEWER_DATA_ROOT=/path/to/external/data
```

### 4. Start the Server

```bash
python3 -m signal_viewer.server.app
# Or use: bash run.sh
```

### 5. Open Browser

Visit: http://127.0.0.1:8050

## Data Format

### HDF5 File Structure

Each `.h5` file must contain one or more batches with the following structure:

```
file.h5
├── batch_001
│   ├── time (1D array, float64): Time coordinates [N]
│   ├── corrected_positions (1D array, float64): Corrected time values [N]
│   ├── value (2D array, float64): Signal data [M, N]
│   │   - M: number of signals
│   │   - N: number of samples per signal
│   ├── name (1D array, string): Signal names [M]
│   ├── units (1D array, string): Signal units [M]
│   └── other_attributes (optional): Additional metadata
├── batch_002
│   └── ...
```

### Example with h5py

```python
import h5py
import numpy as np

with h5py.File('data.h5', 'w') as f:
    grp = f.create_group('batch_001')

    # Create datasets
    t = np.linspace(0, 10, 1000)
    signals = np.random.randn(4, 1000)

    grp.create_dataset('time', data=t)
    grp.create_dataset('corrected_positions', data=t)
    grp.create_dataset('value', data=signals)
    grp.create_dataset('name',
        data=np.array(['position', 'velocity', 'current', 'voltage']))
    grp.create_dataset('units',
        data=np.array(['m', 'm/s', 'A', 'V']))
```

## Configuration

Control behavior via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNAL_VIEWER_DATA_ROOT` | `<project>/data/` | Root directory for HDF5 files |
| `SIGNAL_VIEWER_HOST` | `127.0.0.1` | Server bind address |
| `SIGNAL_VIEWER_PORT` | `8050` | Server port |
| `SIGNAL_VIEWER_DEBUG` | `true` | Enable debug mode (auto-reload) |
| `SIGNAL_VIEWER_CACHE_MB` | `500` | Signal cache budget in MB |

## Architecture

```
signal_viewer/
├── server/              # Tornado web server
│   ├── app.py          # Application setup & routing
│   └── handlers.py     # Request handlers (JSON API)
├── core/               # Data handling
│   ├── hdf5_reader.py  # HDF5 file I/O
│   ├── metadata_index.py  # Filesystem scanning
│   └── signal_cache.py    # In-memory caching
├── processing/         # Signal analysis algorithms
│   ├── spectral.py     # FFT, PSD
│   ├── filtering.py    # Butterworth filters
│   ├── anomaly.py      # Anomaly detection
│   ├── statistics.py   # Descriptive stats
│   ├── correlation.py  # Cross-correlation
│   ├── trend.py        # Polynomial fitting
│   └── resampling.py   # LTTB downsampling
├── templates/          # Jinja2 HTML templates
└── visualization/      # Client-side plotting (JavaScript)
```

## API Reference

### Metadata Endpoints

```
GET /api/serials                          # List serial numbers
GET /api/serials/{serial}/steps           # List steps for serial
GET /api/serials/{serial}/steps/{n}/files # List files in step
GET /api/files/{encoded_path}/batches     # List batches in file
GET /api/files/{encoded_path}/batches/{batch}/meta  # Batch metadata
```

### Signal Loading

```
GET /api/files/{encoded_path}/batches/{batch}/signals/{idx}?downsample=2000
```

Returns: `{time, values, name, units, samples}`

### Analysis Endpoints (POST)

```
POST /api/analysis/fft         # Fourier Transform
POST /api/analysis/psd         # Power Spectral Density
POST /api/analysis/filter      # Digital filtering
POST /api/analysis/anomaly     # Anomaly detection
POST /api/analysis/stats       # Signal statistics
POST /api/analysis/trend       # Polynomial trend fitting
POST /api/analysis/correlation # Cross-correlation
```

### Utility Endpoints

```
GET /api/cache/stats           # Cache performance metrics
POST /api/rescan               # Re-scan filesystem
```

## Development

### Running Tests

Run all unit and integration tests:

```bash
python3 -m unittest discover tests/ -v
```

Run specific test suite:

```bash
python3 -m unittest tests.test_integration -v
python3 -m unittest tests.test_hdf5_reader -v
```

### Test Coverage

- Unit tests for each processing module
- Integration tests for Tornado handlers
- Mock HDF5 file support for testing without h5py

### Project Structure for Testing

Tests expect data in standard format:
```
tests/
├── test_integration.py      # Server & API tests
├── test_hdf5_reader.py      # File I/O tests
├── test_metadata_index.py   # Filesystem scanning tests
├── test_signal_cache.py     # Caching tests
└── ...                      # Signal processing tests
```

## Requirements

- Python 3.8+
- tornado (async web server)
- numpy (numerical computing)
- scipy (signal processing)
- h5py (HDF5 file access, optional but recommended)
- jinja2 (HTML templating)

See `requirements.txt` for exact versions.

## Performance Tips

1. **Caching**: Configure `SIGNAL_VIEWER_CACHE_MB` based on available RAM
2. **Downsampling**: Use `?downsample=N` parameter for large signals (>100k samples)
3. **Batch Loading**: Process data in batches rather than loading entire files
4. **Filtering**: Apply filters server-side before visualization

## License

MIT License - See LICENSE file for details.

## Support

For issues, feature requests, or contributions, please visit the project repository.
