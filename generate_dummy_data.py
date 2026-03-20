#!/usr/bin/env python3
"""
Generate realistic dummy HDF5 data for testing the Signal Viewer.

Creates a folder structure matching the expected layout:
    data/
    └── SN001/
        ├── p001_motor_test/
        │   ├── run_nominal/
        │   │   └── p001_motor_test_2024-06-16_a3f1c2.h5
        │   ├── run_overload/
        │   │   └── ...
        │   └── warmup/
        │       └── ...
        ├── p002_vibration_study/
        │   ├── baseline/
        │   │   └── ...
        │   └── loaded/
        │       └── ...
        └── p003_endurance/
            └── cycle_1/
                └── ...

Each HDF5 file contains groups from HDF5Schema.GROUP_DS_NAMES.
  GROUP_T0 = Type A  (no _ERR/_SQI/_TLS)
  GROUP_T1 = Type B  (has _ERR/_SQI/_TLS)

Both types share these datasets (per group):
  GROUP_NAME_V    float64 [n_signals, max(n_samples)]  signal values
  GROUP_NAME_TIM  float64 [n_signals]  epoch time (ms) of first sample
  GROUP_NAME_FRE  float64 [n_signals]  sampling frequency (Hz)
  GROUP_NAME_SAM  int64   [n_signals]  valid sample count per signal
  GROUP_NAME_N    str     [n_signals]  signal names
  GROUP_NAME_UNI  str     [n_signals]  units

Type B additionally has:
  GROUP_NAME_ERR  float64 [n_signals, max(n_samples)]  signal errors
  GROUP_NAME_SQI  float64 [n_signals]  signal quality metric
  GROUP_NAME_TLS  float64 [n_signals]  max error gap (seconds)

Dummy data spec:
  Type A: 200 signals, max 1,500,000 samples, frequencies 1-100 Hz, same starting epoch
  Type B: 200 signals, 1,500,000 samples each, frequency=100 Hz, same starting epoch
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

# Allow running standalone or as part of the package
try:
    from signal_viewer.config import HDF5Schema as S
except ImportError:
    # Fallback: add parent dir to path so we can import the config
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from signal_viewer.config import HDF5Schema as S

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required to generate dummy data.")
    print("Install it with:  pip install h5py")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _vibration(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vibration sensor: sum of harmonics + broadband noise + occasional resonance."""
    f_base = rng.uniform(25, 60)
    sig = (
        1.0 * np.sin(2 * np.pi * f_base * t)
        + 0.4 * np.sin(2 * np.pi * 2 * f_base * t + rng.uniform(0, np.pi))
        + 0.15 * np.sin(2 * np.pi * 3 * f_base * t)
        + 0.08 * rng.standard_normal(len(t))
    )
    # Add transient burst
    burst_center = rng.uniform(t[len(t) // 4], t[3 * len(t) // 4])
    burst = 2.5 * np.exp(-((t - burst_center) ** 2) / 0.002) * np.sin(2 * np.pi * 400 * t)
    return sig + burst


def _temperature(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Thermocouple: slow ramp + overshoot + settling + sensor noise."""
    setpoint = rng.uniform(60, 200)
    tau = rng.uniform(0.5, 2.0)
    overshoot = rng.uniform(1.02, 1.12)
    response = setpoint * (1 - np.exp(-t / tau))
    # Add overshoot via damped oscillation
    omega = 2 * np.pi / (4 * tau)
    response *= 1.0 + (overshoot - 1.0) * np.exp(-t / (3 * tau)) * np.sin(omega * t)
    response += 0.3 * rng.standard_normal(len(t))
    return response


def _pressure(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pressure transducer: step changes + pulsation + drift."""
    base = rng.uniform(1.0, 5.0)
    sig = np.full_like(t, base)
    # Add 2-4 step changes
    n_steps = rng.integers(2, 5)
    for _ in range(n_steps):
        t_step = rng.uniform(t[0] + 0.5, t[-1] - 0.5)
        delta = rng.uniform(-1.5, 1.5)
        sig += delta / (1 + np.exp(np.clip(-50 * (t - t_step), -500, 500)))
    # Pump pulsation
    f_pump = rng.uniform(8, 15)
    sig += 0.12 * np.sin(2 * np.pi * f_pump * t)
    # Slow drift
    sig += 0.05 * t
    sig += 0.02 * rng.standard_normal(len(t))
    return sig


def _motor_current(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Motor current: inrush spike + steady state + load variations."""
    steady = rng.uniform(3.0, 12.0)
    # Inrush transient
    inrush = 6 * steady * np.exp(-t / 0.05)
    # Steady-state with load ripple
    f_line = 50.0 if rng.random() > 0.5 else 60.0
    ripple = 0.15 * steady * np.sin(2 * np.pi * f_line * t)
    # Load variation
    load_var = 0.3 * steady * np.sin(2 * np.pi * 0.2 * t)
    sig = inrush + steady + ripple + load_var
    sig += 0.05 * steady * rng.standard_normal(len(t))
    return sig


def _position_encoder(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Linear encoder: S-curve motion + quantisation noise."""
    stroke = rng.uniform(50, 300)  # mm
    period = t[-1] if t[-1] > 0 else 1.0
    pos = stroke * (3 * (t / period) ** 2 - 2 * (t / period) ** 3)
    # Quantisation noise (encoder resolution)
    resolution = 0.001  # mm
    pos = np.round(pos / resolution) * resolution
    return pos


def _voltage_rail(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """DC power rail: nominal voltage + ripple + sag events."""
    nominal = rng.choice([3.3, 5.0, 12.0, 24.0, 48.0])
    sig = np.full_like(t, nominal)
    # Switching ripple
    f_sw = rng.uniform(50e3, 200e3)
    sig += 0.01 * nominal * np.sin(2 * np.pi * (f_sw / 1000) * t)
    # Voltage sag
    n_sags = rng.integers(0, 3)
    for _ in range(n_sags):
        t_sag = rng.uniform(t[0], t[-1])
        depth = rng.uniform(0.02, 0.10) * nominal
        width = rng.uniform(0.01, 0.05)
        sig -= depth * np.exp(-((t - t_sag) ** 2) / (2 * width ** 2))
    sig += 0.002 * nominal * rng.standard_normal(len(t))
    return sig


def _flow_rate(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flow meter: pulsating flow + valve transitions."""
    base_flow = rng.uniform(5, 50)  # L/min
    sig = np.full_like(t, base_flow)
    # Valve open/close transitions
    n_valves = rng.integers(1, 4)
    for _ in range(n_valves):
        t_valve = rng.uniform(t[0] + 0.3, t[-1] - 0.3)
        direction = rng.choice([-1, 1])
        change = rng.uniform(0.2, 0.6) * base_flow
        sig += direction * change / (1 + np.exp(np.clip(-30 * (t - t_valve), -500, 500)))
    # Pulsation
    sig += 0.05 * base_flow * np.sin(2 * np.pi * rng.uniform(3, 8) * t)
    sig += 0.01 * base_flow * rng.standard_normal(len(t))
    sig = np.maximum(sig, 0)  # Flow can't be negative
    return sig


def _strain_gauge(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Strain gauge: cyclic loading + creep + fatigue cracks."""
    amplitude = rng.uniform(100, 500)  # microstrain
    f_load = rng.uniform(0.5, 5)
    sig = amplitude * np.sin(2 * np.pi * f_load * t)
    # Creep (slow increase in baseline)
    sig += 10 * np.sqrt(t + 0.01)
    # Simulated fatigue crack: sudden small jump late in signal
    if rng.random() > 0.5:
        crack_t = rng.uniform(0.7 * t[-1], 0.95 * t[-1])
        sig += 30 / (1 + np.exp(np.clip(-100 * (t - crack_t), -500, 500)))
    sig += 2 * rng.standard_normal(len(t))
    return sig


def _accelerometer(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Accelerometer: multi-frequency vibration + impact events."""
    sig = np.zeros_like(t)
    # 3-5 frequency components
    for _ in range(rng.integers(3, 6)):
        f = rng.uniform(10, 500)
        a = rng.uniform(0.1, 2.0)
        sig += a * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    # Impact events
    n_impacts = rng.integers(0, 4)
    for _ in range(n_impacts):
        t_impact = rng.uniform(t[0], t[-1])
        sig += rng.uniform(5, 15) * np.exp(-np.abs(t - t_impact) / 0.003) * np.cos(2 * np.pi * 800 * (t - t_impact))
    sig += 0.1 * rng.standard_normal(len(t))
    return sig


# ---------------------------------------------------------------------------
# Signal catalogue (expanded to 200 unique signal names)
# ---------------------------------------------------------------------------

_BASE_CATALOGUE = [
    ("vibration_x",      "m/s^2",   _vibration),
    ("vibration_y",      "m/s^2",   _vibration),
    ("vibration_z",      "m/s^2",   _vibration),
    ("temperature_1",    "degC",    _temperature),
    ("temperature_2",    "degC",    _temperature),
    ("pressure_inlet",   "bar",     _pressure),
    ("pressure_outlet",  "bar",     _pressure),
    ("motor_current",    "A",       _motor_current),
    ("position_x",       "mm",      _position_encoder),
    ("position_y",       "mm",      _position_encoder),
    ("voltage_24V",      "V",       _voltage_rail),
    ("voltage_5V",       "V",       _voltage_rail),
    ("flow_coolant",     "L/min",   _flow_rate),
    ("strain_1",         "ustrain", _strain_gauge),
    ("strain_2",         "ustrain", _strain_gauge),
    ("accel_spindle",    "g",       _accelerometer),
]


def _build_signal_catalogue(n_signals: int):
    """Build a catalogue of n_signals unique signal entries by repeating base patterns."""
    catalogue = []
    base_len = len(_BASE_CATALOGUE)
    for i in range(n_signals):
        name_base, unit, gen_fn = _BASE_CATALOGUE[i % base_len]
        suffix = f"_{i // base_len}" if i >= base_len else ""
        catalogue.append((f"{name_base}{suffix}", unit, gen_fn))
    return catalogue


# ---------------------------------------------------------------------------
# File/folder generation
# ---------------------------------------------------------------------------

SYSTEMS = [
    {
        "serial": "SN001",
        "folder1": "p001_motor_test",
        "folder2s": ["run_nominal", "run_overload", "warmup"],
    },
    {
        "serial": "SN002",
        "folder1": "p002_vibration_study",
        "folder2s": ["baseline", "loaded"],
    },
    {
        "serial": "SN003",
        "folder1": "p003_endurance",
        "folder2s": ["cycle_1"],
    },
]

# Dummy data parameters
NUM_SIGNALS = 200
MAX_SAMPLES = 1_500_000
EPOCH_MS = 1700000000000.0  # Common starting epoch in ms


def _make_crc(folder1: str, folder2: str) -> str:
    """Deterministic 6-char hex hash for filename."""
    raw = f"{folder1}-{folder2}".encode()
    return hashlib.md5(raw).hexdigest()[:6]


def generate_type_a_group(
    grp: h5py.Group,
    group_name: str,
    rng: np.random.Generator,
    num_signals: int = NUM_SIGNALS,
    max_samples: int = MAX_SAMPLES,
) -> None:
    """
    Populate one HDF5 group as Type A.

    Type A: 200 signals, frequencies 1-100 Hz, variable n_samples per signal.
    """
    catalogue = _build_signal_catalogue(num_signals)

    # Per-signal frequencies (1-100 Hz)
    freqs = rng.uniform(1.0, 100.0, size=num_signals).astype(np.float64)

    # Per-signal n_samples: random between 50% and 100% of max_samples
    n_samples = rng.integers(max_samples // 2, max_samples + 1, size=num_signals).astype(np.int64)

    # All signals share same starting epoch
    epoch_times = np.full(num_signals, EPOCH_MS, dtype=np.float64)

    names = []
    units = []
    values = np.zeros((num_signals, max_samples), dtype=np.float64)

    for i in range(num_signals):
        sig_name, sig_unit, gen_fn = catalogue[i]
        names.append(sig_name)
        units.append(sig_unit)

        n = int(n_samples[i])
        duration = n / freqs[i]
        t = np.linspace(0, duration, n, dtype=np.float64)
        values[i, :n] = gen_fn(t, rng)
        # Remaining columns stay 0 (padding)

    ds = S.GROUP_DS_NAMES[group_name]
    grp.create_dataset(ds['VALUE'], data=values, compression="gzip")
    grp.create_dataset(ds['TIME'], data=epoch_times)
    grp.create_dataset(ds['SAMPLING_FREQ'], data=freqs)
    grp.create_dataset(ds['NSAMPLE'], data=n_samples)
    grp.create_dataset(ds['NAMES'], data=np.array(names, dtype="S64"))
    grp.create_dataset(ds['UNITS'], data=np.array(units, dtype="S32"))


def generate_type_b_group(
    grp: h5py.Group,
    group_name: str,
    rng: np.random.Generator,
    num_signals: int = NUM_SIGNALS,
    max_samples: int = MAX_SAMPLES,
) -> None:
    """
    Populate one HDF5 group as Type B.

    Type B: 200 signals, frequency=100 Hz, all 1,500,000 samples each.
    Additional datasets: _ERR, _SQI, _TLS.
    """
    catalogue = _build_signal_catalogue(num_signals)

    freq = 100.0
    freqs = np.full(num_signals, freq, dtype=np.float64)
    n_samples = np.full(num_signals, max_samples, dtype=np.int64)
    epoch_times = np.full(num_signals, EPOCH_MS, dtype=np.float64)

    duration = max_samples / freq
    t = np.linspace(0, duration, max_samples, dtype=np.float64)

    names = []
    units = []
    values = np.zeros((num_signals, max_samples), dtype=np.float64)
    errors = np.zeros((num_signals, max_samples), dtype=np.float64)

    for i in range(num_signals):
        sig_name, sig_unit, gen_fn = catalogue[i]
        names.append(sig_name)
        units.append(sig_unit)
        values[i, :] = gen_fn(t, rng)
        # Error: small noise proportional to signal amplitude
        errors[i, :] = rng.standard_normal(max_samples) * 0.01 * np.std(values[i, :])

    sqi = rng.uniform(0.85, 1.0, size=num_signals).astype(np.float64)
    tls = rng.uniform(0.0, 5.0, size=num_signals).astype(np.float64)

    # Base datasets (same as Type A)
    ds = S.GROUP_DS_NAMES[group_name]
    grp.create_dataset(ds['VALUE'], data=values, compression="gzip")
    grp.create_dataset(ds['TIME'], data=epoch_times)
    grp.create_dataset(ds['SAMPLING_FREQ'], data=freqs)
    grp.create_dataset(ds['NSAMPLE'], data=n_samples)
    grp.create_dataset(ds['NAMES'], data=np.array(names, dtype="S64"))
    grp.create_dataset(ds['UNITS'], data=np.array(units, dtype="S32"))

    # Type B additional datasets
    grp.create_dataset(ds['ERROR'], data=errors, compression="gzip")
    grp.create_dataset(ds['SQI'], data=sqi)
    grp.create_dataset(ds['TLS'], data=tls)


def generate_all(data_root: Path, seed: int = 42) -> None:
    """Generate the full dummy dataset."""
    rng = np.random.default_rng(seed)
    data_root.mkdir(parents=True, exist_ok=True)
    total_files = 0

    for system in SYSTEMS:
        serial = system["serial"]
        folder1 = system["folder1"]
        print(f"  {serial}/{folder1}:")

        for f2_idx, folder2 in enumerate(system["folder2s"]):
            folder2_dir = data_root / serial / folder1 / folder2
            folder2_dir.mkdir(parents=True, exist_ok=True)

            crc = _make_crc(folder1, folder2)
            filename = f"{folder1}_2024-06-{16 + f2_idx:02d}_{crc}{S.FILE_SUFFIX}"
            filepath = folder2_dir / filename

            print(f"    {folder2}/{filename}  (generating...)", end="", flush=True)

            with h5py.File(filepath, "w") as f:
                for group_name in S.group_names():
                    grp = f.create_group(group_name)
                    if S.has_ds(group_name, 'ERROR'):
                        generate_type_b_group(grp, group_name, rng)
                    else:
                        generate_type_a_group(grp, group_name, rng)

            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"\r    {folder2}/{filename}  "
                  f"({len(S.group_names())} groups, "
                  f"{NUM_SIGNALS} signals, {MAX_SAMPLES} max_samples, "
                  f"{size_mb:.1f} MB)")
            total_files += 1

    print(f"\nGenerated {total_files} HDF5 files in {data_root}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    print("=== Generating Dummy HDF5 Data ===\n")
    generate_all(data_dir)
    print("\nDone. Start the viewer with:  bash run.sh")
