#!/usr/bin/env python3
"""
Generate realistic dummy HDF5 data for testing the Signal Viewer.

Creates a folder structure matching the expected layout:
    data/
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

Each HDF5 file contains groups from HDF5Schema.GROUP_NAMES.
Each group has datasets named GROUP_NAME + suffix:
  - GROUP_NAME_V:  float64[num_signals, num_samples]  (values)
  - GROUP_NAME_T:  float64[num_samples]                (time)
  - GROUP_NAME_P:  float64[num_samples]                (positions)
  - GROUP_NAME_N:  str[num_signals]                    (signal names)
  - GROUP_NAME_U:  str[num_signals]                    (units)

Signals are realistic engineering waveforms: vibration, temperature,
pressure, motor current, position encoder, voltage rail, flow rate, etc.
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
    """Linear encoder: trapezoidal motion profile + quantisation noise."""
    stroke = rng.uniform(50, 300)  # mm
    period = t[-1]
    # Trapezoidal: accel - cruise - decel
    accel_frac = 0.2
    cruise_frac = 0.6
    decel_frac = 0.2
    pos = np.zeros_like(t)
    t_norm = t / period
    for i, tn in enumerate(t_norm):
        if tn < accel_frac:
            pos[i] = 0.5 * stroke * (tn / accel_frac) ** 2 / (0.5 * accel_frac + cruise_frac + 0.5 * decel_frac) * (0.5 * accel_frac + cruise_frac + 0.5 * decel_frac)
        elif tn < accel_frac + cruise_frac:
            pos[i] = stroke * (tn - 0.5 * accel_frac) / (0.5 * accel_frac + cruise_frac + 0.5 * decel_frac)
        else:
            rem = 1.0 - tn
            pos[i] = stroke * (1.0 - 0.5 * (rem / decel_frac) ** 2 / (0.5 * accel_frac + cruise_frac + 0.5 * decel_frac) * (0.5 * accel_frac + cruise_frac + 0.5 * decel_frac))
    # Simplify: just use a smooth S-curve
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
    # Downsample ripple frequency to be visible
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
# Signal catalogue
# ---------------------------------------------------------------------------

SIGNAL_CATALOGUE = [
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


def _make_crc(folder1: str, folder2: str) -> str:
    """Deterministic 6-char hex hash for filename."""
    raw = f"{folder1}-{folder2}".encode()
    return hashlib.md5(raw).hexdigest()[:6]


def generate_group(
    grp: h5py.Group,
    group_name: str,
    rng: np.random.Generator,
    num_signals: int,
    num_samples: int,
    duration_sec: float,
) -> None:
    """Populate one HDF5 group with realistic signal data.

    Dataset names are built as group_name + suffix.
    """
    t = np.linspace(0, duration_sec, num_samples, dtype=np.float64)
    corrected = t + rng.uniform(-0.0001, 0.0001, size=num_samples)

    # Pick a random subset of signals from the catalogue
    indices = rng.choice(len(SIGNAL_CATALOGUE), size=min(num_signals, len(SIGNAL_CATALOGUE)), replace=False)
    indices.sort()

    names = []
    units = []
    values = np.empty((len(indices), num_samples), dtype=np.float64)

    for row, cat_idx in enumerate(indices):
        sig_name, sig_unit, gen_fn = SIGNAL_CATALOGUE[cat_idx]
        values[row, :] = gen_fn(t, rng)
        names.append(sig_name)
        units.append(sig_unit)

    grp.create_dataset(S.ds(group_name, S.TIME_SUFFIX), data=t)
    grp.create_dataset(S.ds(group_name, S.POSITION_SUFFIX), data=corrected)
    grp.create_dataset(S.ds(group_name, S.VALUE_SUFFIX), data=values)
    grp.create_dataset(S.ds(group_name, S.NAMES_SUFFIX), data=np.array(names, dtype="S64"))
    grp.create_dataset(S.ds(group_name, S.UNITS_SUFFIX), data=np.array(units, dtype="S32"))


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

            # Each folder_2 adds more signals
            base_signals = 8
            signals_this = min(base_signals + f2_idx * 4, len(SIGNAL_CATALOGUE))
            samples = rng.integers(50_000, 150_001)
            duration = rng.uniform(5.0, 30.0)

            with h5py.File(filepath, "w") as f:
                for group_name in S.GROUP_NAMES:
                    grp = f.create_group(group_name)
                    generate_group(grp, group_name, rng, signals_this, int(samples), duration)

            size_mb = filepath.stat().st_size / (1024 * 1024)
            n_groups = len(S.GROUP_NAMES)
            print(f"    {folder2}/{filename}  ({n_groups} groups, "
                  f"{signals_this} signals, {samples} samples, {size_mb:.1f} MB)")
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
