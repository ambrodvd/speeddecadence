"""Nucleo condiviso: parsing .fit, pipeline EFD/EFS (Minetti 2002), helper di
formattazione tempi e definizione dei bucket di lunghezza gara.
Nessuna UI qui dentro, a parte la cache di Streamlit sul parsing."""
import gzip
import io

import numpy as np
import pandas as pd
import streamlit as st
from fitparse import FitFile

MIN_FILE_DURATION_S = 10 * 60  # file sotto questa durata esclusi dall'analisi


# ---------------------------------------------------------------------------
# Energy cost of running as a function of slope (Minetti et al. 2002)
# ---------------------------------------------------------------------------
def cost_of_running(slope: np.ndarray) -> np.ndarray:
    """
    Mass-specific energy cost of running, J/(kg*m), as a function of slope
    (decimal fraction). Vectorized. Clamped to +/-45%.
    """
    i = np.clip(slope, -0.45, 0.45)
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)


FLAT_COST = float(cost_of_running(np.array([0.0]))[0])  # 3.6 J/kg/m
SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seconds_to_hhmm(seconds) -> str:
    """Format a duration in seconds as HH:MM:SS (or MM:SS if under an hour)."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def hhmm_to_seconds(t):
    """Converte 'HH:MM:SS' o 'MM:SS' in secondi. Non parsabile -> NaN."""
    if pd.isna(t) or not isinstance(t, str) or not t.strip():
        return np.nan
    try:
        parts = [int(p) for p in t.strip().split(":")]
    except ValueError:
        return np.nan
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    else:
        return np.nan
    return h * 3600 + m * 60 + s


def _maybe_gunzip(upload):
    """Ritorna un file-like leggibile da FitFile, scompattando se gzip.
    Il riconoscimento è sui magic bytes (1f 8b), non sull'estensione:
    TrainingPeaks consegna file .fit.gz e altri export hanno estensioni
    sbagliate o assenti."""
    data = upload.getvalue()
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError) as e:
            raise ValueError(f"gzip non valido: {e}")
    return io.BytesIO(data)


# ---------------------------------------------------------------------------
# .fit parsing (lat/lon/elevation/tempo — nessun dato cardiaco necessario qui)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lettura file .fit...")
def load_raw(file_bytes: bytes) -> pd.DataFrame:
    """Solo parsing (niente EFD/EFS): basta per le analisi che usano FC e tempo."""
    return parse_fit(_maybe_gunzip_bytes(file_bytes))


@st.cache_data(show_spinner="Parsing e calcolo EFD/EFS...")
def load_and_process(file_bytes: bytes, smooth_window: int, resample_step_m: float):
    """Parsing + process in un'unica funzione cached. La chiave è
    (bytes, parametri): finché non cambiano, i rerun sono lookup istantanei
    invece di un re-parse completo di tutti i file."""
    raw = load_raw(file_bytes)
    if raw.empty or len(raw) < 2:
        return None, None, None
    segments, summary = process_track(raw, smooth_window, resample_step_m)
    return raw, segments, summary


def _maybe_gunzip_bytes(data: bytes):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return io.BytesIO(data)


def parse_fit(file_obj) -> pd.DataFrame:
    """Extract lat, lon, elevation, elapsed time (s) per record."""
    fitfile = FitFile(file_obj)
    rows = []
    for record in fitfile.get_messages("record"):
        data = {f.name: f.value for f in record}

        lat_raw = data.get("position_lat")
        lon_raw = data.get("position_long")
        if lat_raw is None or lon_raw is None:
            continue

        lat = lat_raw * SEMICIRCLE_TO_DEG
        lon = lon_raw * SEMICIRCLE_TO_DEG
        ele = data.get("enhanced_altitude", data.get("altitude"))
        hr = data.get("heart_rate")
        dist = data.get("distance")  # distanza del dispositivo (m), può mancare
        ts = data.get("timestamp")

        rows.append((lat, lon, ele, hr, dist, ts))

    df = pd.DataFrame(rows, columns=["lat", "lon", "ele", "hr", "dist_dev_m", "timestamp"])
    df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)

    if df.empty:
        return df

    df["ele"] = df["ele"].astype(float).ffill().bfill().fillna(0.0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    df["dist_dev_m"] = pd.to_numeric(df["dist_dev_m"], errors="coerce")
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized great-circle distance in meters between arrays of points."""
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Core pipeline: smooth -> resample onto uniform distance grid -> cost -> EFD/EFS
# ---------------------------------------------------------------------------
def process_track(df: pd.DataFrame, smooth_window: int, resample_step_m: float):
    lat, lon, ele = df["lat"].to_numpy(), df["lon"].to_numpy(), df["ele"].to_numpy()
    time_s = df["elapsed_s"].to_numpy()

    # Smooth elevation (rolling mean)
    ele_smooth = (pd.Series(ele)
                  .rolling(window=smooth_window, center=True, min_periods=1)
                  .mean()
                  .to_numpy())

    # Cumulative horizontal distance along the raw trace
    seg_dist = haversine_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dist)])
    total_dist = cum_dist[-1]

    if total_dist <= 0:
        return None, None

    # Resample onto a uniform horizontal-distance grid
    n_steps = max(2, int(total_dist // resample_step_m))
    grid = np.linspace(0.0, total_dist, n_steps)
    ele_grid = np.interp(grid, cum_dist, ele_smooth)
    time_grid = np.interp(grid, cum_dist, time_s)

    dx = np.diff(grid)
    dz = np.diff(ele_grid)
    dt = np.diff(time_grid)
    dist3d = np.hypot(dx, dz)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)

    cost = cost_of_running(slope)          # J/(kg*m)
    energy = cost * dist3d                 # J/kg per segment
    efd_m = energy / FLAT_COST             # equivalent flat meters per segment
    efs_ms = np.divide(efd_m, dt, out=np.full_like(efd_m, np.nan), where=dt > 0)

    d_plus = float(dz[dz > 0].sum())
    d_minus = float(-dz[dz < 0].sum())
    total_energy = float(energy.sum())
    total_efd_m = float(efd_m.sum())
    total_time_s = float(dt.sum())

    segments = pd.DataFrame({
        "distance_km": grid[1:] / 1000,
        "elevation_m": ele_grid[1:],
        "slope_pct": slope * 100,
        "dt_s": dt,
        "efd_m": efd_m,
        "efs_ms": efs_ms,
    })

    summary = {
        "horizontal_distance_m": total_dist,
        "d_plus_m": d_plus,
        "d_minus_m": d_minus,
        "total_energy_j_per_kg": total_energy,
        "efd_m": total_efd_m,
        "total_time_s": total_time_s,
    }
    return segments, summary


# ---------------------------------------------------------------------------
# Bucket per lunghezza gara (EFD totale, km)
# ---------------------------------------------------------------------------
def get_bucket_definitions(t1: float, t2: float, t3: float):
    """
    Ritorna la lista ordinata dei nomi bucket e un dict nome -> (lo, hi) in km.
    hi = None per il bucket aperto in alto.
    """
    order = [
        f"< {t1:g} km",
        f"{t1:g}-{t2:g} km",
        f"{t2:g}-{t3:g} km",
        f"> {t3:g} km",
    ]
    bounds = {
        order[0]: (0.0, t1),
        order[1]: (t1, t2),
        order[2]: (t2, t3),
        order[3]: (t3, None),
    }
    return order, bounds


def assign_bucket(efd_totale_km: float, bucket_order, bucket_bounds) -> str:
    for name in bucket_order:
        lo, hi = bucket_bounds[name]
        if hi is None:
            if efd_totale_km >= lo:
                return name
        elif lo <= efd_totale_km < hi:
            return name
    return bucket_order[-1]