import streamlit as st
import matplotlib.pyplot as plt
import io
import numpy as np
import pandas as pd
import streamlit as st
from fitparse import FitFile
import xml.etree.ElementTree as ET

MIN_FILE_DURATION_S = 10 * 60  # file sotto questa durata esclusi dal riepilogo finale

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


# ---------------------------------------------------------------------------
# .fit parsing
# ---------------------------------------------------------------------------
def parse_fit(file_obj) -> pd.DataFrame:
    """Extract lat, lon, elevation, heart rate, elapsed time (s) per record."""
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
        ts = data.get("timestamp")

        rows.append((lat, lon, ele, hr, ts))

    df = pd.DataFrame(rows, columns=["lat", "lon", "ele", "hr", "timestamp"])
    df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)

    if df.empty:
        return df

    df["ele"] = df["ele"].astype(float).ffill().bfill().fillna(0.0)
    df["hr"] = df["hr"].astype(float)  # may contain NaN, handled later
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


def hr_to_zone(hr, z1, z2, z3, z4, z5):
    if np.isnan(hr):
        return None
    if hr <= z1:
        return "Z1"
    elif hr <= z2:
        return "Z2"
    elif hr <= z3:
        return "Z3"
    elif hr <= z4:
        return "Z4"
    else:
        return "Z5"


# ---------------------------------------------------------------------------
# Core pipeline: smooth -> resample onto uniform distance grid -> cost -> EFS/zone
# ---------------------------------------------------------------------------
def process_track(df: pd.DataFrame, smooth_window: int, resample_step_m: float,
                   zones: tuple):
    lat, lon, ele = df["lat"].to_numpy(), df["lon"].to_numpy(), df["ele"].to_numpy()
    time_s = df["elapsed_s"].to_numpy()
    hr_raw = df["hr"].to_numpy()

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

    # HR: interpolate ignoring NaNs (fill gaps first so np.interp has valid data)
    hr_series = pd.Series(hr_raw).interpolate(limit_direction="both").to_numpy()
    hr_grid = np.interp(grid, cum_dist, hr_series)

    dx = np.diff(grid)
    dz = np.diff(ele_grid)
    dt = np.diff(time_grid)
    dist3d = np.hypot(dx, dz)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)

    cost = cost_of_running(slope)          # J/(kg*m)
    energy = cost * dist3d                 # J/kg per segment
    efd_m = energy / FLAT_COST             # equivalent flat meters per segment
    efs_ms = np.divide(efd_m, dt, out=np.full_like(efd_m, np.nan), where=dt > 0)

    hr_seg = (hr_grid[:-1] + hr_grid[1:]) / 2.0   # avg HR across the segment
    zone = np.array([hr_to_zone(h, *zones) for h in hr_seg])

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
        "hr_avg": hr_seg,
        "zone": zone,
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


def zone_efs_table(segments: pd.DataFrame) -> pd.DataFrame:
    """Average EFS per HR zone = total EFD in zone / total time in zone (time-weighted)."""
    valid = segments.dropna(subset=["zone"])
    rows = []
    for z in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
        zdf = valid[valid["zone"] == z]
        if zdf.empty:
            continue
        total_efd = zdf["efd_m"].sum()
        total_time = zdf["dt_s"].sum()
        if total_time <= 0:
            continue
        efs_ms = total_efd / total_time
        rows.append({
            "Zona": z,
            "Tempo (min)": total_time / 60,
            "EFD (km)": total_efd / 1000,
            "EFS media (km/h)": efs_ms * 3.6,
        })
    return pd.DataFrame(rows)

# --- File upload & processing settings ---
with st.sidebar:
    st.header("Impostazioni")
    uploaded_files = st.file_uploader(
        "Carica uno o più file .fit", type=["fit"], accept_multiple_files=True
    )
    smooth_window = st.slider("Finestra di smoothing quota (punti)", 1, 31, 9, step=2)
    resample_step = st.slider("Passo di ricampionamento (m)", 5, 100, 20, step=5)
    st.caption(
        "Lo smoothing ripulisce la quota grezza GPS/barometrica prima di calcolare la pendenza. "
        "Il passo di ricampionamento controlla la risoluzione orizzontale usata per integrare l'energia."
    )

per_file_zone_tables = []  # list of (filename, zone_table, total_time_s)
per_file_segments = {}     # filename -> segments dataframe (needed for Lap Analysis below)

for uploaded in uploaded_files:
    st.header(f"📄 {uploaded.name}")

    df_points = parse_fit(io.BytesIO(uploaded.getvalue()))
    if len(df_points) < 2:
        st.error("Traccia non valida: mancano punti GPS/timestamp validi in questo file.")
        continue
    if df_points["hr"].isna().all():
        st.warning("⚠️ Nessun dato di frequenza cardiaca trovato in questo file: impossibile assegnare le zone.")
        continue

    segments, summary = process_track(df_points, smooth_window, resample_step, zones)
    if segments is None:
        st.error("Impossibile calcolare la distanza percorsa (traccia degenere).")
        continue

    per_file_segments[uploaded.name] = segments

    c1, c2, c3 = st.columns(3)
    c1.metric("Distanza orizzontale", f"{summary['horizontal_distance_m']/1000:.2f} km")
    c2.metric("D+ / D−", f"{summary['d_plus_m']:.0f} m / {summary['d_minus_m']:.0f} m")
    c3.metric("EFD totale", f"{summary['efd_m']/1000:.2f} km")

    st.subheader("EFS media per zona cardiaca")
    zone_table = zone_efs_table(segments)
    if zone_table.empty:
        st.warning("Nessun segmento assegnabile a una zona (dati FC insufficienti).")
    else:
        display_table = zone_table.copy()
        display_table["Tempo (min)"] = display_table["Tempo (min)"].round(0).astype(int)
        st.dataframe(
            display_table.style.format({
                "EFD (km)": "{:.2f}",
                "EFS media (km/h)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            label="📥 Scarica tabella zone (CSV)",
            data=zone_table.to_csv(index=False).encode('utf-8'),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_zone_efs.csv",
            mime='text/csv',
            key=f"dl_{uploaded.name}",
        )
        per_file_zone_tables.append((uploaded.name, zone_table, summary["total_time_s"]))

    with st.expander("Dettaglio segmenti (debug)"):
        st.dataframe(segments, use_container_width=True)

    st.divider()
st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# ===========================================================
# UI — Decadimento EFS nel tempo
# ===========================================================
st.divider()
st.header("📉 Decadimento EFS nel tempo")
st.caption(
    "Analizza come l'EFS varia lungo il tempo trascorso di ciascun allenamento: "
    "ogni file viene tagliato in fette del 2% del tempo totale (50 fette), e per "
    "ciascuna fetta si calcola lo scostamento percentuale dell'EFS di fetta rispetto "
    "all'EFS medio dell'intero file. I dati di tutti i file vengono poi combinati per "
    "stimare un'equazione generale del decadimento, applicabile ad altre gare/allenamenti."
)

N_BINS = 50  # 50 fette del 2% = 100% del tempo totale

decadence_rows = []

for fname, seg in per_file_segments.items():
    total_time_s = float(seg["dt_s"].sum())
    if total_time_s < MIN_FILE_DURATION_S:
        continue  # stesso filtro usato per il riepilogo finale (>= 10 minuti)

    total_efd_m = float(seg["efd_m"].sum())
    if total_time_s <= 0 or total_efd_m <= 0:
        continue

    avg_efs_ms = total_efd_m / total_time_s  # EFS medio dell'intero file

    dt_arr = seg["dt_s"].to_numpy()
    efd_arr = seg["efd_m"].to_numpy()

    cum_time_s = np.cumsum(dt_arr)
    mid_time_s = cum_time_s - dt_arr / 2.0  # punto medio temporale di ogni segmento
    progress_pct = np.clip(mid_time_s / total_time_s * 100.0, 0.0, 100.0 - 1e-9)
    bin_idx = np.clip((progress_pct // 2).astype(int), 0, N_BINS - 1)

    for b in range(N_BINS):
        mask = bin_idx == b
        bin_dt = dt_arr[mask].sum()
        if bin_dt <= 0:
            continue
        bin_efd = efd_arr[mask].sum()
        bin_efs_ms = bin_efd / bin_dt  # EFS della fetta, time-weighted
        deviation_pct = (bin_efs_ms - avg_efs_ms) / avg_efs_ms * 100.0
        decadence_rows.append({
            "file": fname,
            "progress_pct": b * 2 + 1,   # centro della fetta: 1, 3, 5, ..., 99
            "efs_kmh": bin_efs_ms * 3.6,
            "deviation_pct": deviation_pct,
        })

if not decadence_rows:
    st.info(
        f"Nessun file idoneo per l'analisi del decadimento (serve almeno un file "
        f">= {MIN_FILE_DURATION_S // 60} minuti)."
    )
else:
    decadence_df = pd.DataFrame(decadence_rows)
    n_files_used = decadence_df["file"].nunique()
    st.caption(f"File inclusi nell'analisi: {n_files_used}")

    x = decadence_df["progress_pct"].to_numpy()
    y = decadence_df["deviation_pct"].to_numpy()

    # --- Fit lineare: y = b*x + a ---
    lin_coefs = np.polyfit(x, y, deg=1)
    lin_pred = np.polyval(lin_coefs, x)
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res_lin = np.sum((y - lin_pred) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else np.nan

    # --- Fit polinomiale grado 2: y = a2*x^2 + b2*x + c ---
    quad_coefs = np.polyfit(x, y, deg=2)
    quad_pred = np.polyval(quad_coefs, x)
    ss_res_quad = np.sum((y - quad_pred) ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot if ss_tot > 0 else np.nan

    st.subheader("Equazioni stimate")
    col_lin, col_quad = st.columns(2)

    with col_lin:
        b_lin, a_lin = lin_coefs[0], lin_coefs[1]
        st.markdown("**Lineare**")
        st.latex(rf"\Delta EFS(\%) = {a_lin:.3f} + {b_lin:.4f} \cdot progresso(\%)")
        st.caption(f"R² = {r2_lin:.3f}")

    with col_quad:
        a2_quad, b2_quad, c_quad = quad_coefs[0], quad_coefs[1], quad_coefs[2]
        st.markdown("**Polinomiale (grado 2)**")
        st.latex(
            rf"\Delta EFS(\%) = {c_quad:.3f} + {b2_quad:.4f} \cdot progresso(\%) "
            rf"+ {a2_quad:.5f} \cdot progresso(\%)^2"
        )
        st.caption(f"R² = {r2_quad:.3f}")

    st.caption(
        "ΔEFS(%) = scostamento percentuale dell'EFS della fetta rispetto alla media "
        "dell'intero file · progresso(%) = percentuale del tempo totale trascorso (0-100)."
    )

    # --- Grafico: curva di ogni file (sottile) + i due fit combinati ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for fname, fdf in decadence_df.groupby("file"):
        fdf_sorted = fdf.sort_values("progress_pct")
        ax.plot(fdf_sorted["progress_pct"], fdf_sorted["deviation_pct"],
                alpha=0.35, linewidth=1)

    x_line = np.linspace(0, 100, 200)
    ax.plot(x_line, np.polyval(lin_coefs, x_line),
            color="tab:blue", linewidth=2.5, label="Fit lineare")
    ax.plot(x_line, np.polyval(quad_coefs, x_line),
            color="tab:red", linewidth=2.5, label="Fit polinomiale (grado 2)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Progresso nel tempo trascorso (%)")
    ax.set_ylabel("Scostamento EFS rispetto alla media del file (%)")
    ax.set_title("Decadimento EFS lungo il tempo trascorso")
    ax.legend()
    st.pyplot(fig)

    with st.expander("Dati aggregati per fetta (debug/export)"):
        st.dataframe(decadence_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Scarica dati decadimento EFS (CSV)",
            data=decadence_df.to_csv(index=False).encode("utf-8"),
            file_name="decadimento_efs_dati.csv",
            mime="text/csv",
        )