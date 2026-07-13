import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitparse import FitFile

MIN_FILE_DURATION_S = 10 * 60  # file sotto questa durata esclusi dall'analisi
N_BINS = 50  # 50 fette del 2% = 100% dell'EFD totale del file

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
# .fit parsing (lat/lon/elevation/tempo — nessun dato cardiaco necessario qui)
# ---------------------------------------------------------------------------
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
        ts = data.get("timestamp")

        rows.append((lat, lon, ele, ts))

    df = pd.DataFrame(rows, columns=["lat", "lon", "ele", "timestamp"])
    df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)

    if df.empty:
        return df

    df["ele"] = df["ele"].astype(float).ffill().bfill().fillna(0.0)
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
# Model fitting: try several candidate shapes, pick the best by adjusted R²
# ---------------------------------------------------------------------------
def fit_candidates(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Fit several candidate forms of deviation_pct(x) where x = EFD progress (%).
    Returns dict of name -> {coefs, predict_fn, label_fn, r2, adj_r2, k}
    k = number of fitted parameters (used for adjusted R²).
    """
    n = len(x)
    ss_tot = np.sum((y - y.mean()) ** 2)

    def r2_of(pred):
        ss_res = np.sum((y - pred) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def adj_r2_of(r2, k):
        if n - k - 1 <= 0 or np.isnan(r2):
            return np.nan
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)

    candidates = {}

    # Linear: y = a + b*x
    c = np.polyfit(x, y, 1)
    pred = np.polyval(c, x)
    r2 = r2_of(pred)
    candidates["Lineare"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, xx),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot p",
    }

    # Quadratic: y = a + b*x + c*x^2
    c = np.polyfit(x, y, 2)
    pred = np.polyval(c, x)
    r2 = r2_of(pred)
    candidates["Quadratica"] = {
        "coefs": c, "k": 3, "r2": r2, "adj_r2": adj_r2_of(r2, 3),
        "predict": lambda xx, c=c: np.polyval(c, xx),
        "equation": rf"\Delta EFS(\%) = {c[2]:.3f} + {c[1]:.4f} \cdot p + {c[0]:.5f} \cdot p^2",
    }

    # Cubic: y = a + b*x + c*x^2 + d*x^3
    c = np.polyfit(x, y, 3)
    pred = np.polyval(c, x)
    r2 = r2_of(pred)
    candidates["Cubica"] = {
        "coefs": c, "k": 4, "r2": r2, "adj_r2": adj_r2_of(r2, 4),
        "predict": lambda xx, c=c: np.polyval(c, xx),
        "equation": (rf"\Delta EFS(\%) = {c[3]:.3f} + {c[2]:.4f} \cdot p + "
                     rf"{c[1]:.5f} \cdot p^2 + {c[0]:.6f} \cdot p^3"),
    }

    # Logaritmica: y = a + b*ln(x+1)
    x_log = np.log1p(x)
    c = np.polyfit(x_log, y, 1)
    pred = np.polyval(c, x_log)
    r2 = r2_of(pred)
    candidates["Logaritmica"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.log1p(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \ln(p+1)",
    }

    # Radice quadrata: y = a + b*sqrt(x)
    x_sqrt = np.sqrt(x)
    c = np.polyfit(x_sqrt, y, 1)
    pred = np.polyval(c, x_sqrt)
    r2 = r2_of(pred)
    candidates["Radice quadrata"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.sqrt(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \sqrt{{p}}",
    }

    return candidates


# ===========================================================
# Sidebar — upload & impostazioni
# ===========================================================
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

st.title("SPEED DECADENCE")
st.caption("EFD/EFS secondo il modello del costo energetico di Minetti et al. (2002).")

# ===========================================================
# Elaborazione file caricati
# ===========================================================
per_file_segments = {}   # filename -> segments dataframe
per_file_summary = {}    # filename -> summary dict

if uploaded_files:
    summary_rows = []
    for f in uploaded_files:
        raw = parse_fit(f)
        if raw.empty or len(raw) < 2:
            st.warning(f"⚠️ {f.name}: nessun dato GPS valido, file escluso.")
            continue

        segments, summary = process_track(raw, smooth_window, resample_step)
        if segments is None:
            st.warning(f"⚠️ {f.name}: distanza totale nulla, file escluso.")
            continue

        per_file_segments[f.name] = segments
        per_file_summary[f.name] = summary

        avg_efs_ms = summary["efd_m"] / summary["total_time_s"] if summary["total_time_s"] > 0 else np.nan
        summary_rows.append({
            "File": f.name,
            "Durata": seconds_to_hhmm(summary["total_time_s"]),
            "Distanza (km)": summary["horizontal_distance_m"] / 1000,
            "D+ (m)": summary["d_plus_m"],
            "D- (m)": summary["d_minus_m"],
            "EFD (km)": summary["efd_m"] / 1000,
            "EFS media (km/h)": avg_efs_ms * 3.6,
        })

    if summary_rows:
        st.subheader("Riepilogo file")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
else:
    st.info("Carica uno o più file .fit dalla sidebar per iniziare.")

# ===========================================================
# Decadimento EFS in funzione dell'EFD accumulato
# ===========================================================
st.divider()
st.header("📉 Decadimento EFS in funzione dell'EFD accumulato")
st.caption(
    "Per ciascun file viene calcolato l'EFD progressivo. Il file viene poi tagliato in "
    "fette del 2% dell'EFD totale (50 fette): per ciascuna fetta si calcola lo scostamento "
    "percentuale dell'EFS di fetta rispetto all'EFS medio dell'intero file, in funzione "
    "della percentuale di EFD già accumulata (non del tempo trascorso). I dati di tutti i "
    "file vengono combinati per stimare l'equazione che meglio descrive il decadimento, "
    "utilizzabile per altre gare/allenamenti."
)

decadence_rows = []

for fname, seg in per_file_segments.items():
    total_time_s = float(seg["dt_s"].sum())
    if total_time_s < MIN_FILE_DURATION_S:
        continue  # file troppo corto, escluso dall'analisi del decadimento

    total_efd_m = float(seg["efd_m"].sum())
    if total_time_s <= 0 or total_efd_m <= 0:
        continue

    avg_efs_ms = total_efd_m / total_time_s  # EFS medio dell'intero file

    dt_arr = seg["dt_s"].to_numpy()
    efd_arr = seg["efd_m"].to_numpy()

    # Progresso in % dell'EFD accumulato (punto medio di ogni segmento)
    cum_efd_before = np.cumsum(efd_arr) - efd_arr
    mid_efd = cum_efd_before + efd_arr / 2.0
    progress_pct = np.clip(mid_efd / total_efd_m * 100.0, 0.0, 100.0 - 1e-9)
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

    candidates = fit_candidates(x, y)
    best_name = max(candidates, key=lambda k: (candidates[k]["adj_r2"]
                                                if not np.isnan(candidates[k]["adj_r2"]) else -np.inf))

    st.subheader("Equazioni candidate (ordinate per adj. R²)")
    st.caption(
        "L'adjusted R² penalizza i modelli con più parametri, così un grado più alto vince "
        "solo se spiega davvero più varianza, non solo perché ha più gradi di libertà."
    )

    ranked = sorted(candidates.items(), key=lambda kv: (kv[1]["adj_r2"]
                                                          if not np.isnan(kv[1]["adj_r2"]) else -np.inf),
                     reverse=True)

    for name, c in ranked:
        star = " 🏆 **migliore**" if name == best_name else ""
        st.markdown(f"**{name}**{star}")
        st.latex(c["equation"])
        st.caption(f"R² = {c['r2']:.3f} · adj. R² = {c['adj_r2']:.3f}")

    # --- Grafico: curva di ogni file (sottile) + il fit migliore in evidenza ---
    fig, ax = plt.subplots(figsize=(9, 5))
    for fname, fdf in decadence_df.groupby("file"):
        fdf_sorted = fdf.sort_values("progress_pct")
        ax.plot(fdf_sorted["progress_pct"], fdf_sorted["deviation_pct"],
                alpha=0.3, linewidth=1)

    x_line = np.linspace(0, 100, 200)
    colors = {"Lineare": "tab:blue", "Quadratica": "tab:red", "Cubica": "tab:green",
              "Logaritmica": "tab:orange", "Radice quadrata": "tab:purple"}
    for name, c in candidates.items():
        lw = 3 if name == best_name else 1.2
        alpha = 1.0 if name == best_name else 0.5
        label = f"{name}{' (migliore)' if name == best_name else ''}"
        ax.plot(x_line, c["predict"](x_line), color=colors.get(name), linewidth=lw,
                alpha=alpha, label=label)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    ax.set_xlabel("Progresso EFD accumulato (%)")
    ax.set_ylabel("Scostamento EFS rispetto alla media del file (%)")
    ax.set_title("Decadimento EFS in funzione dell'EFD accumulato")
    ax.legend(fontsize=8)
    st.pyplot(fig)

    with st.expander("Dati aggregati per fetta (debug/export)"):
        st.dataframe(decadence_df, use_container_width=True, hide_index=True)
        st.download_button(
            label="📥 Scarica dati decadimento EFS (CSV)",
            data=decadence_df.to_csv(index=False).encode("utf-8"),
            file_name="decadimento_efs_dati.csv",
            mime="text/csv",
        )