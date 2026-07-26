import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fitparse import FitFile

MIN_FILE_DURATION_S = 10 * 60  # file sotto questa durata esclusi dall'analisi

# Soglie minime di file per abilitare modelli con più parametri (evita overfitting)
MIN_FILES_FOR_QUADRATIC = 10
MIN_FILES_FOR_CUBIC = 15

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


def compute_bucket_centers(bucket_order, bucket_bounds, files_efd_km: dict):
    """
    Centro di ciascun bucket in km di EFD totale, usato per l'interpolazione.
    Bucket chiusi: punto medio. Bucket aperto in alto: mediana dei file
    realmente presenti in quel bucket (fallback: hi_precedente * 1.3).
    """
    centers = {}
    for name in bucket_order:
        lo, hi = bucket_bounds[name]
        if hi is not None:
            centers[name] = (lo + hi) / 2.0
        else:
            vals = [v for fname, (b, v) in files_efd_km.items() if b == name]
            if vals:
                centers[name] = float(np.median(vals))
            else:
                centers[name] = lo * 1.3
    return centers


# ---------------------------------------------------------------------------
# Model fitting: try several candidate shapes, pick the best by adjusted R²
# ---------------------------------------------------------------------------
def fit_candidates(x: np.ndarray, y: np.ndarray, n_files: int) -> dict:
    """
    Fit several candidate forms of deviation_pct(x) where x = EFD% accumulata
    nel file (0-100), relativa all'EFD totale del file stesso.

    n_files = numero di file indipendenti a supporto (non righe/bin): usato
    per limitare i gradi di libertà del modello quando il campione è piccolo,
    dato che i bin di uno stesso file NON sono osservazioni indipendenti.
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

    # Linear: y = a + b*x  (k=2, sempre incluso)
    c = np.polyfit(x, y, 1)
    pred = np.polyval(c, x)
    r2 = r2_of(pred)
    candidates["Lineare"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, xx),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot p",
    }

    # Logaritmica: y = a + b*ln(x+1)  (k=2, sempre incluso)
    x_log = np.log1p(x)
    c = np.polyfit(x_log, y, 1)
    pred = np.polyval(c, x_log)
    r2 = r2_of(pred)
    candidates["Logaritmica"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.log1p(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \ln(p+1)",
    }

    # Radice quadrata: y = a + b*sqrt(x)  (k=2, sempre incluso)
    x_sqrt = np.sqrt(x)
    c = np.polyfit(x_sqrt, y, 1)
    pred = np.polyval(c, x_sqrt)
    r2 = r2_of(pred)
    candidates["Radice quadrata"] = {
        "coefs": c, "k": 2, "r2": r2, "adj_r2": adj_r2_of(r2, 2),
        "predict": lambda xx, c=c: np.polyval(c, np.sqrt(xx)),
        "equation": rf"\Delta EFS(\%) = {c[1]:.3f} + {c[0]:.4f} \cdot \sqrt{{p}}",
    }

    # Quadratica: y = a + b*x + c*x^2  (k=3, richiede abbastanza file)
    if n_files >= MIN_FILES_FOR_QUADRATIC:
        c = np.polyfit(x, y, 2)
        pred = np.polyval(c, x)
        r2 = r2_of(pred)
        candidates["Quadratica"] = {
            "coefs": c, "k": 3, "r2": r2, "adj_r2": adj_r2_of(r2, 3),
            "predict": lambda xx, c=c: np.polyval(c, xx),
            "equation": rf"\Delta EFS(\%) = {c[2]:.3f} + {c[1]:.4f} \cdot p + {c[0]:.5f} \cdot p^2",
        }

    # Cubic: y = a + b*x + c*x^2 + d*x^3  (k=4, richiede ancora più file)
    if n_files >= MIN_FILES_FOR_CUBIC:
        c = np.polyfit(x, y, 3)
        pred = np.polyval(c, x)
        r2 = r2_of(pred)
        candidates["Cubica"] = {
            "coefs": c, "k": 4, "r2": r2, "adj_r2": adj_r2_of(r2, 4),
            "predict": lambda xx, c=c: np.polyval(c, xx),
            "equation": (rf"\Delta EFS(\%) = {c[3]:.3f} + {c[2]:.4f} \cdot p + "
                         rf"{c[1]:.5f} \cdot p^2 + {c[0]:.6f} \cdot p^3"),
        }

    return candidates


def best_candidate(candidates: dict):
    return max(candidates, key=lambda k: (candidates[k]["adj_r2"]
                                           if not np.isnan(candidates[k]["adj_r2"]) else -np.inf))


# ---------------------------------------------------------------------------
# Predizione "sfumata": interpola tra le curve dei due bucket più vicini
# per centro, cosi' non ci sono salti bruschi ai bordi dei bucket.
# ---------------------------------------------------------------------------
def predict_blended(efd_pct: float, efd_totale_gara_km: float,
                     bucket_fits: dict, bucket_centers: dict, bucket_order: list) -> float:
    """
    bucket_fits: nome_bucket -> candidato migliore (dict con "predict")
    bucket_centers: nome_bucket -> centro in km
    Ritorna la deviazione % prevista, interpolando linearmente in base alla
    distanza tra efd_totale_gara_km e i centri dei bucket disponibili.
    """
    available = [(name, bucket_centers[name]) for name in bucket_order if name in bucket_fits]
    if not available:
        return float("nan")
    available.sort(key=lambda t: t[1])

    if efd_totale_gara_km <= available[0][1]:
        name = available[0][0]
        return float(bucket_fits[name]["predict"](np.array([efd_pct]))[0])
    if efd_totale_gara_km >= available[-1][1]:
        name = available[-1][0]
        return float(bucket_fits[name]["predict"](np.array([efd_pct]))[0])

    for (name_lo, c_lo), (name_hi, c_hi) in zip(available[:-1], available[1:]):
        if c_lo <= efd_totale_gara_km <= c_hi:
            w_hi = (efd_totale_gara_km - c_lo) / (c_hi - c_lo)
            w_lo = 1.0 - w_hi
            pred_lo = bucket_fits[name_lo]["predict"](np.array([efd_pct]))[0]
            pred_hi = bucket_fits[name_hi]["predict"](np.array([efd_pct]))[0]
            return float(w_lo * pred_lo + w_hi * pred_hi)

    return float("nan")


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

    st.divider()
    st.subheader("Bucket per lunghezza gara (EFD totale)")
    st.caption(
        "Ogni file viene assegnato a un bucket in base al suo EFD totale (km). "
        "Il decadimento viene fittato separatamente per bucket, poi le curve vengono "
        "interpolate in base alla lunghezza della gara target, per evitare salti bruschi ai bordi."
    )
    thr1 = st.number_input("Soglia 1 (km EFD)", min_value=1.0, value=40.0, step=5.0)
    thr2 = st.number_input("Soglia 2 (km EFD)", min_value=thr1 + 1.0, value=max(60.0, thr1 + 1.0), step=5.0)
    thr3 = st.number_input("Soglia 3 (km EFD)", min_value=thr2 + 1.0, value=max(100.0, thr2 + 1.0), step=5.0)

    bin_width_pct = st.slider("Ampiezza fetta (% dell'EFD totale del file)", 1.0, 20.0, 5.0, step=1.0)
    st.caption(
        "Il decadimento viene analizzato per fette di EFD **relativo** (% del totale del file), "
        "non assoluto: cosi' un file corto e uno lungo restano confrontabili sulla stessa posizione "
        "relativa di gara. La lunghezza assoluta del file determina invece il bucket."
    )

st.title("SPEED DECADENCE")
st.caption("EFD/EFS secondo il modello del costo energetico di Minetti et al. (2002).")

bucket_order, bucket_bounds = get_bucket_definitions(thr1, thr2, thr3)

# ===========================================================
# Elaborazione file caricati
# ===========================================================
per_file_segments = {}   # filename -> segments dataframe
per_file_summary = {}    # filename -> summary dict
per_file_bucket = {}     # filename -> (bucket_name, efd_totale_km)

if uploaded_files:
    summary_rows = []
    for f in uploaded_files:
        try:
            raw = parse_fit(f)
        except Exception as e:
            st.warning(f"⚠️ {f.name}: file .fit illeggibile o corrotto, escluso ({e.__class__.__name__}).")
            continue

        if raw.empty or len(raw) < 2:
            st.warning(f"⚠️ {f.name}: nessun dato GPS valido, file escluso.")
            continue

        try:
            segments, summary = process_track(raw, smooth_window, resample_step)
        except Exception as e:
            st.warning(f"⚠️ {f.name}: errore durante l'elaborazione, escluso ({e.__class__.__name__}).")
            continue

        if segments is None:
            st.warning(f"⚠️ {f.name}: distanza totale nulla, file escluso.")
            continue

        per_file_segments[f.name] = segments
        per_file_summary[f.name] = summary

        efd_totale_km = summary["efd_m"] / 1000
        bucket_name = assign_bucket(efd_totale_km, bucket_order, bucket_bounds)
        per_file_bucket[f.name] = (bucket_name, efd_totale_km)

        avg_efs_ms = summary["efd_m"] / summary["total_time_s"] if summary["total_time_s"] > 0 else np.nan
        summary_rows.append({
            "File": f.name,
            "Durata": seconds_to_hhmm(summary["total_time_s"]),
            "Distanza (km)": summary["horizontal_distance_m"] / 1000,
            "D+ (m)": summary["d_plus_m"],
            "D- (m)": summary["d_minus_m"],
            "EFD (km)": efd_totale_km,
            "EFS media (km/h)": avg_efs_ms * 3.6,
            "Bucket": bucket_name,
        })

    if summary_rows:
        st.subheader("Riepilogo file")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
else:
    st.info("Carica uno o più file .fit dalla sidebar per iniziare.")

# ===========================================================
# Pannello diagnostico: quanti file per bucket
# ===========================================================
if per_file_bucket:
    st.divider()
    st.subheader("📊 Distribuzione file per bucket")

    diag_rows = []
    for name in bucket_order:
        files_in_bucket = [f for f, (b, _) in per_file_bucket.items() if b == name]
        eligible = [
            f for f in files_in_bucket
            if per_file_segments[f]["dt_s"].sum() >= MIN_FILE_DURATION_S
        ]
        if len(eligible) == 0:
            note = "❌ nessun file idoneo"
        elif len(eligible) < MIN_FILES_FOR_QUADRATIC:
            note = "⚠️ solo modelli lineari (pochi file)"
        elif len(eligible) < MIN_FILES_FOR_CUBIC:
            note = "🟡 fino a quadratica"
        else:
            note = "✅ tutti i modelli disponibili"
        diag_rows.append({
            "Bucket": name,
            "File totali": len(files_in_bucket),
            "File idonei (>=10 min)": len(eligible),
            "Modelli disponibili": note,
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Un bucket con meno di {MIN_FILES_FOR_QUADRATIC} file idonei usa solo modelli a 2 parametri "
        f"(lineare/log/radice); servono almeno {MIN_FILES_FOR_CUBIC} file per la cubica. "
        "Questo perché i bin di uno stesso file non sono osservazioni indipendenti: "
        "il vero campione statistico è il numero di file, non il numero di righe aggregate."
    )

# ===========================================================
# Decadimento EFS in funzione dell'EFD% accumulato, per bucket
# ===========================================================
st.divider()
st.header("📉 Decadimento EFS per bucket di lunghezza gara")
st.caption(
    f"Per ciascun file viene calcolato l'EFD progressivo in **% del totale del file** e tagliato in fette "
    f"di {bin_width_pct:g}%: per ciascuna fetta si calcola lo scostamento percentuale dell'EFS di fetta "
    "rispetto all'EFS medio dell'intero file. I file vengono raggruppati per bucket di EFD totale "
    "(lunghezza gara) e fittati separatamente, cosi' gare corte e lunghe non vengono mediate insieme."
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
    bucket_name, efd_totale_km = per_file_bucket[fname]

    dt_arr = seg["dt_s"].to_numpy()
    efd_arr = seg["efd_m"].to_numpy()

    # EFD accumulato relativo (% del totale del file), punto medio di ogni segmento
    cum_efd_before = np.cumsum(efd_arr) - efd_arr
    mid_efd_m = cum_efd_before + efd_arr / 2.0
    efd_pct_arr = mid_efd_m / total_efd_m * 100.0
    bin_idx = (efd_pct_arr // bin_width_pct).astype(int)

    bin_df = pd.DataFrame({"bin_idx": bin_idx, "dt_s": dt_arr, "efd_m": efd_arr})
    grouped = bin_df.groupby("bin_idx", as_index=False).sum()
    grouped = grouped[grouped["dt_s"] > 0]

    for _, row in grouped.iterrows():
        bin_efs_ms = row["efd_m"] / row["dt_s"]  # EFS della fetta, time-weighted
        deviation_pct = (bin_efs_ms - avg_efs_ms) / avg_efs_ms * 100.0
        efd_pct_center = (row["bin_idx"] + 0.5) * bin_width_pct
        decadence_rows.append({
            "file": fname,
            "bucket": bucket_name,
            "efd_totale_km": efd_totale_km,
            "efd_pct_accum": efd_pct_center,
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

    bucket_fits = {}       # bucket_name -> miglior candidato
    bucket_all_candidates = {}  # bucket_name -> tutti i candidati (per expander)
    bucket_n_files = {}

    for name in bucket_order:
        bdf = decadence_df[decadence_df["bucket"] == name]
        n_files = bdf["file"].nunique()
        bucket_n_files[name] = n_files
        if n_files == 0:
            continue
        x = bdf["efd_pct_accum"].to_numpy()
        y = bdf["deviation_pct"].to_numpy()
        candidates = fit_candidates(x, y, n_files)
        if not candidates:
            continue
        bucket_all_candidates[name] = candidates
        bucket_fits[name] = candidates[best_candidate(candidates)]

    files_efd_map = {f: per_file_bucket[f] for f in per_file_bucket}
    bucket_centers = compute_bucket_centers(bucket_order, bucket_bounds, files_efd_map)

    st.subheader("Equazioni migliori per bucket")
    for name in bucket_order:
        if name not in bucket_all_candidates:
            st.markdown(f"**{name}** — nessun file idoneo, bucket escluso dal fit.")
            continue
        st.markdown(f"**{name}** — {bucket_n_files[name]} file, centro bucket = {bucket_centers[name]:.1f} km EFD")
        ranked = sorted(
            bucket_all_candidates[name].items(),
            key=lambda kv: (kv[1]["adj_r2"] if not np.isnan(kv[1]["adj_r2"]) else -np.inf),
            reverse=True,
        )
        best_name = best_candidate(bucket_all_candidates[name])
        with st.expander(f"Vedi equazioni candidate — {name}"):
            for cname, c in ranked:
                star = " 🏆 **migliore**" if cname == best_name else ""
                st.markdown(f"{cname}{star}")
                st.latex(c["equation"])
                st.caption(f"R² = {c['r2']:.3f} · adj. R² = {c['adj_r2']:.3f}")

    # --- Grafico: curve per bucket (colore diverso) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {bucket_order[0]: "tab:blue", bucket_order[1]: "tab:orange",
              bucket_order[2]: "tab:green", bucket_order[3]: "tab:red"}

    for name in bucket_order:
        bdf = decadence_df[decadence_df["bucket"] == name]
        if bdf.empty:
            continue
        for fname, fdf in bdf.groupby("file"):
            fdf_sorted = fdf.sort_values("efd_pct_accum")
            ax.plot(fdf_sorted["efd_pct_accum"], fdf_sorted["deviation_pct"],
                    color=colors[name], alpha=0.15, linewidth=1)

    x_line = np.linspace(0, 100, 200)
    for name in bucket_order:
        if name not in bucket_fits:
            continue
        pred = bucket_fits[name]["predict"](x_line)
        ax.plot(x_line, pred, color=colors[name], linewidth=2.5,
                label=f"{name} (n={bucket_n_files[name]})")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("EFD accumulato (% del totale del file)")
    ax.set_ylabel("Scostamento EFS rispetto alla media del file (%)")
    ax.set_title("Decadimento EFS per bucket di lunghezza gara")
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

    # =======================================================
    # Predizione sfumata (blended) per una gara target
    # =======================================================
    st.divider()
    st.subheader("🎯 Prova la predizione sfumata (blended)")
    st.caption(
        "Inserisci l'EFD totale stimato della gara target e una posizione (% di gara): "
        "la previsione interpola tra i bucket vicini, cosi' una gara da 99 km e una da "
        "101 km danno risultati quasi identici invece di un salto netto tra bucket."
    )

    col1, col2 = st.columns(2)
    with col1:
        target_efd_km = st.number_input("EFD totale gara target (km)", min_value=1.0, value=80.0, step=5.0)
    with col2:
        target_pct = st.slider("Posizione in gara (% EFD accumulato)", 0, 100, 50)

    pred_deviation = predict_blended(target_pct, target_efd_km, bucket_fits, bucket_centers, bucket_order)
    if np.isnan(pred_deviation):
        st.warning("Nessun bucket disponibile per la predizione (carica più file idonei).")
    else:
        st.metric(
            f"Scostamento EFS previsto a {target_pct}% di gara",
            f"{pred_deviation:+.1f}%",
        )

    # Grafico di verifica continuità: curva blended per il target scelto
    # sovrapposta alle curve discrete dei bucket
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    for name in bucket_order:
        if name not in bucket_fits:
            continue
        pred = bucket_fits[name]["predict"](x_line)
        ax2.plot(x_line, pred, color=colors[name], linewidth=1.2, alpha=0.4,
                 linestyle="--", label=f"{name} (discreto)")

    blended_line = np.array([
        predict_blended(p, target_efd_km, bucket_fits, bucket_centers, bucket_order)
        for p in x_line
    ])
    ax2.plot(x_line, blended_line, color="black", linewidth=2.5,
             label=f"Blended per {target_efd_km:g} km EFD")
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("EFD accumulato (% di gara)")
    ax2.set_ylabel("Scostamento EFS previsto (%)")
    ax2.set_title("Verifica continuità: curva blended vs curve discrete per bucket")
    ax2.legend(fontsize=7)
    st.pyplot(fig2)