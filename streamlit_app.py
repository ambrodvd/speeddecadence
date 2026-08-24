import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
        hr = data.get("heart_rate")
        ts = data.get("timestamp")

        rows.append((lat, lon, ele, hr, ts))

    df = pd.DataFrame(rows, columns=["lat", "lon", "ele", "hr", "timestamp"])
    df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)

    if df.empty:
        return df

    df["ele"] = df["ele"].astype(float).ffill().bfill().fillna(0.0)
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
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


# ---------------------------------------------------------------------------
# Equazione di decadimento EFS EFFETTIVAMENTE USATA per generare i race plan
# ---------------------------------------------------------------------------
# ATTENZIONE: questa è una copia 1:1 dei coefficienti hardcoded nel trail
# predictor (streamlit_app.py, BUCKET_DELTA_EFS_FN / predict_delta_efs_blended).
# NON va confusa con bucket_fits/predict_blended qui sopra: quelli si
# ri-fittano LIVE sui file di training caricati in questa sessione e possono
# differire (più file, soglie diverse) da quelli usati quando un dato race
# plan è stato effettivamente generato. Per confrontare "previsto vs reale"
# nella sezione di confronto più sotto serve l'equazione congelata al
# momento della generazione del piano, quindi si usa QUESTA.
# Se in futuro ricalibri e aggiorni i coefficienti nel trail predictor,
# aggiornali anche qui di conseguenza.
def _delta_efs_40_60(p):
    return 23.860 - 0.5984 * p + 0.00263 * p**2


def _delta_efs_60_100(p):
    return 36.833 - 0.7490 * p - 0.00277 * p**2 + 0.000057 * p**3


def _delta_efs_100_plus(p):
    return 41.599 - 1.3125 * p + 0.01342 * p**2 - 0.000064 * p**3


_FROZEN_BUCKET_DELTA_EFS_FN = {
    "40-60 km": _delta_efs_40_60,
    "60-100 km": _delta_efs_60_100,
    ">100 km": _delta_efs_100_plus,
}

_FROZEN_BUCKET_CENTERS_KM = {
    "40-60 km": 50.0,
    "60-100 km": 80.0,
    ">100 km": 130.0,
}

_FROZEN_BUCKET_ORDER_BY_CENTER = sorted(
    _FROZEN_BUCKET_CENTERS_KM, key=lambda k: _FROZEN_BUCKET_CENTERS_KM[k]
)


def predict_delta_efs_frozen(efd_pct: float, race_total_efd_km: float) -> float:
    """Stessa logica di predict_delta_efs_blended nel trail predictor, ma con
    i coefficienti congelati sopra (l'equazione che genera davvero i race plan)."""
    centers = [(name, _FROZEN_BUCKET_CENTERS_KM[name]) for name in _FROZEN_BUCKET_ORDER_BY_CENTER]

    if race_total_efd_km <= centers[0][1]:
        return float(_FROZEN_BUCKET_DELTA_EFS_FN[centers[0][0]](efd_pct))
    if race_total_efd_km >= centers[-1][1]:
        return float(_FROZEN_BUCKET_DELTA_EFS_FN[centers[-1][0]](efd_pct))

    for (name_lo, c_lo), (name_hi, c_hi) in zip(centers[:-1], centers[1:]):
        if c_lo <= race_total_efd_km <= c_hi:
            w_hi = (race_total_efd_km - c_lo) / (c_hi - c_lo)
            w_lo = 1.0 - w_hi
            pred_lo = _FROZEN_BUCKET_DELTA_EFS_FN[name_lo](efd_pct)
            pred_hi = _FROZEN_BUCKET_DELTA_EFS_FN[name_hi](efd_pct)
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
bucket_fits, bucket_centers = {}, {}  # inizializzati qui, sovrascritti più sotto se ci sono dati di training

# ===========================================================
# Elaborazione file caricati
# ===========================================================
per_file_segments = {}   # filename -> segments dataframe
per_file_summary = {}    # filename -> summary dict
per_file_bucket = {}     # filename -> (bucket_name, efd_totale_km)
per_file_raw = {}        # filename -> raw record dataframe (serve per la FC)

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
        per_file_raw[f.name] = raw
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

# ===========================================================
# 📐 STUDIO PENDENZE: regressione lineare FC e EFS sul tempo
# ===========================================================
# Pendenze PURE, senza costanti di scala e senza normalizzazioni:
#   FC  -> y = bpm,  x = tempo
#   EFS -> y = km/h, x = tempo
# Si riportano entrambe le unità di x (secondi e ore): è la stessa
# retta, cambia solo il fattore 3600. Il DET index dell'analyzer usa
# la pendenza al secondo, quindi la colonna /s è quella confrontabile.
st.divider()
st.header("📐 Studio pendenze: FC e EFS vs tempo")

if not per_file_segments:
    st.info("Carica dei file .fit per eseguire lo studio.")
else:
    dist_metric = st.radio(
        "Su quale distanza raggruppare i file?",
        ["EFD totale (km)", "Distanza orizzontale (km)"],
        horizontal=True, key="slope_study_metric",
    )

    def _dist_group(km):
        if km < 50:
            return "< 50 km"
        if km <= 100:
            return "50-100 km"
        return "> 100 km"

    slope_rows = []
    for fname, seg in per_file_segments.items():
        summ = per_file_summary[fname]
        raw = per_file_raw.get(fname)

        if summ["total_time_s"] < MIN_FILE_DURATION_S:
            continue

        km_ref = (summ["efd_m"] / 1000 if dist_metric.startswith("EFD")
                  else summ["horizontal_distance_m"] / 1000)

        # --- FC: regressione sui record grezzi ---
        hr_slope_s = hr_intercept = np.nan
        if raw is not None and "hr" in raw.columns and raw["hr"].notna().sum() > 10:
            hr_ok = raw.dropna(subset=["hr", "elapsed_s"])
            hr_coef = np.polyfit(hr_ok["elapsed_s"].to_numpy(),
                                 hr_ok["hr"].to_numpy(), 1)
            hr_slope_s, hr_intercept = float(hr_coef[0]), float(hr_coef[1])

        # --- EFS: regressione sui segmenti, x = tempo al centro segmento ---
        # Si scartano i segmenti con velocità implausibile (soste, glitch):
        # un ristoro di 20 minuti è un singolo segmento a ~0 km/h e da solo
        # sposterebbe la retta.
        efs_slope_s = efs_intercept = np.nan
        dt = seg["dt_s"].to_numpy()
        cum_t = np.cumsum(dt)
        mid_t = cum_t - dt / 2.0
        efs_kmh = seg["efs_ms"].to_numpy() * 3.6
        ok = np.isfinite(efs_kmh) & (efs_kmh >= 0.5) & (efs_kmh <= 30.0)
        if ok.sum() > 10:
            efs_coef = np.polyfit(mid_t[ok], efs_kmh[ok], 1)
            efs_slope_s, efs_intercept = float(efs_coef[0]), float(efs_coef[1])

        slope_rows.append({
            "File": fname,
            "Gruppo": _dist_group(km_ref),
            "Distanza rif. (km)": km_ref,
            "Durata (h)": summ["total_time_s"] / 3600,
            "FC slope (bpm/s)": hr_slope_s,
            "FC slope (bpm/h)": hr_slope_s * 3600,
            "FC intercetta (bpm)": hr_intercept,
            "EFS slope (km/h per s)": efs_slope_s,
            "EFS slope (km/h per h)": efs_slope_s * 3600,
            "EFS intercetta (km/h)": efs_intercept,
        })

    if not slope_rows:
        st.info(f"Nessun file idoneo (serve almeno {MIN_FILE_DURATION_S // 60} minuti).")
    else:
        slope_df = pd.DataFrame(slope_rows)

        st.subheader("Equazioni per file")
        for _, r in slope_df.iterrows():
            st.markdown(f"**{r['File']}** — {r['Gruppo']} · "
                        f"{r['Distanza rif. (km)']:.1f} km · {r['Durata (h)']:.2f} h")
            if np.isfinite(r["FC slope (bpm/s)"]):
                st.latex(
                    rf"FC(t) = {r['FC intercetta (bpm)']:.2f} "
                    rf"{r['FC slope (bpm/s)']:+.6f} \cdot t_{{[s]}}"
                    rf"\qquad ({r['FC slope (bpm/h)']:+.3f}\ bpm/h)"
                )
            else:
                st.caption("— nessun dato di frequenza cardiaca in questo file")
            if np.isfinite(r["EFS slope (km/h per s)"]):
                st.latex(
                    rf"EFS(t) = {r['EFS intercetta (km/h)']:.3f} "
                    rf"{r['EFS slope (km/h per s)']:+.7f} \cdot t_{{[s]}}"
                    rf"\qquad ({r['EFS slope (km/h per h)']:+.4f}\ km/h\ per\ h)"
                )
            else:
                st.caption("— dati EFS insufficienti in questo file")

        st.subheader("Tabella pendenze")
        st.dataframe(slope_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Scarica pendenze (CSV)",
            data=slope_df.to_csv(index=False).encode("utf-8"),
            file_name="studio_pendenze_fc_efs.csv",
            mime="text/csv",
        )

        st.subheader("Medie per gruppo")
        _cols = ["FC slope (bpm/s)", "FC slope (bpm/h)",
                 "EFS slope (km/h per s)", "EFS slope (km/h per h)"]

        agg_rows = []
        for label, sub in [("TUTTI I FILE", slope_df)] + [
            (g, slope_df[slope_df["Gruppo"] == g])
            for g in ["< 50 km", "50-100 km", "> 100 km"]
        ]:
            if sub.empty:
                continue
            row = {"Gruppo": label,
                   "N file": len(sub),
                   "N con FC": int(sub["FC slope (bpm/s)"].notna().sum())}
            for c in _cols:
                row[c] = sub[c].mean(skipna=True)
            agg_rows.append(row)

        st.dataframe(pd.DataFrame(agg_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Media aritmetica delle pendenze dei singoli file, non una regressione "
            "sui dati aggregati: ogni file pesa uguale a prescindere dalla durata. "
            "Le colonne /s e /h sono la stessa pendenza (fattore 3600)."
        )

# ===========================================================
# UI — Confronto Race Plan previsto vs Performance reale
# ===========================================================
st.divider()
st.header("🆚 Confronto Race Plan previsto vs Gara reale")
st.caption(
    "Carica il race plan generato dal Trail Predictor (CSV) e il file .fit della gara "
    "realmente corsa: confronta i tempi di passaggio previsti vs reali ai lap, e lo "
    "scostamento EFS realmente osservato in gara rispetto alla curva teorica che ha "
    "generato il piano."
)

col_up1, col_up2 = st.columns(2)
with col_up1:
    plan_csv = st.file_uploader(
        "📋 Race plan previsto (CSV esportato dal Trail Predictor)",
        type=["csv"], key="compare_plan_csv",
    )
with col_up2:
    real_fit = st.file_uploader(
        "🏁 File .fit della gara realmente corsa",
        type=["fit"], key="compare_real_fit",
    )

if plan_csv is not None and real_fit is not None:
    try:
        plan_df_cmp = pd.read_csv(plan_csv)
    except Exception as e:
        st.error(f"⚠️ Errore nella lettura del CSV: {e}")
        plan_df_cmp = None

    if plan_df_cmp is not None:
        # Individua dinamicamente le colonne "Tempo di gara (...)": il nome
        # del profilo tra parentesi varia (Personalizzato, Competitivo, Profilo A...).
        tempo_gara_cols = [c for c in plan_df_cmp.columns if c.startswith("Tempo di gara")]
        required_base_cols = {"Lap", "Km partenza", "Km arrivo", "Distanza (km)"}
        missing = required_base_cols - set(plan_df_cmp.columns)

        if missing or not tempo_gara_cols:
            st.error(
                "⚠️ Il CSV non sembra un race plan valido. "
                + (f"Colonne mancanti: {missing}. " if missing else "")
                + ("Nessuna colonna 'Tempo di gara (...)' trovata." if not tempo_gara_cols else "")
            )
        else:
            if len(tempo_gara_cols) > 1:
                tempo_gara_col = st.selectbox(
                    "Più profili trovati nel CSV: quale usare come riferimento?",
                    tempo_gara_cols, key="tempo_gara_col_select",
                )
            else:
                tempo_gara_col = tempo_gara_cols[0]

            # --- elabora il file .fit reale ---
            try:
                raw_real = parse_fit(real_fit)
            except Exception as e:
                st.error(f"⚠️ File .fit illeggibile o corrotto: {e}")
                raw_real = pd.DataFrame()

            if raw_real.empty or len(raw_real) < 2:
                st.error("⚠️ Nessun dato GPS valido nel file .fit della gara reale.")
            else:
                segments_real, summary_real = process_track(raw_real, smooth_window, resample_step)
                if segments_real is None:
                    st.error("⚠️ Traccia reale degenere: distanza orizzontale nulla.")
                else:
                    # Griglie cumulate reali: km percorsi, tempo (s), EFD (km)
                    cum_km_real = segments_real["distance_km"].to_numpy()
                    cum_t_real = np.cumsum(segments_real["dt_s"].to_numpy())
                    total_efd_real_km = float(segments_real["efd_m"].sum() / 1000)
                    total_time_real_s = float(cum_t_real[-1])
                    total_km_real = float(cum_km_real[-1])

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Distanza reale", f"{total_km_real:.2f} km")
                    c2.metric("EFD reale", f"{total_efd_real_km:.2f} km")
                    c3.metric("Tempo reale totale", seconds_to_hhmm(total_time_real_s))

                    # --- tabella di confronto ai lap ---
                    rows_cmp = []
                    prev_t_prev_s, prev_t_real_s = 0.0, 0.0
                    for _, lap in plan_df_cmp.iterrows():
                        km_end = float(lap["Km arrivo"])
                        t_prev_s = hhmm_to_seconds(lap[tempo_gara_col])

                        if km_end <= total_km_real:
                            t_real_s = float(np.interp(km_end, cum_km_real, cum_t_real))
                            oltre_percorso = False
                        else:
                            # l'atleta non ha (ancora) raggiunto questo km nel file reale
                            t_real_s = np.nan
                            oltre_percorso = True

                        delta_s = (t_real_s - t_prev_s) if not np.isnan(t_real_s) else np.nan
                        seg_prev_s = t_prev_s - prev_t_prev_s
                        seg_real_s = (t_real_s - prev_t_real_s) if not np.isnan(t_real_s) else np.nan

                        rows_cmp.append({
                            "Lap": lap["Lap"],
                            "Km arrivo": km_end,
                            "Tempo previsto (cum.)": seconds_to_hhmm(t_prev_s),
                            "Tempo reale (cum.)": seconds_to_hhmm(t_real_s) if not oltre_percorso else "—",
                            "Δ cumulato": (
                                f"{'+' if delta_s >= 0 else '−'}{seconds_to_hhmm(abs(delta_s))}"
                                if not np.isnan(delta_s) else "—"
                            ),
                            "Tempo segmento previsto": seconds_to_hhmm(seg_prev_s),
                            "Tempo segmento reale": seconds_to_hhmm(seg_real_s) if not oltre_percorso else "—",
                            "_delta_s": delta_s,
                        })
                        prev_t_prev_s = t_prev_s
                        if not oltre_percorso:
                            prev_t_real_s = t_real_s

                    cmp_table = pd.DataFrame(rows_cmp)

                    st.subheader("📋 Tempi di passaggio: previsto vs reale")
                    display_cmp = cmp_table.drop(columns=["_delta_s"])
                    st.dataframe(display_cmp, use_container_width=True, hide_index=True)
                    st.download_button(
                        "📥 Scarica confronto (CSV)",
                        data=display_cmp.to_csv(index=False).encode("utf-8"),
                        file_name="confronto_race_plan_vs_reale.csv",
                        mime="text/csv",
                    )

                    valid_delta = cmp_table.dropna(subset=["_delta_s"])
                    if not valid_delta.empty:
                        final_delta_s = valid_delta["_delta_s"].iloc[-1]
                        if final_delta_s > 0:
                            st.warning(
                                f"🐢 L'atleta è arrivato **{seconds_to_hhmm(final_delta_s)} più lento** "
                                "del previsto (all'ultimo lap raggiunto)."
                            )
                        elif final_delta_s < 0:
                            st.success(
                                f"🚀 L'atleta è arrivato **{seconds_to_hhmm(abs(final_delta_s))} più veloce** "
                                "del previsto (all'ultimo lap raggiunto)."
                            )
                        else:
                            st.info("⏱️ Tempo esattamente in linea con la previsione.")

                        # --- grafico gap cumulato (previsto vs reale), Plotly ---
                        gap_min = valid_delta["_delta_s"].to_numpy() / 60.0
                        x_gap = valid_delta["Km arrivo"].to_numpy()
                        lap_names_gap = valid_delta["Lap"].to_numpy()
                        delta_s_gap = valid_delta["_delta_s"].to_numpy()
                        bar_labels = [
                            f"{'+' if s >= 0 else '−'}{seconds_to_hhmm(abs(s))}" for s in delta_s_gap
                        ]
                        bar_colors = ["#c1440e" if v >= 0 else "#2a9d8f" for v in gap_min]

                        fig_gap = go.Figure()
                        fig_gap.add_trace(go.Bar(
                            x=x_gap,
                            y=gap_min,
                            marker_color=bar_colors,
                            text=bar_labels,
                            textposition="outside",
                            customdata=lap_names_gap,
                            hovertemplate=(
                                "<b>%{customdata}</b><br>Km %{x:.1f}<br>Scarto: %{text}<extra></extra>"
                            ),
                        ))
                        fig_gap.add_hline(y=0, line_color="#333333", line_width=1)
                        fig_gap.update_layout(
                            title="Scarto cumulato: tempo reale vs tempo previsto",
                            xaxis_title="Km",
                            yaxis_title="Scarto (min)",
                            showlegend=False,
                            height=420,
                            margin=dict(t=60, b=40, l=40, r=20),
                            uniformtext_minsize=10,
                        )
                        st.caption("🟠 rosso = più lento del previsto · 🟢 verde = più veloce del previsto")
                        st.plotly_chart(fig_gap, use_container_width=True)

                    # =======================================================
                    # Equazione teorica (congelata) vs decadimento realmente osservato
                    # =======================================================
                    st.divider()
                    st.subheader("📐 Decadimento EFS: modello teorico vs gara reale")
                    st.caption(
                        "EFD accumulato in % del totale della gara reale, scostamento dell'EFS di "
                        "fetta rispetto all'EFS medio dell'INTERA gara reale, sovrapposto alla curva "
                        "**che ha effettivamente generato il race plan** (equazione congelata, non "
                        "quella ri-fittata live più in alto in questa pagina)."
                    )

                    if total_time_real_s <= 0 or total_efd_real_km <= 0:
                        st.info("Dati reali insufficienti per calcolare il decadimento.")
                    else:
                        dt_arr_r = segments_real["dt_s"].to_numpy()
                        efd_arr_r = segments_real["efd_m"].to_numpy()
                        avg_efs_real_ms = efd_arr_r.sum() / dt_arr_r.sum()

                        cum_efd_before_r = np.cumsum(efd_arr_r) - efd_arr_r
                        mid_efd_r = cum_efd_before_r + efd_arr_r / 2.0
                        efd_pct_arr_r = mid_efd_r / efd_arr_r.sum() * 100.0
                        bin_idx_r = (efd_pct_arr_r // bin_width_pct).astype(int)

                        bin_df_r = pd.DataFrame({"bin_idx": bin_idx_r, "dt_s": dt_arr_r, "efd_m": efd_arr_r})
                        grouped_r = bin_df_r.groupby("bin_idx", as_index=False).sum()
                        grouped_r = grouped_r[grouped_r["dt_s"] > 0]

                        real_decadence_rows = []
                        for _, row in grouped_r.iterrows():
                            bin_efs_ms_r = row["efd_m"] / row["dt_s"]
                            dev_pct_r = (bin_efs_ms_r - avg_efs_real_ms) / avg_efs_real_ms * 100.0
                            efd_pct_center_r = (row["bin_idx"] + 0.5) * bin_width_pct
                            real_decadence_rows.append({
                                "efd_pct_accum": efd_pct_center_r,
                                "deviation_pct_reale": dev_pct_r,
                            })
                        real_decadence_df = pd.DataFrame(real_decadence_rows)

                        x_line_r = np.linspace(0, 100, 200)
                        theoretical_line_r = np.array([
                            predict_delta_efs_frozen(p, total_efd_real_km) for p in x_line_r
                        ])

                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(
                            x=x_line_r, y=theoretical_line_r, mode="lines",
                            name=f"Teorica (piano gara, {total_efd_real_km:.0f} km EFD)",
                            line=dict(color="#1f4e79", width=3),
                        ))
                        fig3.add_trace(go.Scatter(
                            x=real_decadence_df["efd_pct_accum"], y=real_decadence_df["deviation_pct_reale"],
                            mode="markers", name="Reale (fette di gara)",
                            marker=dict(size=9, color="#c1440e"),
                        ))

                        # --- Fit di un'equazione plausibile sui punti discreti di QUESTA gara ---
                        # Stessa libreria di forme candidate usata per i bucket di training
                        # (fit_candidates/best_candidate), ma qui n_files=1: sono i bin di un
                        # solo file, quindi i modelli a più parametri (quadratica/cubica, che
                        # richiedono più file indipendenti per non overfittare) restano esclusi
                        # in automatico e si sceglie tra lineare/logaritmica/radice quadrata.
                        real_best_fit = None
                        if len(real_decadence_df) >= 6:
                            real_fit_candidates = fit_candidates(
                                real_decadence_df["efd_pct_accum"].to_numpy(),
                                real_decadence_df["deviation_pct_reale"].to_numpy(),
                                n_files=MIN_FILES_FOR_CUBIC,  # sblocca tutte le forme (lineare→cubica):
                                # qui n_files non è il vero numero di file indipendenti (è sempre 1,
                                # questa singola gara), ma un'analisi esplorativa su un solo file dove
                                # vogliamo lasciare al fit la libertà di scegliere la forma migliore
                                # tra tutte quelle disponibili, non solo quelle a 2 parametri.
                            )
                            real_best_name = best_candidate(real_fit_candidates)
                            real_best_fit = real_fit_candidates[real_best_name]
                            real_fitted_line = real_best_fit["predict"](x_line_r)

                            fig3.add_trace(go.Scatter(
                                x=x_line_r, y=real_fitted_line, mode="lines",
                                name=f"Fit reale — {real_best_name}",
                                line=dict(color="#c1440e", width=2, dash="dash"),
                            ))

                        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig3.update_layout(
                            title="Equazione che ha generato il piano vs decadimento realmente osservato",
                            xaxis_title="EFD accumulato (% della gara reale)",
                            yaxis_title="Scostamento EFS (%)",
                            height=450,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                        if real_best_fit is not None:
                            st.caption(
                                f"Equazione fittata sui dati reali di questa gara — "
                                f"miglior candidato: **{real_best_name}** "
                                f"(R² = {real_best_fit['r2']:.3f} · adj. R² = {real_best_fit['adj_r2']:.3f}). "
                                "Tutte le forme (lineare→cubica) sono disponibili, ma con un solo file "
                                "i bin non sono osservazioni indipendenti: leggi il fit come descrizione "
                                "della forma osservata in QUESTA gara, non come modello statisticamente "
                                "robusto — un adj. R² molto più alto delle curve viste nei bucket di "
                                "training è spesso solo overfitting sui pochi punti disponibili."
                            )
                            st.latex(real_best_fit["equation"])
                            with st.expander("Vedi tutte le equazioni candidate fittate sui dati reali"):
                                ranked_real = sorted(
                                    real_fit_candidates.items(),
                                    key=lambda kv: (kv[1]["adj_r2"] if not np.isnan(kv[1]["adj_r2"]) else -np.inf),
                                    reverse=True,
                                )
                                for cname, c in ranked_real:
                                    star = " 🏆 **migliore**" if cname == real_best_name else ""
                                    st.markdown(f"{cname}{star}")
                                    st.latex(c["equation"])
                                    st.caption(f"R² = {c['r2']:.3f} · adj. R² = {c['adj_r2']:.3f}")
                        else:
                            st.caption(
                                "Troppo pochi bin per fittare un'equazione affidabile sui dati reali "
                                "(servono almeno 6 fette: riduci l'ampiezza fetta o carica un file più lungo)."
                            )

                        with st.expander("Dati decadimento reale (debug/export)"):
                            st.dataframe(real_decadence_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "📥 Scarica decadimento reale (CSV)",
                                data=real_decadence_df.to_csv(index=False).encode("utf-8"),
                                file_name="decadimento_efs_reale.csv",
                                mime="text/csv",
                            )
elif plan_csv is not None or real_fit is not None:
    st.info("Carica sia il CSV del race plan sia il file .fit della gara reale per procedere.")