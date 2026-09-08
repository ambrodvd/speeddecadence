"""
ANALISI 5 — CPET a step: ricerca dello steady state e pulizia dei bin.

Il metabolimetro è a camera di miscelazione: ogni riga è già la media pesata
sul volume del gas espirato nei 30 s di riempimento della camera, NON la media
di N respiri. Conseguenze pratiche su cui è costruito tutto questo modulo:

1. Il rumore respiro-per-respiro non esiste: la varianza residua dentro uno
   step è camera + maschera + andatura + fluttuazione fisiologica vera.
2. I bin consecutivi sono autocorrelati (la camera non si svuota del tutto tra
   un campione e l'altro), quindi SD/sqrt(n) SOTTOSTIMA l'errore standard.
   CV e SD qui vanno letti come indici descrittivi di qualità, non come
   statistica inferenziale.
3. Il transitorio a inizio step è più lungo della sola cinetica del VO2,
   perché la camera "spalma" il gas dello step precedente: default di taglio
   a 90 s, non 60.
4. Uno spike su UN SOLO bin è sospetto (un colpo di tosse verrebbe mediato via
   dalla camera): serve un evento vero — perdita di maschera, inciampo — per
   muovere un intero campione da 30 s. Per questo il filtro automatico qui
   SUGGERISCE e basta: la decisione resta manuale.

Uso nell'app principale:

    from cpet_steady_state import render_cpet_analysis
    ...
    with tab_cpet:
        run_cpet = st.checkbox(RUN_PROMPT, value=False, key="run_cpet")
        if run_cpet:
            render_cpet_analysis()
        else:
            st.info("☝️ Spunta la casella per eseguire l'analisi CPET a step.")
"""

import csv
import hashlib
import io
import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- colonne del tracciato che hanno senso come numeri ---------------------
CPET_NUMERIC_COLS = [
    "Rf", "Vt", "Ve", "VO2Kg", "VO2", "VCO2", "RQ", "EEm",
    "Ve/VO2", "Ve/VCO2", "FeO2", "FeCO2", "FiO2", "FiCO2",
    "RPE", "HR", "La-", "Speed", "Grade", "RPM",
]

# Colonne mostrate di default: solo quelle che servono a leggere il VT1.
#   VO2 / VCO2      -> V-slope (Beaver 1986): il ginocchio nella retta
#                      VCO2 su VO2 è il criterio primario.
#   Ve/VO2, Ve/VCO2 -> equivalenti ventilatori: al VT1 l'equivalente
#                      dell'O2 risale mentre quello della CO2 resta piatto.
#                      Serve la COPPIA: se risalgono entrambi sei al VT2.
#   FeO2            -> surrogato del criterio end-tidal. ATTENZIONE: con una
#                      camera di miscelazione il PETO2 NON esiste — l'aria
#                      espirata viene mediata per intero, spazio morto
#                      compreso. FeO2 sale al VT1 come farebbe il PETO2 ma
#                      con l'inflessione smussata: corroborazione, non prova.
#   HR              -> contesto.
CPET_DEFAULT_COLS = ["VO2", "VCO2", "Ve/VO2", "Ve/VCO2", "FeO2", "HR"]

# Tutte le colonne selezionabili nella tabella (le altre restano comunque
# nell'export completo).
CPET_ALL_COLS = ["VO2", "VO2Kg", "VCO2", "Ve", "Ve/VO2", "Ve/VCO2", "RQ",
                 "FeO2", "FeCO2", "HR", "Rf", "Vt", "EEm"]

# Decimali per colonna. Senza questo lo Styler stampa 6 cifre su tutto e
# un VO2 di 2701 mL/min diventa "2701.000000", che rende la tabella
# illeggibile proprio dove serve leggerla in fretta.
CPET_COL_DECIMALS = {
    "VO2": 0, "VCO2": 0, "Ve": 1, "VO2Kg": 1,
    "Ve/VO2": 1, "Ve/VCO2": 1, "RQ": 2,
    "FeO2": 2, "FeCO2": 2, "HR": 0, "Rf": 1, "Vt": 0, "EEm": 2,
}


def cpet_col_formats(cols):
    """Dizionario colonna -> stringa di formato per Styler.format."""
    return {c: f"{{:.{CPET_COL_DECIMALS.get(c, 2)}f}}"
            for c in cols if c in CPET_COL_DECIMALS}


# Colori delle celle. Sono rgba con alpha moderata e NON impostano il colore
# del testo: così restano leggibili sia sul tema chiaro sia su quello scuro,
# dove un background pieno con testo nero sarebbe illeggibile.
CELL_HARD = "background-color: rgba(193,68,14,0.45);"
CELL_SOFT = "background-color: rgba(193,68,14,0.18);"
CELL_GATE = "background-color: rgba(140,30,120,0.45);"
# ATTENZIONE: st.dataframe applica dallo Styler SOLO `color` e
# `background-color`. `opacity`, `text-decoration`, `font-weight` vengono
# ignorati in silenzio — è il motivo per cui le righe non usate risultavano
# identiche alle altre. Il "barrato" qui è quindi un testo grigio.
ROW_UNUSED = " color: #7a7a7a;"


# Variabili su cui si valuta lo steady state di ogni step.
CPET_SUMMARY_VARS = ["VO2", "VCO2", "Ve", "Ve/VO2", "Ve/VCO2", "VO2Kg",
                     "RQ", "FeO2", "HR"]

# Variabili candidate per la regola combinata di flagging. Solo segnali
# MISURATI, mai rapporti: Ve/VO2 e RQ sono funzioni di VO2, VCO2 e Ve, quindi
# un solo VO2 sbagliato le farebbe scattare tutte insieme e la regola "almeno
# due variabili" — che esiste per pretendere conferme indipendenti — si
# accontenterebbe della stessa anomalia contata tre volte.
CPET_FLAG_VARS = ["VO2", "VCO2", "Ve", "HR", "Rf", "Vt"]
CPET_FLAG_DEFAULT = ["VO2", "VCO2", "Ve"]

CPET_TRIM_OPTIONS = [0, 30, 60, 90, 120, 150, 180]

# Cancelli di plausibilità fisiologica (bin-level, assoluti).
CPET_GATES = {
    "RQ": (0.60, 1.25),
    "VO2": (200.0, 8000.0),
    "VO2Kg": (1.0, 100.0),
    "Ve": (5.0, 250.0),
    "Rf": (5.0, 90.0),
    "HR": (30.0, 230.0),
}

# ---------------------------------------------------------------------------
# Quanto si restringe la SD togliendo i k punti più estremi da n osservazioni
# di puro rumore gaussiano (nessun outlier vero). Monte Carlo, 120k repliche.
# Serve a smascherare la pulizia cosmetica: se togliendo 2 bin su 8 la SD cala
# del 36% non hai tolto outlier, hai tolto le code.
# ---------------------------------------------------------------------------
SD_SHRINK_EXPECTED = {
    4: [0.567, 0.456, None, None],
    5: [0.655, 0.521, 0.285, None],
    6: [0.707, 0.566, 0.387, 0.271],
    7: [0.743, 0.604, 0.455, 0.353],
    8: [0.770, 0.634, 0.505, 0.408],
    9: [0.790, 0.661, 0.544, 0.450],
    10: [0.806, 0.683, 0.576, 0.487],
    11: [0.819, 0.702, 0.602, 0.517],
    12: [0.831, 0.719, 0.624, 0.543],
    13: [0.841, 0.734, 0.644, 0.567],
    14: [0.849, 0.747, 0.661, 0.586],
    15: [0.857, 0.758, 0.676, 0.605],
    16: [0.863, 0.769, 0.690, 0.621],
    17: [0.869, 0.778, 0.703, 0.636],
    18: [0.875, 0.787, 0.714, 0.650],
    19: [0.879, 0.795, 0.724, 0.662],
    20: [0.884, 0.802, 0.734, 0.673],
    21: [0.888, 0.808, 0.742, 0.684],
    22: [0.892, 0.814, 0.750, 0.693],
    23: [0.895, 0.820, 0.757, 0.702],
    24: [0.898, 0.825, 0.764, 0.711],
}


def expected_sd_ratio(n: int, k: int):
    """Rapporto SD atteso (dopo/prima) togliendo k punti su n per solo caso."""
    if k <= 0:
        return 1.0
    if n > 24:
        n = 24
    row = SD_SHRINK_EXPECTED.get(n)
    if not row or k > len(row):
        return np.nan
    return row[k - 1]


# ===========================================================================
# Parsing del CSV del metabolimetro
# ===========================================================================
_TIME_RE = re.compile(r"^\d{1,3}:\d{2}:\d{2}$")


def _to_float(v):
    if v is None:
        return np.nan
    s = str(v).strip().replace(",", ".")
    if s in ("", "-", "--", "N/A", "n/a"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _hhmmss_to_s(t: str):
    parts = str(t).strip().split(":")
    try:
        parts = [int(p) for p in parts]
    except ValueError:
        return np.nan
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return np.nan


def parse_cpet_csv(file_bytes: bytes):
    """Legge l'export del metabolimetro.

    Il file ha un blocco di metadati nelle prime colonne e la tabella del
    tracciato che parte dalla colonna dove compare l'intestazione 'Time'.
    L'offset viene cercato, non assunto: export diversi possono spostarlo.

    Ritorna (meta: dict, df: DataFrame) oppure (None, None) se illeggibile.
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = file_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return None, None

    first_line = text.split("\n", 1)[0]
    delim = ";" if first_line.count(";") >= first_line.count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    if not rows:
        return None, None

    header_row = rows[0]
    offset = None
    for i, cell in enumerate(header_row):
        if str(cell).strip().lower() == "time":
            offset = i
            break
    if offset is None:
        return None, None

    names = [str(c).strip() for c in header_row[offset:]]
    units = ([str(c).strip() for c in rows[1][offset:]]
             if len(rows) > 1 and len(rows[1]) > offset else [""] * len(names))

    # --- metadati: coppie etichetta/valore nel blocco a sinistra ---
    meta = {}
    for r in rows:
        for a, b in ((0, 1), (3, 4), (6, 7)):
            if len(r) > b:
                k = str(r[a]).strip()
                v = str(r[b]).strip()
                if k and v and k not in meta:
                    meta[k] = v

    # --- righe dati: quelle in cui la prima colonna del tracciato è un tempo ---
    data = []
    for r in rows:
        if len(r) <= offset:
            continue
        if not _TIME_RE.match(str(r[offset]).strip()):
            continue
        vals = list(r[offset:]) + [""] * (len(names) - len(r[offset:]))
        data.append(vals[:len(names)])

    if not data:
        return meta, None

    df = pd.DataFrame(data, columns=names)
    df["t_s"] = df["Time"].map(_hhmmss_to_s)
    for c in CPET_NUMERIC_COLS:
        if c in df.columns:
            df[c] = df[c].map(_to_float)
    for c in ("Phase", "Marker", "Info"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df = df.dropna(subset=["t_s"]).sort_values("t_s").reset_index(drop=True)
    df.attrs["units"] = dict(zip(names, units))
    return meta, df


# ===========================================================================
# Segmentazione in step
# ===========================================================================
def assign_stages(df: pd.DataFrame) -> pd.DataFrame:
    """Blocchi contigui alla stessa velocità.

    Il raggruppamento è sulla SOLA velocità: se l'atleta corre 6 min di
    riscaldamento a 10.5 e poi 4 min di 'Exercise' sempre a 10.5, è un unico
    step da 10 minuti, perché è quello che ha fatto davvero il metabolismo.
    La fase viene riportata come etichetta, non usata per tagliare.
    """
    df = df.copy()
    spd = df["Speed"].fillna(-999.0).round(2)
    new_block = (spd != spd.shift()).cumsum()
    df["stage_id"] = new_block

    # epoca di campionamento (di norma 30 s): mediana dei delta
    diffs = df["t_s"].diff().dropna()
    epoch = float(diffs.median()) if len(diffs) else 30.0
    if not np.isfinite(epoch) or epoch <= 0:
        epoch = 30.0
    df.attrs["epoch_s"] = epoch

    # t_s è la FINE del bin; l'inizio dello step è l'inizio del suo primo bin
    starts = df.groupby("stage_id")["t_s"].transform("min") - epoch
    df["t_in_stage_s"] = df["t_s"] - starts
    df["stage_speed"] = df.groupby("stage_id")["Speed"].transform("median")

    if "Phase" in df.columns:
        ph = (df.groupby("stage_id")["Phase"]
                .transform(lambda s: "/".join(pd.unique(s.dropna()))))
        df["stage_phase"] = ph
    else:
        df["stage_phase"] = ""

    df["stage_label"] = df.apply(
        lambda r: f"{r['stage_speed']:.1f} km/h  ({r['stage_phase']})", axis=1
    )
    return df


def is_rest_stage(speed, phase) -> bool:
    """Step da escludere di default: fermi o recupero."""
    if not np.isfinite(speed) or speed <= 0.1:
        return True
    return "recovery" in str(phase).lower()


# ===========================================================================
# Flagging combinato (row-level)
# ===========================================================================
def detrended_residuals(t, y):
    """Residui rispetto alla retta interna allo step.

    Il detrending prima del flagging serve a non punire una deriva vera: una
    deriva è il SEGNALE dello steady state mancato, non una fila di outlier.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    ok = np.isfinite(y) & np.isfinite(t)
    resid = np.full(len(y), np.nan)
    if ok.sum() < 3:
        return resid
    if np.ptp(t[ok]) > 0:
        coef = np.polyfit(t[ok], y[ok], 1)
        resid[ok] = y[ok] - np.polyval(coef, t[ok])
    else:
        resid[ok] = y[ok] - np.nanmedian(y[ok])
    return resid


def pooled_noise_scale(df: pd.DataFrame, var: str, mask=None) -> float:
    """Un unico metro del rumore per variabile, stimato su TUTTO il test.

    Il MAD calcolato dentro il singolo step è inutilizzabile: con 5 bin capita
    che tre valori siano quasi identici, il MAD collassa a ~0 e ogni normale
    fluttuazione diventa uno z di 15. Qui i residui detrendizzati di tutti gli
    step vengono messi in comune e il MAD si stima su ~100 punti: il metro
    diventa stabile e gli z sono confrontabili tra step. È anche l'analogo
    autoconsistente dell'MDC da test-retest, che non hai.
    """
    if var not in df.columns:
        return np.nan
    pool = []
    sub_all = df if mask is None else df[mask]
    for _, sub in sub_all.groupby("stage_id"):
        if len(sub) < 4:
            continue
        r = detrended_residuals(sub["t_in_stage_s"].to_numpy(dtype=float),
                                sub[var].to_numpy(dtype=float))
        r = r[np.isfinite(r)]
        if len(r) < 3:
            continue
        # Il detrending consuma 2 gradi di libertà: su 4 bin i residui sono
        # sistematicamente più piccoli del rumore vero e il metro si
        # accorcerebbe man mano che alzi il taglio, facendo esplodere i flag
        # proprio quando i dati sono di meno. Riscalatura per sqrt(n/(n-2)).
        pool.append(r * np.sqrt(len(r) / (len(r) - 2)))
    if not pool:
        return np.nan
    allr = np.concatenate(pool)
    if len(allr) < 8:
        return np.nan
    scale = 1.4826 * float(np.median(np.abs(allr - np.median(allr))))
    return scale if np.isfinite(scale) and scale > 1e-9 else np.nan


def flag_stage(sub: pd.DataFrame, flag_vars, z_hard: float, z_soft: float,
               scales: dict):
    """Regola COMBINATA, esito a livello di riga.

    Un bin viene segnalato se:
      - una qualsiasi variabile supera z_hard (evento grosso e isolato), oppure
      - almeno DUE variabili superano z_soft (anomalia coerente su più segnali:
        è la firma di un evento reale, non di rumore su un canale solo), oppure
      - una variabile esce dai cancelli di plausibilità fisiologica.
    Un solo campione di camera è una sola misura fisica: se è sporco, lo è per
    tutti i segnali che ne derivano. Per questo l'esclusione è di riga.
    """
    n = len(sub)
    t = sub["t_in_stage_s"].to_numpy(dtype=float)
    zmat, used = {}, []
    for v in flag_vars:
        s = scales.get(v, np.nan)
        if v not in sub.columns or not np.isfinite(s):
            continue
        zmat[v] = detrended_residuals(t, sub[v].to_numpy(dtype=float)) / s
        used.append(v)

    reasons = [[] for _ in range(n)]
    zmax = np.zeros(n)

    for v in used:
        z = zmat[v]
        az = np.abs(z)
        zmax = np.fmax(zmax, np.nan_to_num(az, nan=0.0))
        for i in range(n):
            if np.isfinite(az[i]):
                if az[i] >= z_hard:
                    reasons[i].append(f"{v} z={z[i]:+.1f}")
                elif az[i] >= z_soft:
                    reasons[i].append(f"({v} z={z[i]:+.1f})")

    soft_counts = np.zeros(n, dtype=int)
    hard_hits = np.zeros(n, dtype=bool)
    for v in used:
        az = np.abs(zmat[v])
        soft_counts += np.nan_to_num(az, nan=0.0) >= z_soft
        hard_hits |= np.nan_to_num(az, nan=0.0) >= z_hard

    gate_hits = np.zeros(n, dtype=bool)
    gate_map = {}
    for v, (lo, hi) in CPET_GATES.items():
        if v not in sub.columns:
            continue
        x = sub[v].to_numpy(dtype=float)
        bad = np.isfinite(x) & ((x < lo) | (x > hi))
        gate_map[v] = bad
        for i in np.where(bad)[0]:
            reasons[i].append(f"{v} fuori range")
        gate_hits |= bad

    for v in flag_vars:
        if v in sub.columns:
            miss = ~np.isfinite(sub[v].to_numpy(dtype=float))
            for i in np.where(miss)[0]:
                reasons[i].append(f"{v} mancante")
            gate_hits |= miss

    flagged = hard_hits | (soft_counts >= 2) | gate_hits
    # zmat serve alla tabella per colorare la SINGOLA cella fuori scala, non
    # tutta la riga: è la differenza tra "questo bin è sospetto" e "questo
    # bin è sospetto per colpa della VCO2".
    return flagged, zmax, ["; ".join(r) for r in reasons], zmat, gate_map


# ===========================================================================
# Statistiche di step
# ===========================================================================
def stage_stats(sub: pd.DataFrame, var: str) -> dict:
    """Descrittive + due indici di steady state (deriva e delta metà/metà)."""
    y = pd.to_numeric(sub[var], errors="coerce").dropna()
    out = {"N": int(len(y))}
    if len(y) == 0:
        return {**out, "Media": np.nan, "SD": np.nan, "CV %": np.nan,
                "SEM": np.nan, "Mediana": np.nan, "Min": np.nan, "Max": np.nan,
                "Range": np.nan, "IQR": np.nan, "Deriva (/min)": np.nan,
                "Δ metà (%)": np.nan}

    t = sub.loc[y.index, "t_in_stage_s"].to_numpy(dtype=float) / 60.0
    arr = y.to_numpy(dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan

    slope = np.nan
    if len(arr) >= 3 and np.ptp(t) > 0:
        slope = float(np.polyfit(t, arr, 1)[0])

    half = len(arr) // 2
    delta_pct = np.nan
    if half >= 1 and len(arr) >= 4 and mean != 0:
        delta_pct = float((arr[half:].mean() - arr[:half].mean()) / mean * 100.0)

    return {
        **out,
        "Media": mean,
        "SD": sd,
        "CV %": (sd / mean * 100.0) if (np.isfinite(sd) and mean) else np.nan,
        "SEM": (sd / np.sqrt(len(arr))) if np.isfinite(sd) else np.nan,
        "Mediana": float(np.median(arr)),
        "Min": float(arr.min()),
        "Max": float(arr.max()),
        "Range": float(arr.max() - arr.min()),
        "IQR": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "Deriva (/min)": slope,
        "Δ metà (%)": delta_pct,
    }


def steady_verdict(stats: dict, cv_max: float, delta_max: float) -> str:
    """Verdetto sullo steady state di uno step.

    Due criteri, entrambi descrittivi e nessuno dei due inferenziale (i bin
    sono autocorrelati, un test formale sulla pendenza sarebbe una finzione):
    dispersione (CV) e delta tra prima e seconda metà dello step, che è
    l'analogo del confronto 4° vs 5° minuto usato in letteratura.
    """
    if stats["N"] < 4:
        return "◌ pochi bin"
    cv = stats.get("CV %", np.nan)
    d = abs(stats.get("Δ metà (%)", np.nan))
    if not np.isfinite(cv) or not np.isfinite(d):
        return "◌ pochi bin"
    if cv <= cv_max and d <= delta_max:
        return "✅ raggiunto"
    if cv <= cv_max * 1.5 and d <= delta_max * 1.5:
        return "⚠️ marginale"
    return "❌ non raggiunto"


def suggest_trim(sub: pd.DataFrame, var: str, cv_max: float, delta_max: float,
                 options=CPET_TRIM_OPTIONS, min_n: int = 4,
                 min_trim: int = 60):
    """Taglio minimo, per QUESTO step, che porta la finestra a passare.

    Risponde al 'dove' invece che al solo 'se': se a 60 s lo step è già
    stabile, tagliarne 120 butta metà campione per niente; se serve andare a
    150 s, quello step non ha avuto il tempo di equilibrarsi.

    Il pavimento a 60 s non è cosmetico. Senza, la funzione consiglierebbe
    taglio 0 su step che a 90 s falliscono: tenendo dentro il transitorio in
    salita, la prima metà si abbassa, il delta metà/metà si annulla per
    compensazione e uno step in deriva sembra stabile. Nei primi 60 s, tra
    cinetica del VO2 e riempimento della camera, lo steady state non può
    esserci per costruzione.
    """
    for tr in options:
        if tr < min_trim:
            continue
        win = sub[sub["t_in_stage_s"] > tr]
        if len(win) < min_n:
            return None
        if steady_verdict(stage_stats(win, var), cv_max,
                          delta_max).startswith("✅"):
            return tr
    return None


# ===========================================================================
# VT1 — soglia aerobica
# ===========================================================================
# Il VT1 non è una domanda sul singolo step ma sul RAPPORTO tra gli step: la
# pulizia dei bin serve a produrre una media pulita per step, e il VT1 si
# legge nella forma della nuvola di quelle medie.
#
# Due pannelli, entrambi con barre d'errore pari al SEM dei bin usati: senza
# quelle non si distingue un ginocchio vero da una gobba di rumore, ed è
# esattamente l'errore che si commette guardando solo i punti.
#
#   1. V-slope (Beaver 1986): VCO2 su VO2. Non dipende dal drive
#      ventilatorio, quindi è il criterio primario.
#   2. Equivalenti ventilatori: Ve/VO2 risale mentre Ve/VCO2 resta piatto.
#      Se risalgono insieme non sei al VT1 ma al VT2.

def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Frazione continua di Lentz per la beta incompleta (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = c if abs(c) > 1e-300 else 1e-300
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Beta incompleta regolarizzata I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_sf(f_stat, d1, d2):
    """P(F > f_stat). Scritta a mano per non aggiungere scipy alle dipendenze
    dell'app: verificata contro scipy.stats.f.sf a meno di 1e-15."""
    if not np.isfinite(f_stat) or f_stat <= 0 or d1 <= 0 or d2 <= 0:
        return 1.0
    return _betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f_stat))


def fit_two_segment(x, y, min_side=3, s1_min=None, s2_min=None,
                    require_s2_gt_s1=True):
    """Regressione a due segmenti continui, punto di rottura per griglia.

    Il punto di rottura viene cercato provando ogni possibile divisione dei
    punti e tenendo quella a somma dei quadrati minima, con i due segmenti
    vincolati a incontrarsi (modello continuo, 4 parametri: intercetta, due
    pendenze, ascissa di rottura).

    I vincoli NON sono cosmetici. Su questi dati la ricerca libera trova
    facilmente un ottimo matematico privo di senso fisiologico — su un test
    reale il primo segmento del V-slope è uscito con pendenza -0.49, cioè
    VCO2 che cala mentre VO2 sale. `s1_min`/`s2_min` e la richiesta che la
    seconda pendenza superi la prima escludono quelle soluzioni.

    Il test F confronta il modello spezzato con la retta singola: se la
    rottura non riduce i residui più di quanto farebbero due parametri in
    più su del rumore, l'onesta conclusione è "non risolvibile", non un
    numero con tre decimali.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    if n < 2 * min_side:
        return {"valid": False, "reason": f"servono almeno {2 * min_side} step, "
                                          f"ce ne sono {n}", "n": n}

    c_lin = np.polyfit(x, y, 1)
    ssr_lin = float(((y - np.polyval(c_lin, x)) ** 2).sum())

    best = None
    for i in range(min_side - 1, n - min_side):
        # candidato: rottura tra x[i] e x[i+1], cercata su una griglia fine
        for xb in np.linspace(x[i], x[i + 1], 12):
            # base spline lineare continua: y = a + b1*x + b2*max(0, x-xb)
            basis = np.column_stack([np.ones(n), x, np.maximum(0.0, x - xb)])
            try:
                coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            resid = y - basis @ coef
            ssr = float((resid ** 2).sum())
            s1 = float(coef[1])
            s2 = float(coef[1] + coef[2])
            if s1_min is not None and s1 < s1_min:
                continue
            if s2_min is not None and s2 < s2_min:
                continue
            if require_s2_gt_s1 and s2 <= s1:
                continue
            if best is None or ssr < best["ssr"]:
                best = {"ssr": ssr, "xb": float(xb), "s1": s1, "s2": s2,
                        "intercept": float(coef[0])}

    if best is None:
        return {"valid": False, "n": n, "ssr_lin": ssr_lin,
                "reason": "nessuna rottura compatibile con i vincoli "
                          "fisiologici (pendenze del segno giusto e in aumento)"}

    df_seg = n - 4
    if df_seg <= 0 or best["ssr"] <= 0:
        return {**best, "valid": False, "n": n, "ssr_lin": ssr_lin,
                "reason": "troppo pochi step per testare la rottura"}

    f_stat = ((ssr_lin - best["ssr"]) / 2.0) / (best["ssr"] / df_seg)
    p = f_sf(f_stat, 2, df_seg)
    yb = best["intercept"] + best["s1"] * best["xb"]
    return {**best, "n": n, "ssr_lin": ssr_lin, "F": float(f_stat),
            "p": float(p), "yb": float(yb), "df": df_seg,
            "valid": p < 0.05,
            "reason": ("" if p < 0.05 else
                       "la rottura non spiega i dati meglio di una retta "
                       f"singola (p = {p:.2f}): non risolvibile")}


def segment_lines(fit, x_min, x_max):
    """Due spezzoni pronti da disegnare: (x1, y1), (x2, y2)."""
    xb, a, s1, s2 = fit["xb"], fit["intercept"], fit["s1"], fit["s2"]
    yb = a + s1 * xb
    x1 = np.array([x_min, xb])
    x2 = np.array([xb, x_max])
    return (x1, a + s1 * x1), (x2, yb + s2 * (x2 - xb))


VT1_MEAN_VARS = ["VO2", "VCO2", "Ve", "Ve/VO2", "Ve/VCO2", "FeO2", "HR"]


def step_means(df: pd.DataFrame, var_list=VT1_MEAN_VARS) -> pd.DataFrame:
    """Una riga per step con media e SEM dei bin selezionati.

    Il SEM è SD/sqrt(N) dei bin usati. Va letto come indicazione di
    dispersione, non come intervallo di confidenza: i bin di una camera di
    miscelazione sono autocorrelati, quindi il vero errore standard è più
    grande di così.
    """
    rows = []
    for _, sub in df.groupby("stage_id"):
        used = sub[sub["used"]]
        if used.empty:
            continue
        row = {
            "stage_id": int(sub["stage_id"].iloc[0]),
            "Step": sub["stage_label"].iloc[0],
            "km/h": float(sub["stage_speed"].iloc[0]),
            "N bin": int(len(used)),
        }
        for v in var_list:
            if v not in used.columns:
                continue
            s = pd.to_numeric(used[v], errors="coerce").dropna()
            row[v] = float(s.mean()) if len(s) else np.nan
            row[f"{v} SEM"] = (float(s.std(ddof=1) / np.sqrt(len(s)))
                               if len(s) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("km/h").reset_index(drop=True)


def _interp_at(means: pd.DataFrame, x_col: str, x_val: float, y_col: str):
    """Valore di y in corrispondenza di x_val, interpolando tra gli step."""
    d = means[[x_col, y_col]].dropna().sort_values(x_col)
    if len(d) < 2 or not np.isfinite(x_val):
        return np.nan
    xs, ys = d[x_col].to_numpy(), d[y_col].to_numpy()
    if x_val < xs[0] or x_val > xs[-1]:
        return np.nan
    return float(np.interp(x_val, xs, ys))


def _errorbar_trace(x, y, ex, ey, name, color, symbol="circle"):
    return go.Scatter(
        x=x, y=y, mode="markers", name=name,
        marker=dict(size=9, color=color, symbol=symbol),
        error_x=dict(type="data", array=ex, visible=True, color=color,
                     thickness=1.2, width=4) if ex is not None else None,
        error_y=dict(type="data", array=ey, visible=True, color=color,
                     thickness=1.2, width=4) if ey is not None else None,
    )


def render_vslope_panel(means: pd.DataFrame):
    """Pannello 1 — V-slope: VCO2 su VO2."""
    st.markdown("#### 1. V-slope — VCO₂ su VO₂")
    st.caption(
        "Criterio primario (Beaver 1986): al VT1 il tampone bicarbonato "
        "aggiunge CO₂ non metabolica, quindi la VCO₂ inizia a salire più in "
        "fretta della VO₂ e la retta si piega verso l'alto. Non dipende dal "
        "drive ventilatorio, per questo regge anche quando la ventilazione è "
        "irregolare. Le barre sono il SEM dei bin selezionati in ogni step."
    )
    d = means.dropna(subset=["VO2", "VCO2"])
    if len(d) < 6:
        st.info("Servono almeno 6 step con dati validi per cercare la rottura.")
        return None

    fit = fit_two_segment(d["VO2"].to_numpy(), d["VCO2"].to_numpy(),
                          min_side=3, s1_min=0.30, s2_min=0.30,
                          require_s2_gt_s1=True)

    fig = go.Figure()
    x0, x1 = float(d["VO2"].min()), float(d["VO2"].max())
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[x0, x1], mode="lines", name="pendenza 1 (rif.)",
        line=dict(color="#bbbbbb", dash="dot", width=1),
        hoverinfo="skip",
    ))
    fig.add_trace(_errorbar_trace(
        d["VO2"], d["VCO2"], d.get("VO2 SEM"), d.get("VCO2 SEM"),
        "Medie di step", "#1f4e79"))

    if fit.get("xb") is not None:
        (xa, ya), (xb_, yb_) = segment_lines(fit, x0, x1)
        col = "#2a9d8f" if fit["valid"] else "#9e9e9e"
        fig.add_trace(go.Scatter(x=xa, y=ya, mode="lines",
                                 name=f"S1 = {fit['s1']:.3f}",
                                 line=dict(color=col, width=2.5)))
        fig.add_trace(go.Scatter(x=xb_, y=yb_, mode="lines",
                                 name=f"S2 = {fit['s2']:.3f}",
                                 line=dict(color=col, width=2.5, dash="dash")))
        fig.add_vline(x=fit["xb"], line_dash="dot",
                      line_color=("#c1440e" if fit["valid"] else "#cccccc"))

    fig.update_layout(
        xaxis_title="VO₂ (mL/min)", yaxis_title="VCO₂ (mL/min)",
        height=430, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    if fit["valid"]:
        vo2_b = fit["xb"]
        spd = _interp_at(means, "VO2", vo2_b, "km/h")
        hr = _interp_at(means, "VO2", vo2_b, "HR")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VO₂ al VT1", f"{vo2_b:.0f} mL/min")
        c2.metric("Velocità", f"{spd:.2f} km/h" if np.isfinite(spd) else "—")
        c3.metric("FC", f"{hr:.0f} bpm" if np.isfinite(hr) else "—")
        c4.metric("Δ pendenza", f"{fit['s2'] - fit['s1']:+.3f}",
                  help=f"F = {fit['F']:.2f} su (2, {fit['df']}) · p = {fit['p']:.3f}")
        st.caption(
            f"La rottura spiega i dati meglio di una retta singola "
            f"(F = {fit['F']:.2f}, p = {fit['p']:.3f}). Velocità e FC sono "
            "interpolate tra gli step, quindi la loro risoluzione non può "
            "essere migliore del passo del protocollo (0.5 km/h)."
        )
    else:
        st.warning(f"⚠️ **VT1 non risolvibile dal V-slope** — {fit['reason']}")
        st.caption(
            "Non è un errore del calcolo: con 9-12 medie di step e un SEM "
            "dell'ordine dell'1-2% un ginocchio piccolo è indistinguibile dal "
            "rumore. Serve o un protocollo con più step sotto la soglia "
            "attesa, o incrementi di velocità più fini vicino ad essa."
        )
    return fit


def render_equivalents_panel(means: pd.DataFrame):
    """Pannello 2 — equivalenti ventilatori contro velocità."""
    st.markdown("#### 2. Equivalenti ventilatori")
    st.caption(
        "Al VT1 la ventilazione cresce più in fretta del consumo di O₂ ma "
        "resta accoppiata alla CO₂ prodotta: Ve/VO₂ risale mentre Ve/VCO₂ è "
        "ancora piatto o in calo. **Se risalgono insieme non sei al VT1, sei "
        "al VT2.** Il minimo va letto sul fit, non sul punto più basso: i due "
        "rapporti condividono la Ve, quindi un singolo step con ventilazione "
        "bassa li fa scendere entrambi e produce un falso nadir."
    )
    d = means.dropna(subset=["km/h", "Ve/VO2"])
    if len(d) < 6:
        st.info("Servono almeno 6 step con dati validi.")
        return None

    fit = fit_two_segment(d["km/h"].to_numpy(), d["Ve/VO2"].to_numpy(),
                          min_side=3, s2_min=0.0, require_s2_gt_s1=True)

    fig = go.Figure()
    fig.add_trace(_errorbar_trace(
        d["km/h"], d["Ve/VO2"], None, d.get("Ve/VO2 SEM"),
        "Ve/VO₂", "#2a9d8f"))
    if "Ve/VCO2" in means.columns:
        d2 = means.dropna(subset=["km/h", "Ve/VCO2"])
        fig.add_trace(_errorbar_trace(
            d2["km/h"], d2["Ve/VCO2"], None, d2.get("Ve/VCO2 SEM"),
            "Ve/VCO₂", "#c1440e", symbol="square"))

    if fit.get("xb") is not None:
        x0, x1 = float(d["km/h"].min()), float(d["km/h"].max())
        (xa, ya), (xb_, yb_) = segment_lines(fit, x0, x1)
        col = "#2a9d8f" if fit["valid"] else "#9e9e9e"
        fig.add_trace(go.Scatter(x=xa, y=ya, mode="lines", showlegend=False,
                                 line=dict(color=col, width=2)))
        fig.add_trace(go.Scatter(x=xb_, y=yb_, mode="lines", showlegend=False,
                                 line=dict(color=col, width=2, dash="dash")))
        fig.add_vline(x=fit["xb"], line_dash="dot",
                      line_color=("#c1440e" if fit["valid"] else "#cccccc"))

    fig.update_layout(
        xaxis_title="Velocità (km/h)", yaxis_title="Equivalente ventilatorio",
        height=430, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")

    raw_min_speed = float(d.loc[d["Ve/VO2"].idxmin(), "km/h"])

    if fit["valid"]:
        spd = fit["xb"]
        vo2 = _interp_at(means, "km/h", spd, "VO2")
        hr = _interp_at(means, "km/h", spd, "HR")
        c1, c2, c3 = st.columns(3)
        c1.metric("Velocità al nadir", f"{spd:.2f} km/h")
        c2.metric("VO₂", f"{vo2:.0f} mL/min" if np.isfinite(vo2) else "—")
        c3.metric("FC", f"{hr:.0f} bpm" if np.isfinite(hr) else "—")
        if abs(raw_min_speed - spd) > 0.75:
            st.caption(
                f"Il punto grezzo più basso cade a {raw_min_speed:.1f} km/h, "
                f"lontano dal nadir del fit ({spd:.2f} km/h): quello step ha "
                "semplicemente una Ve bassa. È il motivo per cui il minimo si "
                "legge sul fit."
            )
    else:
        st.warning(f"⚠️ **Nadir non risolvibile** — {fit['reason']}")
        st.caption(
            f"Il punto grezzo più basso è a {raw_min_speed:.1f} km/h, ma senza "
            "una risalita statisticamente distinguibile è solo il minimo del "
            "rumore. Se gli equivalenti salgono da subito, il VT1 può trovarsi "
            "**sotto la prima velocità testata**: in quel caso il protocollo "
            "va esteso verso il basso, non analizzato meglio."
        )
    return fit


def render_vt1_section(df: pd.DataFrame):
    st.subheader("🎯 VT1 — soglia aerobica")
    means = step_means(df)
    if means.empty or len(means) < 6:
        st.info("Servono almeno 6 step con bin selezionati per stimare il VT1.")
        return

    thin = means[means["N bin"] < 3]
    if not thin.empty:
        st.warning(
            "⚠️ Step con meno di 3 bin selezionati: "
            + ", ".join(thin["Step"].tolist())
            + ". Le loro medie pesano quanto le altre nel fit pur essendo "
            "molto meno affidabili."
        )

    v_fit = render_vslope_panel(means)
    st.divider()
    e_fit = render_equivalents_panel(means)

    if v_fit and e_fit and v_fit.get("valid") and e_fit.get("valid"):
        v_spd = _interp_at(means, "VO2", v_fit["xb"], "km/h")
        e_spd = e_fit["xb"]
        if np.isfinite(v_spd):
            gap = abs(v_spd - e_spd)
            msg = (f"V-slope {v_spd:.2f} km/h · equivalenti {e_spd:.2f} km/h "
                   f"· scarto {gap:.2f} km/h.")
            if gap <= 0.5:
                st.success("✅ I due criteri concordano. " + msg)
            else:
                st.warning(
                    "⚠️ I due criteri non concordano. " + msg
                    + " Lo scarto è il risultato, non un dettaglio da mediare: "
                    "riporta l'intervallo, non un numero solo."
                )

    with st.expander("Medie di step usate per il VT1"):
        st.dataframe(
            means.drop(columns=["stage_id"]).style.format(
                {c: "{:.2f}" for c in means.columns
                 if means[c].dtype.kind == "f"}, na_rep="—"),
            width="stretch", hide_index=True)
        st.download_button(
            "📥 Scarica le medie di step (CSV)",
            data=means.to_csv(index=False).encode("utf-8"),
            file_name="cpet_medie_step.csv", mime="text/csv",
            key="cpet_dl_means")


# ===========================================================================
# Stato della selezione
# ===========================================================================
# Ogni bin ha una sua st.checkbox con una key stabile: è il widget stesso a
# tenere il proprio valore in session_state, non c'è nessun dizionario da
# risincronizzare a ogni rerun e nessuna key che cambia sotto i piedi. Il
# valore `value=` viene letto SOLO alla prima comparsa del widget, quindi la
# preselezione è un punto di partenza e le spunte manuali sopravvivono a
# tutte le interazioni successive.
#
# Streamlit rieseguirà comunque lo script a ogni click — è il suo modello di
# esecuzione — ma con le checkbox non si ridisegna nessuna tabella editabile
# e la posizione nella pagina resta dov'era.
def _file_token(fname: str) -> str:
    """Token breve e stabile per file, usato nelle key dei widget."""
    return hashlib.md5(fname.encode("utf-8")).hexdigest()[:8]


def _use_key(fname: str, t_s) -> str:
    """La key DEVE contenere il file: due test diversi hanno gli stessi
    timestamp, e senza il token le checkbox del file A si portavano dietro lo
    stato del file B. La copia in `_keep` resta comunque necessaria, perché
    Streamlit scarta lo stato dei widget non disegnati nel rerun corrente."""
    return f"cpet_use_{_file_token(fname)}_{int(t_s)}"


def _keep(fname: str) -> dict:
    """Copia non-widget della selezione, per file.

    Serve perché Streamlit butta via lo stato di un widget che non viene
    istanziato in un rerun: cambiando file e tornando indietro, le checkbox
    del primo file sono sparite dallo schermo e con loro le spunte manuali.
    Il valore viene quindi rispecchiato qui a ogni giro e riletto come
    `value=` quando il widget rinasce. Le key delle checkbox non includono
    il nome del file proprio perché file diversi non coesistono mai nella
    stessa pagina: a distinguerli è questo dizionario."""
    st.session_state.setdefault("cpet_keep", {})
    return st.session_state["cpet_keep"].setdefault(fname, {})


def purge_selection(fname: str, df: pd.DataFrame):
    """Cancella selezione salvata e key dei widget: al giro successivo le
    checkbox ricompaiono con il valore di preselezione."""
    _keep(fname).clear()
    for t in df["t_s"]:
        st.session_state.pop(_use_key(fname, t), None)


def sync_preselection(fname: str, df: pd.DataFrame, signature) -> None:
    """Se cambiano i parametri che generano la proposta (taglio, variabili,
    soglie z), la preselezione va rifatta: si azzerano le key PRIMA che i
    widget vengano creati in questo stesso rerun."""
    st.session_state.setdefault("cpet_sig", {})
    if st.session_state["cpet_sig"].get(fname) != signature:
        purge_selection(fname, df)
        st.session_state["cpet_sig"][fname] = signature


def _short_time(t: str) -> str:
    """'00:12:30' -> '12:30': l'ora è sempre 0 in questi test e le colonne
    delle checkbox sono strette."""
    parts = str(t).split(":")
    return ":".join(parts[1:]) if len(parts) == 3 and parts[0] in ("0", "00") else str(t)


# ===========================================================================
# Render
# ===========================================================================
def render_cpet_analysis():
    st.header("🫁 CPET a step: steady state e pulizia dei bin")
    st.caption(
        "Metabolimetro a camera di miscelazione: ogni riga è già una media sui "
        "30 s di riempimento della camera. I bin vengono raggruppati per "
        "velocità e **preselezionati**; sotto ogni step una casella per bin, "
        "già spuntata dove serve. Deseleziona quello che non vuoi."
    )

    files = st.file_uploader(
        "📄 CSV del metabolimetro (uno o più test)",
        type=["csv"], accept_multiple_files=True, key="cpet_files",
    )
    if not files:
        st.info("Carica almeno un CSV esportato dal metabolimetro per iniziare.")
        return

    parsed = {}
    for f in files:
        meta, df = parse_cpet_csv(f.getvalue())
        if df is None or df.empty:
            st.warning(f"⚠️ {f.name}: nessun tracciato riconosciuto, file saltato.")
            continue
        parsed[f.name] = (meta, assign_stages(df))

    if not parsed:
        st.error("Nessun file leggibile.")
        return

    # ------------------------------------------------------------------
    # Comandi principali
    # ------------------------------------------------------------------
    t1, t2, t3 = st.columns([2, 1.2, 1.2])
    with t1:
        fname = st.selectbox("File", list(parsed.keys()), key="cpet_active_file")
    with t2:
        trim_s = st.select_slider("Scarta i primi … (s)", options=CPET_TRIM_OPTIONS,
                                  value=90, key="cpet_trim")
    with t3:
        summary_var = st.selectbox(
            "Variabile per lo steady state", CPET_SUMMARY_VARS, index=0,
            key="cpet_summary_var_v2",
            help="Su quale segnale si giudicano CV, deriva e verdetto di ogni "
                 "step. Il VO2 è il riferimento abituale.")

    show_cols = st.multiselect(
        "Colonne in tabella", CPET_ALL_COLS, default=CPET_DEFAULT_COLS,
        key="cpet_show_cols",
        help="Default = il minimo per leggere il VT1: V-slope (VO2, VCO2), "
             "equivalenti ventilatori (Ve/VO2, Ve/VCO2), FeO2 e FC. "
             "L'export contiene comunque tutto.")
    if not show_cols:
        show_cols = CPET_DEFAULT_COLS

    with st.expander("⚙️ Impostazioni avanzate (soglie di preselezione)"):
        st.caption(
            "Toccando queste manopole la preselezione viene rifatta da capo e "
            "le spunte manuali si azzerano: stai cambiando la proposta, non "
            "correggendola."
        )
        a1, a2 = st.columns(2)
        with a1:
            flag_vars = st.multiselect(
                "Variabili della regola combinata", CPET_FLAG_VARS,
                default=CPET_FLAG_DEFAULT, key="cpet_flag_vars_v2")
            z_hard = st.slider("z 'duro' (basta 1 variabile)", 2.5, 6.0, 3.5,
                               step=0.1, key="cpet_zhard")
            z_soft = st.slider("z 'morbido' (servono 2 variabili)", 1.5, 4.0,
                               2.5, step=0.1, key="cpet_zsoft")
        with a2:
            cv_max = st.slider("CV max per 'steady' (%)", 1.0, 12.0, 5.0,
                               step=0.5, key="cpet_cvmax")
            delta_max = st.slider("|Δ metà| max per 'steady' (%)", 1.0, 10.0,
                                  3.0, step=0.5, key="cpet_dmax")
            show_rest = st.checkbox("Mostra riposo e recupero", value=False,
                                    key="cpet_show_rest")
        st.caption(
            "z robusto = residuo dopo detrending interno allo step, diviso il "
            "MAD comune a tutto il test. Il detrending evita di scambiare una "
            "deriva vera per una fila di outlier; il MAD messo in comune tra "
            "gli step evita che dentro 5 bin collassi a zero e produca z da 15."
        )

    meta, df = parsed[fname]
    df = df.copy()

    who = " ".join(x for x in [meta.get("First Name", ""),
                               meta.get("Last Name", "")] if x).strip()
    bits = [b for b in [
        who or None,
        f"{meta.get('Weight (kg)')} kg" if meta.get("Weight (kg)") else None,
        meta.get("Test date (DDMMYYYY)"),
        f"durata {meta.get('Duration (hh:mm:ss)')}" if meta.get("Duration (hh:mm:ss)") else None,
        f"epoca {df.attrs.get('epoch_s', 30):.0f} s",
    ] if b]
    st.caption(" · ".join(bits))

    # ------------------------------------------------------------------
    # Preselezione
    # ------------------------------------------------------------------
    df["in_window"] = df["t_in_stage_s"] > trim_s
    df["rest_stage"] = [is_rest_stage(s, p)
                        for s, p in zip(df["stage_speed"], df["stage_phase"])]

    work = df[df["in_window"] & ~df["rest_stage"]]
    scales = {v: pooled_noise_scale(work, v) for v in flag_vars}

    df["flag"] = False
    df["zmax"] = np.nan
    df["motivo"] = ""
    for _, sub in df.groupby("stage_id"):
        cand = sub[sub["in_window"]]
        if len(cand) < 4 or not flag_vars:
            continue
        fl, zm, rs, zmat, gates = flag_stage(cand, flag_vars, z_hard,
                                             z_soft, scales)
        df.loc[cand.index, "flag"] = fl
        df.loc[cand.index, "zmax"] = zm
        df.loc[cand.index, "motivo"] = rs
        for v, z in zmat.items():
            df.loc[cand.index, f"z::{v}"] = z
        for v, bad in gates.items():
            df.loc[cand.index, f"gate::{v}"] = bad

    df["default_use"] = df["in_window"] & ~df["flag"] & ~df["rest_stage"]

    def _nota(r):
        if r["rest_stage"]:
            return "riposo / recupero"
        if not r["in_window"]:
            return f"transitorio (primi {trim_s} s)"
        if r["flag"]:
            return f"anomalia — {r['motivo']}" if r["motivo"] else "anomalia"
        return ""

    df["Nota"] = df.apply(_nota, axis=1)

    signature = (trim_s, tuple(sorted(flag_vars)), z_hard, z_soft)
    sync_preselection(fname, df, signature)

    if st.button("↺ Ripristina la preselezione di tutti gli step",
                 key="cpet_reset_all"):
        purge_selection(fname, df)
        st.rerun()

    # ------------------------------------------------------------------
    # Una sezione per step: checkbox + tabella dei bin
    # ------------------------------------------------------------------
    st.divider()
    plot_df = df if show_rest else df[~df["rest_stage"]]
    used_flags = {}

    for _, sub in plot_df.groupby("stage_id"):
        sub = sub.sort_values("t_s")
        label = sub["stage_label"].iloc[0]
        st.markdown(f"### {label}")

        # --- riga di caselle, una per bin, a blocchi di 10 ---
        st.caption(
            "⏳ = transitorio di inizio step · ⚑ = segnalata dalla regola "
            "combinata. Entrambe nascono deselezionate: rimettile dentro se "
            "secondo te il dato è buono."
        )
        times = list(sub["Time"])
        keys = [_use_key(fname, t) for t in sub["t_s"]]
        keep = _keep(fname)
        defaults = [bool(keep.get(int(t), d))
                    for t, d in zip(sub["t_s"], sub["default_use"])]
        marks = ["⏳" if not w else ("⚑" if f else "")
                 for w, f in zip(sub["in_window"], sub["flag"])]

        checked = []
        for start in range(0, len(sub), 10):
            chunk = range(start, min(start + 10, len(sub)))
            cols = st.columns(10)
            for col, i in zip(cols, chunk):
                checked.append(col.checkbox(
                    f"{marks[i]}{_short_time(times[i])}",
                    value=defaults[i], key=keys[i],
                ))

        for t, v in zip(sub["t_s"], checked):
            used_flags[int(t)] = bool(v)
            keep[int(t)] = bool(v)

        step_used = sub[np.array(checked, dtype=bool)]
        cln = stage_stats(step_used, summary_var)
        base = stage_stats(sub[sub["default_use"]], summary_var)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"{summary_var} medio",
                  f"{cln['Media']:.2f}" if np.isfinite(cln["Media"]) else "—",
                  help=f"su {cln['N']} bin di {len(sub)}")
        m2.metric("SD", f"{cln['SD']:.2f}" if np.isfinite(cln["SD"]) else "—")
        m3.metric("CV %", f"{cln['CV %']:.2f}" if np.isfinite(cln["CV %"]) else "—",
                  delta=(f"{cln['CV %'] - base['CV %']:+.2f} vs preselez."
                         if np.isfinite(cln["CV %"]) and np.isfinite(base["CV %"])
                         else None),
                  delta_color="inverse")
        m4.metric("Deriva /min",
                  f"{cln['Deriva (/min)']:+.2f}"
                  if np.isfinite(cln["Deriva (/min)"]) else "—")
        m5.metric("Steady?", steady_verdict(cln, cv_max, delta_max))

        # --- tabella dei bin dello step, righe escluse in grigio barrato ---
        cols_show = ["Time"] + [c for c in show_cols if c in sub.columns] + ["Nota"]
        tbl = sub[cols_show].rename(columns={"Time": "Tempo"})
        tbl = tbl.reset_index(drop=True)
        incl = list(checked)

        # Due canali visivi indipendenti, perché rispondono a due domande
        # diverse: lo SFONDO della cella dice quale numero è fuori scala, il
        # testo barrato dice se il bin entra nel calcolo. Tenerli separati fa
        # sì che un'anomalia che hai deciso di rimettere dentro resti
        # comunque rossa e visibile, invece di sparire nella massa dei bin
        # buoni.
        sub_r = sub.reset_index(drop=True)

        def _style(frame, incl=incl, sub_r=sub_r):
            out = pd.DataFrame("", index=frame.index, columns=frame.columns)
            for c in frame.columns:
                zc, gc = f"z::{c}", f"gate::{c}"
                if gc in sub_r.columns:
                    bad = sub_r[gc].fillna(False).to_numpy(dtype=bool)
                    out.loc[bad, c] = CELL_GATE
                if zc in sub_r.columns:
                    az = np.abs(pd.to_numeric(sub_r[zc], errors="coerce")
                                .fillna(0).to_numpy())
                    out.loc[(az >= z_hard) & (out[c] == ""), c] = CELL_HARD
                    out.loc[(az >= z_soft) & (az < z_hard)
                            & (out[c] == ""), c] = CELL_SOFT
            for i, keep in enumerate(incl):
                if not keep:
                    out.iloc[i] = out.iloc[i] + ROW_UNUSED
            return out

        styled = (tbl.style
                  .apply(_style, axis=None)
                  .format(cpet_col_formats(tbl.columns), na_rep="—"))
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption(
            "Sfondo <span style='background:rgba(193,68,14,.45);padding:0 4px'>"
            "pieno</span> = oltre z duro · <span style='background:"
            "rgba(193,68,14,.18);padding:0 4px'>tenue</span> = oltre z morbido "
            "· <span style='background:rgba(140,30,120,.45);padding:0 4px'>"
            "viola</span> = fuori dai limiti fisiologici · testo grigio = "
            "bin non usato. Colorate solo le variabili della regola combinata "
            "(" + ", ".join(flag_vars) + "): sui rapporti lo z non si calcola, "
            "perché derivano dalle stesse misure.",
            unsafe_allow_html=True,
        )

        if cln["N"] < 4:
            st.warning(
                f"⚠️ Solo {cln['N']} bin selezionati: SD e CV su così pochi "
                "punti non descrivono niente."
            )
        st.divider()

    df["used"] = [bool(used_flags.get(int(t), False)) for t in df["t_s"]]

    # ------------------------------------------------------------------
    # Riepilogo di tutti gli step
    # ------------------------------------------------------------------
    st.subheader("Riepilogo per step")
    rows = []
    for _, sub in plot_df.groupby("stage_id"):
        raw = stage_stats(sub, summary_var)
        base = stage_stats(sub[sub["default_use"]], summary_var)
        cln = stage_stats(sub[df.loc[sub.index, "used"]], summary_var)
        k = int(base["N"] - cln["N"])
        rows.append({
            "Step": sub["stage_label"].iloc[0],
            "Steady?": steady_verdict(cln, cv_max, delta_max),
            "Taglio sugg. (s)": suggest_trim(sub, summary_var, cv_max, delta_max),
            "N usati": cln["N"],
            "N totali": raw["N"],
            "Media": cln["Media"], "SD": cln["SD"], "CV %": cln["CV %"],
            "SEM": cln["SEM"], "Min": cln["Min"], "Max": cln["Max"],
            "Range": cln["Range"], "IQR": cln["IQR"],
            "Deriva (/min)": cln["Deriva (/min)"],
            "Δ metà (%)": cln["Δ metà (%)"],
            "CV % preselez.": base["CV %"],
            "CV % tutti i bin": raw["CV %"],
            "SD ratio reale": (cln["SD"] / base["SD"]
                               if np.isfinite(cln.get("SD", np.nan))
                               and np.isfinite(base.get("SD", np.nan))
                               and base["SD"] > 0 else np.nan),
            "SD ratio atteso": expected_sd_ratio(int(base["N"]), k),
        })

    stage_df = pd.DataFrame(rows)
    if stage_df.empty:
        st.info("Nessuno step da mostrare.")
        return

    n_dec = int(CPET_COL_DECIMALS.get(summary_var, 2))
    stage_fmt = {c: f"{{:.{n_dec}f}}" for c in
                 ("Media", "SD", "SEM", "Min", "Max", "Range", "IQR")}
    stage_fmt.update({
        "Deriva (/min)": "{:+.2f}", "Δ metà (%)": "{:+.2f}",
        "CV %": "{:.2f}", "CV % preselez.": "{:.2f}",
        "CV % tutti i bin": "{:.2f}",
        "SD ratio reale": "{:.2f}", "SD ratio atteso": "{:.2f}",
        "Taglio sugg. (s)": "{:.0f}",
    })
    def _style_summary(frame):
        out = pd.DataFrame("", index=frame.index, columns=frame.columns)
        if "CV %" in frame.columns:
            cv = pd.to_numeric(frame["CV %"], errors="coerce")
            out.loc[cv > cv_max * 1.5, "CV %"] = CELL_HARD
            out.loc[(cv > cv_max) & (cv <= cv_max * 1.5), "CV %"] = CELL_SOFT
        if "Δ metà (%)" in frame.columns:
            dm = pd.to_numeric(frame["Δ metà (%)"], errors="coerce").abs()
            out.loc[dm > delta_max * 1.5, "Δ metà (%)"] = CELL_HARD
            out.loc[(dm > delta_max) & (dm <= delta_max * 1.5),
                    "Δ metà (%)"] = CELL_SOFT
        if "N usati" in frame.columns:
            out.loc[pd.to_numeric(frame["N usati"], errors="coerce") < 4,
                    "N usati"] = CELL_GATE
        # Il confronto reale/atteso è il controllo anti-illusione: si accende
        # quando la SD è scesa NON più di quanto sarebbe scesa per puro caso,
        # cioè quando la pulizia non ha guadagnato nulla.
        if {"SD ratio reale", "SD ratio atteso"} <= set(frame.columns):
            a = pd.to_numeric(frame["SD ratio reale"], errors="coerce")
            e = pd.to_numeric(frame["SD ratio atteso"], errors="coerce")
            out.loc[a.notna() & e.notna() & (a > e), "SD ratio reale"] = CELL_SOFT
        return out

    st.dataframe(stage_df.style
                 .apply(_style_summary, axis=None)
                 .format(stage_fmt, na_rep="—"),
                 width="stretch", hide_index=True)
    st.caption(
        "**SD ratio reale vs atteso** è il controllo anti-illusione. Togliere i "
        "punti più estremi abbassa la SD anche su rumore puro: da 8 bin, "
        "toglierne 2 la porta in media al 63% pur non essendoci alcun outlier. "
        "La colonna *atteso* è quel calo dovuto al solo caso, rispetto alla "
        "preselezione. Se il *reale* è molto più basso, hai tolto anomalie "
        "vere; se gli somiglia, hai solo tagliato le code."
    )

    # ------------------------------------------------------------------
    # VT1
    # ------------------------------------------------------------------
    st.divider()
    render_vt1_section(plot_df.assign(used=df.loc[plot_df.index, "used"]))

    # ------------------------------------------------------------------
    # Grafico
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Grafico dei bin")
    used = plot_df[df.loc[plot_df.index, "used"]]
    unused = plot_df[~df.loc[plot_df.index, "used"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=unused["t_s"] / 60, y=unused[summary_var], mode="markers",
        name="Non usati", customdata=unused["Nota"],
        marker=dict(size=8, color="rgba(160,160,160,0.6)",
                    symbol="circle-open", line=dict(width=1.5)),
        hovertemplate="%{y:.1f}<br>%{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=used["t_s"] / 60, y=used[summary_var], mode="markers",
        name="Usati", marker=dict(size=9, color="#2a9d8f"),
        hovertemplate="%{y:.1f}<extra></extra>",
    ))
    for _, sub in plot_df.groupby("stage_id"):
        u = sub[df.loc[sub.index, "used"]]
        if len(u) < 2:
            continue
        m = float(pd.to_numeric(u[summary_var], errors="coerce").mean())
        if np.isfinite(m):
            fig.add_shape(type="line", x0=sub["t_s"].min() / 60,
                          x1=sub["t_s"].max() / 60, y0=m, y1=m,
                          line=dict(color="#1f4e79", width=2))
    for b in plot_df.groupby("stage_id")["t_s"].min():
        fig.add_vline(x=(b - df.attrs.get("epoch_s", 30)) / 60,
                      line_width=1, line_dash="dot", line_color="#cccccc")

    fig.update_layout(
        title=f"{summary_var} per bin — {fname}",
        xaxis_title="Tempo (min)", yaxis_title=summary_var,
        height=440, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Linea blu = media dei bin usati nello step. "
               "Passa sopra un punto grigio per sapere perché è fuori.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.subheader("Export")
    e1, e2 = st.columns(2)
    out_bins = df[[
        "Time", "t_s", "stage_label", "stage_speed", "stage_phase",
        "t_in_stage_s", "in_window", "flag", "zmax", "used", "Nota"]
        + [c for c in CPET_ALL_COLS if c in df.columns]]
    e1.download_button(
        "📥 Bin (tutti, con la selezione)",
        data=out_bins.to_csv(index=False).encode("utf-8"),
        file_name=f"cpet_bin_{fname}", mime="text/csv", key="cpet_dl_bins")
    e2.download_button(
        "📥 Statistiche per step",
        data=stage_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cpet_step_{fname}", mime="text/csv", key="cpet_dl_stages")
    st.caption(
        "L'export dei bin contiene anche quelli scartati, con la colonna *used* "
        "e il motivo: la pulizia resta ricostruibile da chi rilegge il dato."
    )