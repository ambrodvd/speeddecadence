"""Calcolo dell'Efficiency Factor a blocchi, condiviso dalle tab Efficiency
Factor ed EF a blocchi.

Race Analyzer (metodo vecchio, d/t) = copia fedele di det_graphs.compute_segment_efs
+ logica dei blocchi del DU COACHING RACE Analyzer:
  - distanza del dispositivo (campo `distance` del .fit), forzata monotona;
  - per ogni blocco: quota smussata (rolling 9 punti) sul solo blocco,
    ricampionamento a 20 m, EFD Minetti, EFS = EFD ÷ tempo del blocco;
  - FC = media dei record del blocco (almeno 10);
  - blocchi consecutivi; l'ultimo, tronco, si tiene solo se copre almeno
    metà blocco.
  Ogni secondo pesa uguale, soste incluse.

moto (metodo attuale del Race Analyzer) = stessa distanza, stessa EFD,
stessi blocchi: cambia solo il trattamento delle soste. Si tengono i soli
secondi in movimento:
  - moto deciso secondo per secondo dallo spostamento in linea d'aria
    (lat/lon) su MOVING_WIN_S secondi contro una soglia (default 0.5 km/h);
  - EFS = EFD dei secondi in moto ÷ secondi in moto (l'EFD si accumula con
    la distanza, non col tempo, quindi da fermo non si perde nulla);
  - FC = media sugli stessi secondi in moto.
Senza soste i due metodi coincidono (a meno della registrazione "smart",
dove la media dei record del metodo vecchio non è pesata sul tempo).

Smoothing e passo di ricampionamento sono quelli fissi del Race Analyzer,
non quelli della sidebar: lo scopo è riprodurne i numeri.
"""
import numpy as np
import pandas as pd

from fit_core import FLAT_COST, cost_of_running, haversine_vec

EF_BLOCK_MIN_DEFAULT = 15
EF_BLOCK_CHOICES = [5, 10, 15, 20, 30, 45, 60]   # come lo slider del Race Analyzer
RA_RESAMPLE_STEP_M = 20.0     # EFS_RESAMPLE_STEP_M del Race Analyzer
RA_SMOOTH_WINDOW = 9          # EFS_SMOOTH_WINDOW del Race Analyzer
RA_MIN_BLOCK_FRAC = 0.5       # EF_MIN_BLOCK_FRAC del Race Analyzer
RA_MIN_HR_RECORDS = 10
RA_MIN_BLOCKS = 4             # sotto, il Race Analyzer non calcola l'EF

MOVING_WIN_S = 20             # finestra (s) dello spostamento netto per decidere il moto
MOVING_MIN_KMH_DEFAULT = 0.5  # default del Race Analyzer: sotto = fermo
MIN_MOVING_S = 60             # sotto questo tempo in moto il blocco non è valutabile

_TABLE_COLS = ["Blocco", "Durata (min)", "D+ (m)", "Distanza (km)", "EFD (km)",
               "Tempo fermo (s)",
               "Velocità RA (km/h)", "Velocità moto (km/h)",
               "EFS RA (km/h)", "EFS moto (km/h)",
               "FC media RA (bpm)", "FC media moto (bpm)",
               "% soglia RA", "% soglia moto",
               "EF Race Analyzer", "EF moto", "Δ EF (moto − RA)",
               "_mid_h"]


def read_hr_zones_csv(csv_file):
    """Legge il CSV zone FC esportato dal Race Analyzer.
    Ritorna (dict_zone, None) oppure (None, messaggio_errore).
    Serve almeno z4 (soglia): è l'unico valore usato dall'EF."""
    try:
        zdf = pd.read_csv(csv_file)
    except Exception as e:
        return None, f"CSV illeggibile ({e.__class__.__name__})"
    if zdf.empty:
        return None, "CSV vuoto"

    cols = {str(c).strip().lower(): c for c in zdf.columns}
    if "z4" not in cols:
        return None, "manca la colonna 'z4' (FC di soglia)"

    thr = pd.to_numeric(zdf[cols["z4"]].iloc[0], errors="coerce")
    if not np.isfinite(thr) or thr <= 0:
        return None, "valore di 'z4' non valido"

    zones = {"threshold": float(thr)}
    for z in ["z1", "z2", "z3", "z4", "z5"]:
        if z in cols:
            v = pd.to_numeric(zdf[cols[z]].iloc[0], errors="coerce")
            zones[z] = float(v) if np.isfinite(v) else np.nan
    if "athlete_name" in cols:
        zones["athlete"] = str(zdf[cols["athlete_name"]].iloc[0])
    return zones, None


def _device_distance(raw):
    """Distanza cumulata (m) come la usa il Race Analyzer: quella del
    dispositivo, forzata monotona. Se il file non la ha, ripiega sulla
    distanza calcolata da lat/lon. Ritorna (array, fonte)."""
    if "dist_dev_m" in raw.columns and raw["dist_dev_m"].notna().sum() > 1:
        d = raw["dist_dev_m"].ffill().fillna(0.0).to_numpy(dtype=float)
        return np.maximum.accumulate(d), "dispositivo"
    lat, lon = raw["lat"].to_numpy(), raw["lon"].to_numpy()
    d = np.concatenate([[0.0], np.cumsum(haversine_vec(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    return d, "GPS (lat/lon)"


def _ra_block_profile(dist, t, ele):
    """Replica di compute_segment_efs su un blocco. Oltre all'EFS ritorna la
    griglia per distanza e l'EFD cumulata su di essa, che servono al metodo
    moto per assegnare l'EFD ai singoli secondi. None se il tratto è degenere."""
    if len(dist) < 3:
        return None
    d = np.maximum.accumulate(dist)
    d = d - d[0]
    e = (pd.Series(ele).ffill().bfill().fillna(0.0)
           .rolling(window=RA_SMOOTH_WINDOW, center=True, min_periods=1)
           .mean().to_numpy())
    total = float(d[-1])
    if total <= 0:
        return None

    n = max(2, int(total // RA_RESAMPLE_STEP_M))
    grid = np.linspace(0.0, total, n)
    ele_g = np.interp(grid, d, e)
    t_g = np.interp(grid, d, t)
    dx, dz, dt = np.diff(grid), np.diff(ele_g), np.diff(t_g)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)
    efd_m = cost_of_running(slope) * np.hypot(dx, dz) / FLAT_COST

    total_t = float(dt.sum())
    if total_t <= 0:
        return None
    return {
        "efs_kmh": float(efd_m.sum() / total_t * 3.6),
        "speed_kmh": total / total_t * 3.6,
        "dist_m": total,
        "efd_m": float(efd_m.sum()),
        "gain_m": float(dz[dz > 0].sum()),
        "d_rel": d,                                   # distanza del blocco ai record
        "grid": grid,
        "cum_efd": np.concatenate([[0.0], np.cumsum(efd_m)]),
    }


def _moving_mask(raw, moving_min_kmh):
    """moving[k] per il secondo k = intervallo (k-1, k], k = 1..N.
    Spostamento IN LINEA D'ARIA su MOVING_WIN_S secondi, non somma dei passi:
    da fermo il GPS oscilla attorno allo stesso punto e sommando i passi quel
    rumore diventerebbe 5-8 km/h fittizi; lo spostamento netto si annulla."""
    t_raw = raw["elapsed_s"].to_numpy(dtype=float)
    edges = np.arange(0.0, np.floor(t_raw[-1]) + 1.0)
    lat_s = np.interp(edges, t_raw, raw["lat"].to_numpy())
    lon_s = np.interp(edges, t_raw, raw["lon"].to_numpy())
    n = len(edges)
    half = MOVING_WIN_S // 2
    k = np.arange(1, n)
    i0 = np.clip(k - half, 0, n - 1)
    i1 = np.clip(k + half, 0, n - 1)
    span = (i1 - i0).astype(float)
    disp = haversine_vec(lat_s[i0], lon_s[i0], lat_s[i1], lon_s[i1])
    spd_kmh = np.divide(disp, span, out=np.zeros_like(disp), where=span > 0) * 3.6
    moving = np.concatenate([[False], spd_kmh >= moving_min_kmh])  # indice = k
    return moving


def compute_block_table(raw, hr_threshold, moving_min_kmh=MOVING_MIN_KMH_DEFAULT,
                        block_min=EF_BLOCK_MIN_DEFAULT):
    """Una riga per blocco. Ritorna (tabella, fonte distanza) oppure
    (None, motivo)."""
    if raw is None or raw.empty or len(raw) < 3:
        return None, "traccia vuota"
    if not hr_threshold or hr_threshold <= 0 or "hr" not in raw.columns:
        return None, "soglia o FC mancanti"
    hr_ok = raw.dropna(subset=["hr", "elapsed_s"])
    if len(hr_ok) < RA_MIN_HR_RECORDS:
        return None, "FC insufficiente"

    t_all = raw["elapsed_s"].to_numpy(dtype=float)
    ele_all = raw["ele"].to_numpy(dtype=float)
    hr_all = raw["hr"].to_numpy(dtype=float)
    dist_all, dist_src = _device_distance(raw)
    hr_t = hr_ok["elapsed_s"].to_numpy(dtype=float)
    hr_v = hr_ok["hr"].to_numpy(dtype=float)
    moving = _moving_mask(raw, moving_min_kmh)

    race_end = float(t_all[-1])
    blk = block_min * 60.0
    n_blocks = int(np.ceil(race_end / blk))

    def ef_of(efs, hr):
        rel = hr / hr_threshold if np.isfinite(hr) else np.nan
        ef = efs / rel if np.isfinite(efs) and np.isfinite(rel) and rel > 0 else np.nan
        return rel, ef

    rows = []
    for b in range(n_blocks):
        t0, t1 = b * blk, (b + 1) * blk
        t1_eff = min(t1, race_end)
        if (t1_eff - t0) / blk < RA_MIN_BLOCK_FRAC:
            continue

        sel = (t_all >= t0) & (t_all < t1)
        hr_blk = hr_all[sel]
        hr_blk = hr_blk[np.isfinite(hr_blk)]
        prof = _ra_block_profile(dist_all[sel], t_all[sel], ele_all[sel])
        if prof is None or len(hr_blk) < RA_MIN_HR_RECORDS:
            continue

        # --- Race Analyzer ---
        hr_ra = float(hr_blk.mean())
        rel_ra, ef_ra = ef_of(prof["efs_kmh"], hr_ra)

        # --- moto: stessi secondi del blocco, solo quelli in movimento ---
        k_lo, k_hi = int(t0) + 1, int(np.floor(t1_eff))
        ks = np.arange(k_lo, k_hi + 1)
        ks = ks[ks < len(moving)]
        edges = np.concatenate([[k_lo - 1.0], ks.astype(float)])
        # EFD per secondo: distanza percorsa nel secondo → EFD cumulata per
        # distanza. Da fermo la distanza non cresce, quindi l'EFD nemmeno.
        d_edges = np.interp(edges, t_all[sel], prof["d_rel"])
        efd_s = np.diff(np.interp(d_edges, prof["grid"], prof["cum_efd"]))
        dist_s = np.diff(d_edges)
        mov = moving[ks]
        moving_s = int(mov.sum())
        if moving_s >= MIN_MOVING_S:
            efs_mov = float(efd_s[mov].sum()) / moving_s * 3.6
            spd_mov = float(dist_s[mov].sum()) / moving_s * 3.6
            hr_mov = float(np.mean(np.interp(ks[mov] - 0.5, hr_t, hr_v)))
        else:
            efs_mov = spd_mov = hr_mov = np.nan
        rel_mov, ef_mov = ef_of(efs_mov, hr_mov)

        m0, m1 = int(t0 // 60), int(round(t1_eff / 60))
        rows.append({
            "Blocco": f"{m0}'-{m1}'",
            "Durata (min)": (t1_eff - t0) / 60.0,
            "D+ (m)": prof["gain_m"],
            "Distanza (km)": prof["dist_m"] / 1000.0,
            "EFD (km)": prof["efd_m"] / 1000.0,
            "Tempo fermo (s)": len(ks) - moving_s,
            "Velocità RA (km/h)": prof["speed_kmh"],
            "Velocità moto (km/h)": spd_mov,
            "EFS RA (km/h)": prof["efs_kmh"],
            "EFS moto (km/h)": efs_mov,
            "FC media RA (bpm)": hr_ra,
            "FC media moto (bpm)": hr_mov,
            "% soglia RA": rel_ra * 100.0,
            "% soglia moto": rel_mov * 100.0 if np.isfinite(rel_mov) else np.nan,
            "EF Race Analyzer": ef_ra,
            "EF moto": ef_mov,
            "Δ EF (moto − RA)": ef_mov - ef_ra,
            "_mid_h": (t0 + t1_eff) / 2.0 / 3600.0,
        })

    if not rows:
        return None, f"nessun blocco valido (file più corto di {block_min / 2:g} min o senza dati)"
    return pd.DataFrame(rows).reindex(columns=_TABLE_COLS), dist_src