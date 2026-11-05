"""
Tab "Salita vs piano": per ogni file .fit/.gz individua i tratti in salita
(pendenza > +2%) e in piano (tra -2% e +2%) lunghi almeno N km, li mostra sul
profilo per conferma e poi riporta velocità media, EFS media e pendenza media
per tratto e in totale.

Riceve i record grezzi già parsati dall'app principale (load_raw, cached):
colonne lat, lon, ele, elapsed_s.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

FLAT_SLOPE_PCT = 2.0      # piano = pendenza in [-2%, +2%]
SPEED_MIN_KMH = 0.5       # sotto = soste/ristori, esclusi da tempo e velocità
SPEED_MAX_KMH = 30.0      # sopra = glitch GPS

UPHILL, FLAT, DOWNHILL = 1, 0, -1
KIND_NAME = {UPHILL: "Salita", FLAT: "Piano"}
KIND_COLOR = {UPHILL: "#e07b39", FLAT: "#2a9d8f"}


# ---------------------------------------------------------------------------
# Minetti et al. 2002 — copia della funzione dell'app principale.
# Se la modifichi là, aggiornala anche qui.
# ---------------------------------------------------------------------------
def cost_of_running(slope: np.ndarray) -> np.ndarray:
    i = np.clip(slope, -0.45, 0.45)
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)


FLAT_COST = float(cost_of_running(np.array([0.0]))[0])  # 3.6 J/kg/m


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _seconds_to_hhmm(seconds) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Griglia di distanza e classificazione
# ---------------------------------------------------------------------------
def _track_grid(raw, step_m, smooth_m):
    """Quota e tempo su griglia a passo costante di distanza orizzontale.
    Lo smoothing è in metri (come nel climb detector), quindi non dipende
    dalla frequenza di registrazione né dalla velocità."""
    lat = raw["lat"].to_numpy(float)
    lon = raw["lon"].to_numpy(float)
    ele = raw["ele"].to_numpy(float)
    t = raw["elapsed_s"].to_numpy(float)
    if len(lat) < 3:
        return None

    cum = np.concatenate([[0.0], np.cumsum(_haversine(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    total = float(cum[-1])
    if total < 3 * step_m:
        return None

    n = int(total // step_m) + 1
    grid = np.linspace(0.0, total, n)
    ele_g = np.interp(grid, cum, ele)
    t_g = np.interp(grid, cum, t)

    w = max(1, int(round(smooth_m / (grid[1] - grid[0]))))
    if w % 2 == 0:
        w += 1
    ele_s = pd.Series(ele_g).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return grid, ele_s, t_g


def _runs(labels):
    """Run-length encoding: [(label, i_inizio, i_fine_esclusa), ...]."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _absorb_short_runs(labels, dx, tol_m):
    """Un tratto più corto di tol_m con lo stesso tipo da entrambi i lati
    viene assorbito: 100 m di falsopiano in mezzo a una salita sono parte
    della salita, non la spezzano in due tratti sotto la lunghezza minima.
    Si parte dai più corti; in ogni passata si toccano solo tratti non
    adiacenti, poi si ricalcola."""
    if tol_m <= 0:
        return labels
    labels = labels.copy()
    cum_dx = np.concatenate([[0.0], np.cumsum(dx)])
    while True:
        runs = _runs(labels)
        lengths = [cum_dx[b] - cum_dx[a] for _, a, b in runs]
        short = sorted((k for k in range(1, len(runs) - 1) if lengths[k] < tol_m),
                       key=lambda k: lengths[k])
        touched, changed = set(), False
        for k in short:
            if touched & {k - 1, k, k + 1}:
                continue
            lab, a, b = runs[k]
            prev_lab, next_lab = runs[k - 1][0], runs[k + 1][0]
            if prev_lab == next_lab and prev_lab != lab:
                labels[a:b] = prev_lab
                touched.update({k - 1, k, k + 1})
                changed = True
        if not changed:
            return labels


def analyze_track(raw, step_m, smooth_m, tol_m, min_len_m):
    """Ritorna {'grid', 'ele', 'segments'} oppure None se la traccia è degenere.
    Pendenza e D+ di un tratto usano tutta la geometria; velocità ed EFS
    escludono i passi fermi (< SPEED_MIN_KMH) o glitch (> SPEED_MAX_KMH),
    altrimenti un ristoro di 10 minuti abbatte la media del tratto."""
    g = _track_grid(raw, step_m, smooth_m)
    if g is None:
        return None
    grid, ele, t = g

    dx, dz, dt = np.diff(grid), np.diff(ele), np.diff(t)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)
    slope_pct = slope * 100.0

    labels = np.where(slope_pct > FLAT_SLOPE_PCT, UPHILL,
                      np.where(slope_pct >= -FLAT_SLOPE_PCT, FLAT, DOWNHILL))
    labels = _absorb_short_runs(labels, dx, tol_m)

    speed = np.divide(dx, dt, out=np.full_like(dx, np.nan), where=dt > 0) * 3.6
    valid = np.isfinite(speed) & (speed >= SPEED_MIN_KMH) & (speed <= SPEED_MAX_KMH)
    efd = cost_of_running(slope) * np.hypot(dx, dz) / FLAT_COST

    segments = []
    for lab, a, b in _runs(labels):
        if lab == DOWNHILL:
            continue
        length = float(dx[a:b].sum())
        if length < min_len_m:
            continue
        rise = float(dz[a:b].sum())
        avg_slope = rise / length * 100.0
        # il tipo finale segue la pendenza media del tratto: dopo
        # l'assorbimento un "piano" può finire a +2.1%
        if avg_slope > FLAT_SLOPE_PCT:
            kind = UPHILL
        elif avg_slope >= -FLAT_SLOPE_PCT:
            kind = FLAT
        else:
            continue

        v = valid[a:b]
        dist_v = float(dx[a:b][v].sum())
        time_v = float(dt[a:b][v].sum())
        efd_v = float(efd[a:b][v].sum())
        if dist_v <= 0 or time_v <= 0:
            continue

        seg_dz = dz[a:b]
        segments.append({
            "kind": kind,
            "i_start": a,
            "i_end": b,
            "km_start": grid[a] / 1000.0,
            "km_end": grid[b] / 1000.0,
            "len_m": length,
            "rise_m": rise,
            "d_plus_m": float(seg_dz[seg_dz > 0].sum()),
            "slope_pct": avg_slope,
            "dist_m": dist_v,
            "time_s": time_v,
            "efd_m": efd_v,
        })

    return {"grid": grid, "ele": ele, "segments": segments}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def build_classification_figure(res, title):
    x = res["grid"] / 1000.0
    ele = res["ele"]
    lo, hi = float(np.nanmin(ele)), float(np.nanmax(ele))
    span = max(hi - lo, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=ele, mode="lines", name="Escluso (discesa / corto)",
        line=dict(color="rgba(140,140,140,0.85)", width=1.2),
        hovertemplate="km %{x:.2f} — %{y:.0f} m<extra></extra>",
    ))

    counters = {UPHILL: 0, FLAT: 0}
    for s in res["segments"]:
        k = s["kind"]
        counters[k] += 1
        name = f"{KIND_NAME[k]} {counters[k]}"
        col = KIND_COLOR[k]
        a, b = s["i_start"], s["i_end"]
        fig.add_trace(go.Scatter(
            x=x[a:b + 1], y=ele[a:b + 1], mode="lines",
            name=KIND_NAME[k], legendgroup=KIND_NAME[k],
            showlegend=counters[k] == 1,
            line=dict(color=col, width=2.6),
            fill="tozeroy", fillcolor=_hex_rgba(col, 0.25),
            hovertemplate=(f"<b>{name}</b> · {s['len_m'] / 1000:.2f} km · "
                           f"{s['slope_pct']:+.1f}%<br>km %{{x:.2f}} — %{{y:.0f}} m"
                           "<extra></extra>"),
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Distanza (km)"),
        yaxis=dict(title="Quota (m)", tickformat="d",
                   range=[lo - 0.06 * span, hi + 0.12 * span]),
        hovermode="closest", height=380, margin=dict(t=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def _segments_table(athlete, fname, segments):
    rows, counters = [], {UPHILL: 0, FLAT: 0}
    for s in segments:
        counters[s["kind"]] += 1
        rows.append({
            "Atleta": athlete,
            "File": fname,
            "Tipo": KIND_NAME[s["kind"]],
            "Tratto": f"{KIND_NAME[s['kind']]} {counters[s['kind']]}",
            "Km inizio": s["km_start"],
            "Km fine": s["km_end"],
            "Lunghezza (km)": s["len_m"] / 1000.0,
            "D+ (m)": s["d_plus_m"],
            "Pendenza media (%)": s["slope_pct"],
            "Tempo": _seconds_to_hhmm(s["time_s"]),
            "Velocità media (km/h)": s["dist_m"] / s["time_s"] * 3.6,
            "EFS media (km/h)": s["efd_m"] / s["time_s"] * 3.6,
            "_len_m": s["len_m"], "_rise_m": s["rise_m"],
            "_dist_m": s["dist_m"], "_time_s": s["time_s"], "_efd_m": s["efd_m"],
        })
    return pd.DataFrame(rows)


def _agg(sub, prefix):
    """Medie pesate di un gruppo di tratti: velocità = distanza/tempo,
    EFS = EFD/tempo, pendenza = dislivello netto/lunghezza. Non medie delle
    medie dei tratti."""
    cols = ["N tratti", "km", "Pendenza (%)", "Velocità (km/h)", "EFS (km/h)"]
    if sub.empty:
        return {f"{prefix} {c}": np.nan for c in cols}
    len_m, time_s = sub["_len_m"].sum(), sub["_time_s"].sum()
    return {
        f"{prefix} N tratti": len(sub),
        f"{prefix} km": len_m / 1000.0,
        f"{prefix} Pendenza (%)": sub["_rise_m"].sum() / len_m * 100.0,
        f"{prefix} Velocità (km/h)": sub["_dist_m"].sum() / time_s * 3.6,
        f"{prefix} EFS (km/h)": sub["_efd_m"].sum() / time_s * 3.6,
    }


def _athlete_summary(seg_df):
    """Una riga per atleta (più una riga complessiva), salita e piano affiancati."""
    rows = []
    groups = [(a, seg_df[seg_df["Atleta"] == a]) for a in seg_df["Atleta"].unique()]
    if len(groups) > 1:
        groups.append(("TUTTI GLI ATLETI", seg_df))
    for name, sub in groups:
        row = {"Atleta": name, "N file": sub["File"].nunique()}
        for kind in (UPHILL, FLAT):
            row.update(_agg(sub[sub["Tipo"] == KIND_NAME[kind]], KIND_NAME[kind]))
        rows.append(row)
    return pd.DataFrame(rows)


def _public(df):
    return df[[c for c in df.columns if not c.startswith("_")]].round(2)


def render_uphill_flat(raw_by_file, resample_step_m):
    st.header("⛰️ Salita vs piano")
    st.caption(
        f"Piano = pendenza tra −{FLAT_SLOPE_PCT:g}% e +{FLAT_SLOPE_PCT:g}%, salita = oltre "
        f"+{FLAT_SLOPE_PCT:g}%. Le discese sono escluse. Velocità ed EFS sono "
        f"calcolate senza i passi fermi (< {SPEED_MIN_KMH:g} km/h) o glitch "
        f"(> {SPEED_MAX_KMH:g} km/h); pendenza e lunghezza usano tutto il tratto."
    )

    if not raw_by_file:
        st.info("Carica uno o più file .fit / .gz dalla sidebar.")
        return

    c1, c2, c3 = st.columns(3)
    min_len_km = c1.slider("Lunghezza minima tratto (km)", 0.2, 5.0, 1.0, 0.1,
                           key="uf_min_len_km")
    smooth_m = c2.slider("Smoothing quota (m)", 20, 500, 150, 10, key="uf_smooth_m",
                         help="Più alto = ignora le micro-ondulazioni della quota.")
    tol_m = c3.slider("Tolleranza interruzioni (m)", 0, 500, 200, 20, key="uf_tol_m",
                      help="Un tratto di tipo diverso più corto di così, in mezzo a due "
                           "tratti dello stesso tipo, viene assorbito invece di spezzarli.")

    results = {}
    for fname, raw in raw_by_file.items():
        try:
            res = analyze_track(raw, resample_step_m, smooth_m, tol_m, min_len_km * 1000.0)
        except Exception as e:
            st.warning(f"⚠️ {fname}: errore di elaborazione ({e.__class__.__name__}), escluso.")
            continue
        if res is None:
            st.warning(f"⚠️ {fname}: traccia troppo corta o senza GPS, esclusa.")
            continue
        results[fname] = res

    if not results:
        st.info("Nessun file elaborabile.")
        return

    # --- 1) classificazione a schermo + nome atleta per file ---
    # Il nome non entra nella firma della conferma: correggerlo non cambia
    # la classificazione, quindi non deve far sparire le tabelle.
    st.subheader("1. Classificazione salita / piano")
    athletes = {}
    for fname, res in results.items():
        st.markdown(f"#### {fname}")
        athletes[fname] = st.text_input(
            f"Nome atleta — {fname}", key=f"uf_athlete_{fname}",
            placeholder="Nome atleta (vuoto = nome file)",
        ).strip() or fname
        st.plotly_chart(build_classification_figure(res, athletes[fname]),
                        use_container_width=True, key=f"uf_fig_{fname}")
        segs = res["segments"]
        n_up = sum(s["kind"] == UPHILL for s in segs)
        n_fl = sum(s["kind"] == FLAT for s in segs)
        km_up = sum(s["len_m"] for s in segs if s["kind"] == UPHILL) / 1000.0
        km_fl = sum(s["len_m"] for s in segs if s["kind"] == FLAT) / 1000.0
        km_tot = res["grid"][-1] / 1000.0
        st.caption(
            f"🟠 **{n_up} salite** ({km_up:.1f} km) · 🟢 **{n_fl} tratti in piano** "
            f"({km_fl:.1f} km) · escluso {max(km_tot - km_up - km_fl, 0):.1f} km "
            f"su {km_tot:.1f} km totali"
        )

    # La conferma decade se cambiano file o parametri: altrimenti si
    # leggerebbero tabelle di una classificazione mai vista.
    sig = (tuple(sorted((f, len(r)) for f, r in raw_by_file.items())),
           resample_step_m, smooth_m, tol_m, min_len_km)
    if st.session_state.get("_uf_sig") != sig:
        st.session_state["_uf_sig"] = sig
        st.session_state["uf_confirmed"] = False

    # Un bottone è True solo nel rerun del click: lo stato confermato vive
    # in session_state, così le tabelle restano quando tocchi altri widget.
    st.divider()
    if not st.session_state.get("uf_confirmed"):
        st.info("Controlla i profili qui sopra, regola i parametri se serve, poi conferma.")
        if st.button("✅ Conferma classificazione e calcola", type="primary",
                     key="uf_confirm_btn"):
            st.session_state["uf_confirmed"] = True
            st.rerun()
        return

    st.success("Classificazione confermata. Cambiando file o parametri va riconfermata.")

    seg_df = pd.concat(
        [_segments_table(athletes[f], f, r["segments"]) for f, r in results.items()],
        ignore_index=True,
    )
    if seg_df.empty:
        st.warning("Nessun tratto di salita o piano sopra la lunghezza minima.")
        return

    # --- 2) tratti per file ---
    st.subheader("2. Tratti per file")
    for fname in results:
        sub = seg_df[seg_df["File"] == fname]
        st.markdown(f"**{athletes[fname]}** · {fname}")
        if sub.empty:
            st.caption("— nessun tratto sopra la lunghezza minima")
            continue
        st.dataframe(_public(sub), use_container_width=True, hide_index=True)

    # --- 3) riepilogo per atleta ---
    st.subheader("3. Riepilogo per atleta")
    totals_df = _athlete_summary(seg_df)
    st.dataframe(totals_df.round(2), use_container_width=True, hide_index=True)
    st.caption(
        "Medie pesate sui tratti: velocità = distanza ÷ tempo, EFS = EFD ÷ tempo, "
        "pendenza = dislivello netto ÷ lunghezza. I tratti lunghi pesano di più. "
        "Più file con lo stesso nome atleta vengono aggregati in un'unica riga."
    )

    d1, d2 = st.columns(2)
    d1.download_button(
        "📥 Scarica tratti (CSV)",
        data=_public(seg_df).to_csv(index=False).encode("utf-8"),
        file_name="salita_piano_tratti.csv", mime="text/csv",
    )
    d2.download_button(
        "📥 Scarica riepilogo atleti (CSV)",
        data=totals_df.round(3).to_csv(index=False).encode("utf-8"),
        file_name="salita_piano_riepilogo_atleti.csv", mime="text/csv",
    )