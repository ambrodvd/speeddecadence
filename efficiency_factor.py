"""ANALISI — Efficiency Factor per file su finestra mobile, con statistiche
aggregate e report HTML."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fit_core import seconds_to_hhmm

# --- Efficiency Factor (stessi parametri del DU COACHING RACE Analyzer) ---
EF_WIN_MIN = 60      # finestra mobile (minuti)
EF_STEP_MIN = 15     # passo tra finestre (minuti)
EF_MIN_KMH = 0.5     # sotto = soste/ristori, non velocità reale
EF_MAX_KMH = 30.0    # sopra = glitch GPS/quota


# ===========================================================
# ANALISI 4 — Efficiency Factor per file
# ===========================================================
# EF = EFS / (FC / soglia). La FC viene normalizzata sulla soglia (z4) e non
# lasciata in bpm: la FC di soglia è in larga parte genetica e non dice nulla
# su velocità o fitness, mentre dividendo per z4 due atleti alla stessa
# velocità relativa alla propria soglia ottengono lo stesso EF. Il numero
# risultante è "km/h equivalenti pianeggianti se corresse esattamente a soglia".
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


def _segment_time_and_efs(segments):
    """Tempo al centro segmento (ore) e EFS (km/h) ripulita da soste e glitch.
    I valori fuori range restano NaN invece di essere eliminati, così gli
    indici restano allineati alla griglia dei segmenti."""
    dt = segments["dt_s"].to_numpy()
    cum = np.cumsum(dt)
    mid_h = (cum - dt / 2.0) / 3600.0
    efs = segments["efs_ms"].to_numpy() * 3.6
    efs = np.where(np.isfinite(efs) & (efs >= EF_MIN_KMH) & (efs <= EF_MAX_KMH),
                   efs, np.nan)
    return mid_h, efs


def compute_ef_series(raw, segments, hr_threshold):
    """Serie EF a finestra mobile. Ritorna un DataFrame o None se la traccia
    è troppo corta / senza FC per riempire almeno 4 finestre."""
    if raw is None or segments is None or not hr_threshold or hr_threshold <= 0:
        return None
    if "hr" not in raw.columns:
        return None

    hr_ok = raw.dropna(subset=["hr", "elapsed_s"])
    if len(hr_ok) < 10:
        return None

    hr_h = hr_ok["elapsed_s"].to_numpy() / 3600.0
    hr_v = hr_ok["hr"].rolling(window=3, min_periods=1).mean().to_numpy()

    seg_h, seg_efs = _segment_time_and_efs(segments)
    if not np.isfinite(seg_efs).any():
        return None

    w = EF_WIN_MIN / 60.0
    t = max(float(hr_h.min()), float(seg_h.min())) + w / 2
    t_end = min(float(hr_h.max()), float(seg_h.max())) - w / 2

    rows = []
    while t <= t_end:
        mh = (hr_h >= t - w / 2) & (hr_h <= t + w / 2)
        me = (seg_h >= t - w / 2) & (seg_h <= t + w / 2) & np.isfinite(seg_efs)
        if mh.sum() >= 10 and me.sum() >= 5:
            hrm = float(np.nanmean(hr_v[mh]))
            efm = float(np.nanmean(seg_efs[me]))
            if hrm > 0 and np.isfinite(efm):
                rel = hrm / hr_threshold
                rows.append({
                    "t_h": t,
                    "hr_bpm": hrm,
                    "hr_rel_pct": rel * 100.0,
                    "efs_kmh": efm,
                    "ef": efm / rel,
                })
        t += EF_STEP_MIN / 60.0

    if len(rows) < 4:
        return None
    return pd.DataFrame(rows)


def _pad_range(arr, frac=0.10):
    """Range con margine: con più assi sovrapposti Plotly autoscala ciascuno
    per conto suo e taglia i picchi."""
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    span = max(hi - lo, 1e-9)
    return [lo - frac * span, hi + frac * span]


def build_ef_figure(ef_df, raw, title):
    """Stesso grafico del Race Analyzer: EF in primo piano, FC (% soglia) e
    EFS come curve di contesto su assi separati, profilo altimetrico di
    sfondo su un quarto asse invisibile schiacciato in basso."""
    fig = go.Figure()

    bg_x = raw["elapsed_s"].to_numpy() / 3600.0
    bg_y = raw["ele"].rolling(window=20, min_periods=1).mean().to_numpy()
    fig.add_trace(go.Scatter(
        x=bg_x, y=bg_y, mode="lines", name="Elevation",
        yaxis="y4", showlegend=False,
        line=dict(color="rgba(150,150,150,0.45)", width=0.8),
        fill="tozeroy", fillcolor="rgba(150,150,150,0.10)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ef_df["t_h"], y=ef_df["ef"], mode="lines",
        name="EF (km/h a soglia)",
        line=dict(color="#e07b39", width=2.6),
        hovertemplate="EF: %{y:.2f} km/h per unità di sforzo<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ef_df["t_h"], y=ef_df["hr_rel_pct"], mode="lines",
        name="FC (% soglia)", yaxis="y2",
        line=dict(color="rgba(70,140,220,0.6)", width=1.4),
        hovertemplate="FC: %{y:.0f}% della soglia<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=ef_df["t_h"], y=ef_df["efs_kmh"], mode="lines",
        name="EFS (km/h)", yaxis="y3",
        line=dict(color="rgba(42,157,143,0.6)", width=1.4),
        hovertemplate="EFS: %{y:.2f} km/h<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Elapsed Time (hours)", domain=[0.0, 0.88],
                   hoverformat=".2f"),
        yaxis=dict(title="EF (km/h per sforzo relativo)",
                   range=_pad_range(ef_df["ef"].to_numpy())),
        yaxis2=dict(title="FC (% soglia)", overlaying="y", side="right",
                    showgrid=False,
                    range=_pad_range(ef_df["hr_rel_pct"].to_numpy())),
        yaxis3=dict(title="EFS (km/h)", overlaying="y", side="right",
                    position=0.97, showgrid=False,
                    range=_pad_range(ef_df["efs_kmh"].to_numpy())),
        yaxis4=dict(overlaying="y", side="right", visible=False,
                    range=[float(np.nanmin(bg_y)), float(np.nanmax(bg_y)) * 2.6]),
        height=460, hovermode="x unified",
        margin=dict(t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="left", x=0),
    )
    return fig


def ef_stats(ef_df):
    """Metriche di sintesi: EF prima metà, seconda metà, media, delta."""
    ef = ef_df["ef"].to_numpy()
    half = len(ef) // 2
    ef1, ef2 = float(ef[:half].mean()), float(ef[half:].mean())
    return {
        "ef_first_half": ef1,
        "ef_second_half": ef2,
        "ef_delta": ef2 - ef1,
        "ef_mean": float(ef.mean()),
        "n_windows": int(len(ef)),
    }


# Colonne numeriche del riepilogo su cui ha senso calcolare statistiche.
EF_STAT_COLS = ["EF 1ª metà", "EF 2ª metà", "Delta", "EF medio",
                "Soglia (bpm)", "EFD (km)", "Durata (h)"]


def ef_descriptive_stats(summary_df):
    """Statistiche descrittive per ciascuna metrica, un file = un'osservazione.
    La moda su valori continui è quasi sempre priva di senso (ogni valore
    compare una volta sola), quindi si calcola sui valori arrotondati a 0.1
    e si riporta solo se c'è davvero una ripetizione."""
    rows = []
    for col in EF_STAT_COLS:
        if col not in summary_df.columns:
            continue
        s = pd.to_numeric(summary_df[col], errors="coerce").dropna()
        if s.empty:
            continue

        rounded = s.round(1)
        counts = rounded.value_counts()
        moda = float(counts.index[0]) if len(counts) and counts.iloc[0] > 1 else np.nan

        sd = float(s.std(ddof=1)) if len(s) > 1 else np.nan
        mean = float(s.mean())
        rows.append({
            "Metrica": col,
            "N": int(s.count()),
            "Min": float(s.min()),
            "Max": float(s.max()),
            "Range": float(s.max() - s.min()),
            "Media": mean,
            "Mediana": float(s.median()),
            "Moda (arr. 0.1)": moda,
            "Dev. std": sd,
            "CV %": (sd / mean * 100.0) if (np.isfinite(sd) and mean not in (0,)) else np.nan,
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "IQR": float(s.quantile(0.75) - s.quantile(0.25)),
        })
    return pd.DataFrame(rows)


def ef_group_stats(summary_df, by, label, min_groups=2, order=None):
    """Aggregazione dell'EF per una chiave di raggruppamento (atleta, bucket
    di lunghezza). min_groups=2 sopprime la tabella quando c'è un gruppo solo
    (non aggiungerebbe nulla al riepilogo); i bucket usano min_groups=1 perché
    lì la tabella serve comunque come vista per distanza. `order` impone
    l'ordine delle righe (i bucket vanno dal più corto al più lungo, non in
    ordine alfabetico)."""
    if by not in summary_df.columns:
        return None
    sub = summary_df.dropna(subset=["EF medio"])
    if sub.empty or sub[by].nunique() < min_groups:
        return None

    g = (sub.groupby(by, dropna=False)
            .agg(N=("File", "count"),
                 ef_mean=("EF medio", "mean"),
                 ef_median=("EF medio", "median"),
                 ef_min=("EF medio", "min"),
                 ef_max=("EF medio", "max"),
                 delta_mean=("Delta", "mean"),
                 efd_mean=("EFD (km)", "mean"))
            .reset_index())

    if order:
        rank = {name: i for i, name in enumerate(order)}
        g = (g.assign(_ord=g[by].map(lambda v: rank.get(v, len(rank))))
               .sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True))

    return g.rename(columns={
        by: label, "N": "N file",
        "ef_mean": "EF medio (media)", "ef_median": "EF medio (mediana)",
        "ef_min": "EF min", "ef_max": "EF max",
        "delta_mean": "Delta medio", "efd_mean": "EFD media (km)",
    })


def ef_correlations(summary_df):
    """Correlazioni di Pearson tra coppie che potrebbero spiegare la
    dispersione dell'EF. Con pochi file r è instabile: la colonna N serve
    a ricordarlo, non è un test di significatività."""
    pairs = [
        ("EFD (km)", "EF medio"),
        ("EFD (km)", "Delta"),
        ("Durata (h)", "Delta"),
        ("Soglia (bpm)", "EF medio"),
    ]
    rows = []
    for a, b in pairs:
        if a not in summary_df.columns or b not in summary_df.columns:
            continue
        d = summary_df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < 3 or d[a].nunique() < 2 or d[b].nunique() < 2:
            continue
        r = float(np.corrcoef(d[a], d[b])[0, 1])
        rows.append({"X": a, "Y": b, "N": len(d), "r di Pearson": r,
                     "R²": r ** 2})
    return pd.DataFrame(rows)


def _df_to_html(df):
    return df.to_html(index=False, float_format=lambda v: f"{v:.2f}",
                      na_rep="—")


def build_ef_html(blocks, summary_df, stats_df, group_tables, corr_df,
                  n_excluded=0):
    """Report HTML autoconsistente: tabella riassuntiva + un grafico e le
    metriche per ciascun file. plotly.js caricato una volta sola da CDN,
    quindi le figure vanno inserite con include_plotlyjs=False."""
    parts = [
        "<!DOCTYPE html><html lang='it'><head><meta charset='utf-8'>",
        "<title>Efficiency Factor — report</title>",
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "margin:32px auto;max-width:1100px;color:#1c1c1c;line-height:1.5;}",
        "h1{margin-bottom:4px;} h2{margin-top:38px;border-bottom:1px solid #ddd;padding-bottom:6px;}",
        "table{border-collapse:collapse;margin:12px 0;font-size:14px;}",
        "th,td{border:1px solid #ddd;padding:6px 10px;text-align:right;}",
        "th{background:#f4f4f4;text-align:center;} td:first-child,th:first-child{text-align:left;}",
        ".meta{color:#555;font-size:14px;} .note{color:#555;font-size:13px;margin-top:10px;}",
        "</style></head><body>",
        "<h1>Efficiency Factor</h1>",
        f"<p class='meta'>Finestra mobile {EF_WIN_MIN} min, passo {EF_STEP_MIN} min. "
        "EF = EFS &divide; (FC / soglia): velocità equivalente pianeggiante per "
        "unità di sforzo relativo (1.0 = a soglia). Essendo la FC normalizzata "
        "sulla soglia di ciascun atleta, il numero è confrontabile tra atleti "
        "con soglie diverse.</p>",
        f"<p class='meta'>File analizzati: <b>{len(blocks)}</b>"
        + (f", di cui <b>{n_excluded}</b> esclusi dalle statistiche"
           if n_excluded else "")
        + ".</p>",
    ]

    if stats_df is not None and not stats_df.empty:
        parts.append("<h2>Statistiche descrittive</h2>")
        parts.append("<p class='note'>Un file = un'osservazione. La moda è "
                     "calcolata sui valori arrotondati a 0.1 e riportata solo "
                     "se almeno due file cadono sullo stesso valore.</p>")
        parts.append(_df_to_html(stats_df))

    for label, gdf in group_tables:
        parts.append(f"<h2>EF per {label}</h2>")
        parts.append(_df_to_html(gdf))

    if corr_df is not None and not corr_df.empty:
        parts.append("<h2>Correlazioni</h2>")
        parts.append("<p class='note'>Pearson su pochi file è instabile: "
                     "leggilo come indizio di una relazione, non come prova.</p>")
        parts.append(_df_to_html(corr_df))

    for b in blocks:
        s = b["stats"]
        parts.append(f"<h2>{b['name']}"
                     + (" <small>(escluso dalle statistiche)</small>"
                        if b.get("excluded") else "")
                     + "</h2>")
        parts.append(
            "<p class='meta'>"
            f"EFD <b>{b.get('efd_km', float('nan')):.2f} km</b> &middot; "
            f"durata <b>{b.get('durata', '—')}</b> &middot; "
            f"bucket <b>{b.get('bucket', '—')}</b> &middot; "
            f"soglia FC (z4) <b>{b['threshold']:.0f} bpm</b>"
            + (f" &middot; atleta: {b['athlete']}" if b.get("athlete") else "")
            + f" &middot; finestre valide: {s['n_windows']}</p>"
        )
        parts.append(
            "<table><tr><th>EF 1ª metà</th><th>EF 2ª metà</th>"
            "<th>Delta</th><th>EF medio gara</th></tr>"
            f"<tr><td>{s['ef_first_half']:.2f}</td><td>{s['ef_second_half']:.2f}</td>"
            f"<td>{s['ef_delta']:+.2f}</td><td>{s['ef_mean']:.2f}</td></tr></table>"
        )
        parts.append(b["fig"].to_html(full_html=False, include_plotlyjs=False,
                                      default_width="100%"))

    parts.append("<h2>Riepilogo per file</h2>")
    parts.append(_df_to_html(summary_df))

    parts.append("</body></html>")
    return "\n".join(parts)


def render_ef_analysis(per_file_raw, per_file_segments, per_file_summary,
                       per_file_bucket, bucket_order):
    st.header("💓 Efficiency Factor per file")
    st.caption(
        f"EF = EFS ÷ (FC / soglia), su finestra mobile di {EF_WIN_MIN} minuti con passo "
        f"{EF_STEP_MIN} minuti. Serve la FC di soglia (z4) di ciascun file: caricala con "
        "il CSV delle zone esportato dal Race Analyzer. I file senza CSV vengono saltati."
    )

    if not per_file_raw:
        st.info("Carica uno o più file .fit dalla sidebar per eseguire questa analisi.")
        return

    # --- 1) una casella di upload per ogni file .fit caricato ---
    # Qui sta anche l'interruttore di esclusione: le statistiche vengono
    # stampate PRIMA dei grafici, quindi i controlli che le influenzano
    # devono trovarsi sopra, altrimenti si modifica un risultato già letto.
    st.subheader("1. Zone FC per file")
    st.caption(
        "Spunta **Escludi** per togliere un file dalle statistiche aggregate "
        "(dati sporchi, fascia cardio ballerina, gara interrotta): il grafico "
        "resta comunque visibile più sotto."
    )
    zones_by_file, excluded_files = {}, set()
    for fname in per_file_raw:
        c_name, c_up, c_ex = st.columns([1, 1.3, 0.5])
        c_name.markdown(f"**{fname}**")
        csv_file = c_up.file_uploader(
            f"CSV zone FC — {fname}", type=["csv"],
            key=f"ef_zones_csv_{fname}", label_visibility="collapsed",
        )
        if c_ex.checkbox("Escludi", key=f"ef_exclude_{fname}"):
            excluded_files.add(fname)
        if csv_file is None:
            continue
        zones, err = read_hr_zones_csv(csv_file)
        if err:
            c_name.caption(f"⚠️ {err}")
            continue
        zones_by_file[fname] = zones
        c_name.caption(f"✅ soglia {zones['threshold']:.0f} bpm"
                       + (f" · {zones['athlete']}" if zones.get("athlete") else ""))

    if not zones_by_file:
        st.info("Carica almeno un CSV di zone FC per procedere. "
                "Il CSV deve contenere una colonna `z4` (FC di soglia).")
        return

    # --- calcolo EF sui soli file con soglia disponibile ---
    blocks, summary_rows = [], []
    for fname, zones in zones_by_file.items():
        ef_df = compute_ef_series(per_file_raw[fname], per_file_segments[fname],
                                  zones["threshold"])
        if ef_df is None:
            st.warning(f"⚠️ {fname}: traccia troppo corta o senza FC utilizzabile "
                       f"(servono almeno 4 finestre da {EF_WIN_MIN} min), file saltato.")
            continue

        stats = ef_stats(ef_df)
        fig = build_ef_figure(
            ef_df, per_file_raw[fname],
            f"{fname} — Efficiency Factor (finestra {EF_WIN_MIN} min)",
        )
        summ = per_file_summary.get(fname, {})
        total_time_s = summ.get("total_time_s", np.nan)
        efd_km = summ.get("efd_m", np.nan) / 1000
        bucket_name = per_file_bucket.get(fname, ("—", np.nan))[0]

        blocks.append({
            "name": fname, "ef_df": ef_df, "fig": fig, "stats": stats,
            "threshold": zones["threshold"], "athlete": zones.get("athlete"),
            "excluded": fname in excluded_files,
            "efd_km": efd_km, "durata": seconds_to_hhmm(total_time_s),
            "bucket": bucket_name,
        })

        summary_rows.append({
            "File": fname,
            "Escluso": "✖" if fname in excluded_files else "",
            "_excl": fname in excluded_files,
            "Atleta": zones.get("athlete", "—"),
            "Bucket": bucket_name,
            "Soglia (bpm)": zones["threshold"],
            "Durata": seconds_to_hhmm(total_time_s),
            "Durata (h)": total_time_s / 3600 if np.isfinite(total_time_s) else np.nan,
            "EFD (km)": efd_km,
            "EF 1ª metà": stats["ef_first_half"],
            "EF 2ª metà": stats["ef_second_half"],
            "Delta": stats["ef_delta"],
            "EF medio": stats["ef_mean"],
            "Finestre": stats["n_windows"],
        })

    if not blocks:
        st.info("Nessun file idoneo al calcolo dell'EF.")
        return

    full_df = pd.DataFrame(summary_rows)
    summary_df = full_df.drop(columns=["_excl"])
    stats_source = full_df[~full_df["_excl"]].drop(columns=["_excl"])
    n_excluded = int(full_df["_excl"].sum())

    # --- 2) statistiche sull'insieme dei file (in cima, prima dei grafici) ---
    st.divider()
    st.subheader("2. Statistiche sull'insieme dei file")
    if stats_source.empty:
        st.warning("Tutti i file sono esclusi: nessuna statistica da calcolare.")
        stats_df, group_tables, corr_df = pd.DataFrame(), [], pd.DataFrame()
    else:
        st.caption(
            f"**{len(stats_source)} file nelle statistiche**"
            + (f" ({n_excluded} esclusi manualmente)" if n_excluded else "")
            + ". Un file = un'osservazione. La moda è calcolata sui valori "
            "arrotondati a 0.1 e mostrata solo se almeno due file cadono sullo "
            "stesso valore: su misure continue una moda \"vera\" non esiste quasi mai."
        )
        stats_df = ef_descriptive_stats(stats_source)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        group_tables = []

        # Bucket: sempre mostrato (min_groups=1) e ordinato per lunghezza
        # crescente, non alfabeticamente. Sono gli stessi bucket impostati
        # nella sidebar, quindi le soglie si cambiano da lì.
        gdf = ef_group_stats(stats_source, "Bucket", "Bucket",
                             min_groups=1, order=bucket_order)
        if gdf is not None:
            group_tables.append(("bucket di lunghezza gara", gdf))
            st.markdown("**EF per bucket di lunghezza gara**")
            st.dataframe(gdf, use_container_width=True, hide_index=True)
            st.caption("Bucket definiti dalle soglie EFD nella sidebar.")

        gdf = ef_group_stats(stats_source, "Atleta", "Atleta")
        if gdf is not None:
            group_tables.append(("atleta", gdf))
            st.markdown("**EF per atleta**")
            st.dataframe(gdf, use_container_width=True, hide_index=True)

        corr_df = ef_correlations(stats_source)
        if corr_df is not None and not corr_df.empty:
            st.markdown("**Correlazioni**")
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
            st.caption(
                "Pearson su pochi file è instabile: leggilo come indizio di una "
                "relazione (es. gare più lunghe = calo di EF maggiore), non come prova."
            )

    # --- 3) grafici per file ---
    st.divider()
    st.subheader("3. Grafici per file")
    for b in blocks:
        st.markdown(f"#### {b['name']}"
                    + ("  ·  *escluso dalle statistiche*" if b["excluded"] else ""))
        st.caption(
            f"EFD **{b['efd_km']:.2f} km** · durata **{b['durata']}** · "
            f"bucket **{b['bucket']}** · soglia FC **{b['threshold']:.0f} bpm**"
            + (f" · atleta **{b['athlete']}**" if b.get("athlete") else "")
        )
        st.plotly_chart(b["fig"], use_container_width=True,
                        key=f"ef_chart_{b['name']}")
        s = b["stats"]
        c1, c2, c3 = st.columns(3)
        c1.metric("EF 1ª metà", f"{s['ef_first_half']:.2f}")
        c2.metric("EF 2ª metà", f"{s['ef_second_half']:.2f}", f"{s['ef_delta']:+.2f}")
        c3.metric("EF medio gara", f"{s['ef_mean']:.2f}")
        st.caption(
            "Un EF di 12.3 significa 12.3 km/h equivalenti pianeggianti se "
            "l'atleta corresse esattamente a soglia."
        )

    # I dati per finestra non vengono più stampati (né a schermo né nel report):
    # con molti file erano decine di tabelle e un HTML enorme. Restano
    # disponibili in un unico CSV per chi vuole rilavorarli.
    windows_df = pd.concat(
        [b["ef_df"].assign(file=b["name"], escluso=b["excluded"]) for b in blocks],
        ignore_index=True
    )[["file", "escluso", "t_h", "ef", "hr_bpm", "hr_rel_pct", "efs_kmh"]]
    st.download_button(
        "📥 Scarica i dati EF per finestra, tutti i file (CSV)",
        data=windows_df.to_csv(index=False).encode("utf-8"),
        file_name="efficiency_factor_finestre.csv",
        mime="text/csv",
    )

    # --- 4) report HTML unico ---
    st.divider()
    st.subheader("4. Report HTML")
    html_report = build_ef_html(blocks, summary_df, stats_df, group_tables,
                                corr_df, n_excluded)
    st.download_button(
        "📥 Scarica il report HTML",
        data=html_report.encode("utf-8"),
        file_name="efficiency_factor_report.html",
        mime="text/html",
    )
    st.caption(
        "Il file è autoconsistente (plotly.js da CDN): si apre in qualsiasi browser "
        "e contiene gli stessi grafici e tabelle mostrati qui sopra."
    )

    # --- 5) riepilogo per file, in fondo ---
    st.divider()
    st.subheader("5. Riepilogo per file")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Scarica il riepilogo (CSV)",
        data=summary_df.to_csv(index=False).encode("utf-8"),
        file_name="efficiency_factor_riepilogo.csv",
        mime="text/csv",
    )