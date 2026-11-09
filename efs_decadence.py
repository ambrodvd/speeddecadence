"""ANALISI — Decadimento EFS per bucket di lunghezza gara, con predizione
sfumata (blended) e pannelli diagnostici sui file caricati."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from efs_models import (MIN_FILES_FOR_CUBIC, MIN_FILES_FOR_QUADRATIC,
                        best_candidate, fit_candidates)
from fit_core import MIN_FILE_DURATION_S, seconds_to_hhmm


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
# ANALISI 1 — Decadimento EFS per bucket
# ===========================================================
def render_decadence_analysis(per_file_segments, per_file_bucket,
                              bucket_order, bucket_bounds, bin_width_pct):
    st.header("📉 Decadimento EFS per bucket di lunghezza gara")
    st.caption(
        f"Per ciascun file viene calcolato l'EFD progressivo in **% del totale del file** e tagliato in fette "
        f"di {bin_width_pct:g}%: per ciascuna fetta si calcola lo scostamento percentuale dell'EFS di fetta "
        "rispetto all'EFS medio dell'intero file. I file vengono raggruppati per bucket di EFD totale "
        "(lunghezza gara) e fittati separatamente, cosi' gare corte e lunghe non vengono mediate insieme."
    )

    if not per_file_segments:
        st.info("Carica uno o più file .fit dalla sidebar per eseguire questa analisi.")
        return

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
        return

    decadence_df = pd.DataFrame(decadence_rows)

    bucket_fits = {}            # bucket_name -> miglior candidato
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
        target_efd_km = st.number_input("EFD totale gara target (km)", min_value=1.0,
                                        value=80.0, step=5.0, key="target_efd_km")
    with col2:
        target_pct = st.slider("Posizione in gara (% EFD accumulato)", 0, 100, 50,
                               key="target_pct")

    pred_deviation = predict_blended(target_pct, target_efd_km, bucket_fits,
                                     bucket_centers, bucket_order)
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
# Pannelli diagnostici sui file caricati (riepilogo + file per bucket)
# ===========================================================
# Stavano sopra le tab e comparivano ogni volta che si attivava un'analisi
# basata su EFD (anche EF ed EF a blocchi). Riguardano i bucket, quindi
# vivono qui, in cima alla tab del decadimento.
_SUMMARY_FMT = {
    "Distanza (km)": "{:.2f}",
    "D+ (m)": "{:.0f}",
    "D- (m)": "{:.0f}",
    "EFD (km)": "{:.2f}",
    "EFS media (km/h)": "{:.2f}",
}


def render_file_overview(per_file_summary, per_file_segments, per_file_bucket,
                         bucket_order):
    if not per_file_summary:
        return

    summary_rows = []
    for fname, summ in per_file_summary.items():
        t = summ["total_time_s"]
        summary_rows.append({
            "File": fname,
            "Durata": seconds_to_hhmm(t),
            "Distanza (km)": summ["horizontal_distance_m"] / 1000,
            "D+ (m)": summ["d_plus_m"],
            "D- (m)": summ["d_minus_m"],
            "EFD (km)": summ["efd_m"] / 1000,
            "EFS media (km/h)": summ["efd_m"] / t * 3.6 if t > 0 else np.nan,
            "Bucket": per_file_bucket[fname][0],
        })
    with st.expander(f"Riepilogo file ({len(summary_rows)})", expanded=False):
        st.dataframe(pd.DataFrame(summary_rows).style.format(_SUMMARY_FMT, na_rep="—"),
                     use_container_width=True, hide_index=True)

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
            f"File idonei (>={MIN_FILE_DURATION_S // 60} min)": len(eligible),
            "Modelli disponibili": note,
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Un bucket con meno di {MIN_FILES_FOR_QUADRATIC} file idonei usa solo modelli a 2 parametri "
        f"(lineare/log/radice); servono almeno {MIN_FILES_FOR_CUBIC} file per la cubica. "
        "Questo perché i bin di uno stesso file non sono osservazioni indipendenti: "
        "il vero campione statistico è il numero di file, non il numero di righe aggregate."
    )
    st.divider()