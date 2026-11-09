"""ANALISI — Race plan previsto (CSV del Trail Predictor) vs gara reale (.fit)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from efs_models import MIN_FILES_FOR_CUBIC, best_candidate, fit_candidates
from fit_core import (_maybe_gunzip, hhmm_to_seconds, parse_fit, process_track,
                      seconds_to_hhmm)


# ---------------------------------------------------------------------------
# Equazione di decadimento EFS EFFETTIVAMENTE USATA per generare i race plan
# ---------------------------------------------------------------------------
# ATTENZIONE: questa è una copia 1:1 dei coefficienti hardcoded nel trail
# predictor (streamlit_app.py, BUCKET_DELTA_EFS_FN / predict_delta_efs_blended).
# NON va confusa con bucket_fits/predict_blended qui sopra: quelli si
# ri-fittano LIVE sui file di training caricati in questa sessione e possono
# differire (più file, soglie diverse) da quelli usati quando un dato race
# plan è stato effettivamente generato. Per confrontare "previsto vs reale"
# nella sezione di confronto serve l'equazione congelata al momento della
# generazione del piano, quindi si usa QUESTA.
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
# ANALISI 3 — Confronto Race Plan previsto vs Performance reale
# ===========================================================
def render_comparison_analysis(smooth_window, resample_step, bin_width_pct):
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
            type=["fit", "gz"], key="compare_real_fit",
        )

    if plan_csv is None or real_fit is None:
        st.info("Carica sia il CSV del race plan sia il file .fit della gara reale per procedere.")
        return

    try:
        plan_df_cmp = pd.read_csv(plan_csv)
    except Exception as e:
        st.error(f"⚠️ Errore nella lettura del CSV: {e}")
        return

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
        return

    if len(tempo_gara_cols) > 1:
        tempo_gara_col = st.selectbox(
            "Più profili trovati nel CSV: quale usare come riferimento?",
            tempo_gara_cols, key="tempo_gara_col_select",
        )
    else:
        tempo_gara_col = tempo_gara_cols[0]

    # --- elabora il file .fit reale ---
    try:
        raw_real = parse_fit(_maybe_gunzip(real_fit))
    except Exception as e:
        st.error(f"⚠️ File .fit illeggibile o corrotto: {e}")
        return

    if raw_real.empty or len(raw_real) < 2:
        st.error("⚠️ Nessun dato GPS valido nel file .fit della gara reale.")
        return

    segments_real, summary_real = process_track(raw_real, smooth_window, resample_step)
    if segments_real is None:
        st.error("⚠️ Traccia reale degenere: distanza orizzontale nulla.")
        return

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
        "quella ri-fittata live nella tab del decadimento)."
    )

    if total_time_real_s <= 0 or total_efd_real_km <= 0:
        st.info("Dati reali insufficienti per calcolare il decadimento.")
        return

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
    # (fit_candidates/best_candidate), ma qui i punti vengono da un solo file.
    real_best_fit = None
    real_best_name = None
    real_fit_candidates = {}
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