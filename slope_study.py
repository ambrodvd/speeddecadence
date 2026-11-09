"""ANALISI — Studio pendenze: regressione lineare di FC ed EFS sul tempo."""
import numpy as np
import pandas as pd
import streamlit as st

from fit_core import MIN_FILE_DURATION_S


# ===========================================================
# ANALISI 2 — Studio pendenze: regressione lineare FC e EFS sul tempo
# ===========================================================
# Pendenze PURE, senza costanti di scala e senza normalizzazioni:
#   FC  -> y = bpm,  x = tempo
#   EFS -> y = km/h, x = tempo
# Si riportano entrambe le unità di x (secondi e ore): è la stessa
# retta, cambia solo il fattore 3600. Il DET index dell'analyzer usa
# la pendenza al secondo, quindi la colonna /s è quella confrontabile.
def render_slope_analysis(per_file_segments, per_file_summary, per_file_raw):
    st.header("📐 Studio pendenze: FC e EFS vs tempo")

    if not per_file_segments:
        st.info("Carica dei file .fit dalla sidebar per eseguire lo studio.")
        return

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
        return

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