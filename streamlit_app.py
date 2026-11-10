import streamlit as st

from fit_core import assign_bucket, get_bucket_definitions, load_and_process, load_raw
from efs_decadence import render_decadence_analysis, render_file_overview
from slope_study import render_slope_analysis
from race_plan_compare import render_comparison_analysis
from efficiency_factor import render_ef_analysis
from ef_blocks import render_ef_blocks
from cpet_steady_state import render_cpet_analysis
from forceplate_norms import render_forceplate_norms
from hr_summary import render as render_hr_summary
from jump_height_compare import render_jump_height_compare
from uphill_flat import render_uphill_flat

st.set_page_config(
    page_title="AMBRO BIG DATA BROTHER",
    page_icon="👁️‍🗨️",
    layout="wide",
)

# Testo unico del gate: ogni analisi è pesante, quindi parte solo su richiesta.
RUN_PROMPT = "Vuoi eseguire l'analisi?"


# ===========================================================
# Sidebar — upload & impostazioni
# ===========================================================
with st.sidebar:
    st.header("Impostazioni")
    uploaded_files = st.file_uploader(
        "Carica uno o più file .fit (anche .gz)",
        type=["fit", "gz"], accept_multiple_files=True
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

st.title("👁️‍🗨️ AMBRO BIG DATA BROTHER 👁️‍🗨️")
st.info("This is where your data gets crunched")

bucket_order, bucket_bounds = get_bucket_definitions(thr1, thr2, thr3)

# ===========================================================
# Elaborazione file caricati (comune a tutte le tab)
# ===========================================================
per_file_segments = {}   # filename -> segments dataframe
per_file_summary = {}    # filename -> summary dict
per_file_bucket = {}     # filename -> (bucket_name, efd_totale_km)
per_file_raw = {}        # filename -> raw record dataframe (serve per la FC)

# Il calcolo EFD/EFS (pesante) parte solo se è attiva un'analisi che lo usa.
# I checkbox sono in fondo alla pagina, ma il loro valore è già in
# session_state all'inizio del rerun scatenato dal click.
# Sopra le tab non si stampa nulla oltre agli avvisi sui file scartati:
# riepilogo e distribuzione per bucket stanno nella tab del decadimento.
EFD_ANALYSIS_KEYS = ("run_decadence", "run_slopes", "run_ef")
needs_efd = any(st.session_state.get(k, False) for k in EFD_ANALYSIS_KEYS)

if uploaded_files and needs_efd:
    for f in uploaded_files:
        try:
            raw, segments, summary = load_and_process(
                f.getvalue(), smooth_window, resample_step)
        except Exception as e:
            st.warning(f"⚠️ {f.name}: file illeggibile o errore di elaborazione, escluso ({e.__class__.__name__}).")
            continue

        if raw is None:
            st.warning(f"⚠️ {f.name}: nessun dato GPS valido, file escluso.")
            continue

        if segments is None:
            st.warning(f"⚠️ {f.name}: distanza totale nulla, file escluso.")
            continue

        per_file_segments[f.name] = segments
        per_file_raw[f.name] = raw
        per_file_summary[f.name] = summary

        efd_totale_km = summary["efd_m"] / 1000
        per_file_bucket[f.name] = (assign_bucket(efd_totale_km, bucket_order, bucket_bounds),
                                   efd_totale_km)
elif uploaded_files:
    st.info(
        f"{len(uploaded_files)} file caricati. Il calcolo EFD/EFS parte solo quando "
        "attivi un'analisi che lo richiede (decadimento EFS, studio pendenze, "
        "Efficiency Factor)."
    )
else:
    st.info("Carica uno o più file .fit dalla sidebar per iniziare.")


def load_raw_only(files):
    """Solo parsing (niente EFD/EFS), per le tab che usano FC e tempo."""
    out = {}
    for f in files or []:
        try:
            raw = load_raw(f.getvalue())
        except Exception as e:
            st.warning(f"⚠️ {f.name}: file illeggibile, escluso ({e.__class__.__name__}).")
            continue
        if not raw.empty:
            out[f.name] = raw
    return out


# ===========================================================
# TAB — una analisi per tab, ognuna dietro il proprio checkbox
# ===========================================================
# Ogni analisi è costosa (decine di file, fit multipli, grafici): il checkbox
# fa da interruttore. È un checkbox e non un bottone perché un bottone è True
# solo nel rerun del click e il risultato sparirebbe al widget successivo.
st.divider()

tab_dec, tab_slope, tab_cmp, tab_ef, tab_efb, tab_cpet, tab_fpn, tab_jh, tab_fc, tab_uf = st.tabs([
    "📉 Decadimento EFS per bucket",
    "📐 Studio pendenze FC / EFS",
    "🆚 Race plan vs gara reale",
    "💓 Efficiency Factor",
    "⏱️ EF a blocchi 15'",
    "🫁 CPET step & steady state",
    "🏋️ Norme Force Plate",
    "🦘 Jump height FT vs NI",
    "❤️ Riepilogo FC",
    "⛰️ Salita vs piano",
])

with tab_dec:
    run_decadence = st.checkbox(RUN_PROMPT, value=False, key="run_decadence")
    if run_decadence:
        render_file_overview(per_file_summary, per_file_segments, per_file_bucket,
                             bucket_order)
        render_decadence_analysis(per_file_segments, per_file_bucket,
                                  bucket_order, bucket_bounds, bin_width_pct)
    else:
        st.info("☝️ Spunta la casella per eseguire l'analisi del decadimento EFS per bucket.")

with tab_slope:
    run_slopes = st.checkbox(RUN_PROMPT, value=False, key="run_slopes")
    if run_slopes:
        render_slope_analysis(per_file_segments, per_file_summary, per_file_raw)
    else:
        st.info("☝️ Spunta la casella per eseguire lo studio delle pendenze FC / EFS.")

with tab_cmp:
    run_compare = st.checkbox(RUN_PROMPT, value=False, key="run_compare")
    if run_compare:
        render_comparison_analysis(smooth_window, resample_step, bin_width_pct)
    else:
        st.info("☝️ Spunta la casella per eseguire il confronto race plan vs gara reale.")

with tab_ef:
    run_ef = st.checkbox(RUN_PROMPT, value=False, key="run_ef")
    if run_ef:
        render_ef_analysis(per_file_raw, per_file_summary,
                           per_file_bucket, bucket_order)
    else:
        st.info("☝️ Spunta la casella per calcolare l'Efficiency Factor.")

with tab_efb:
    run_ef_blocks = st.checkbox(RUN_PROMPT, value=False, key="run_ef_blocks")
    if run_ef_blocks:
        render_ef_blocks(load_raw_only(uploaded_files))
    else:
        st.info("☝️ Spunta la casella per calcolare l'EF a blocchi di 15 minuti.")

with tab_cpet:
    run_cpet = st.checkbox(RUN_PROMPT, value=False, key="run_cpet")
    if run_cpet:
        render_cpet_analysis()
    else:
        st.info("☝️ Spunta la casella per eseguire l'analisi CPET a step.")

with tab_fpn:
    run_fpn = st.checkbox(RUN_PROMPT, value=False, key="run_fpn")
    if run_fpn:
        render_forceplate_norms()
    else:
        st.info("☝️ Spunta la casella per calcolare le norme di popolazione Force Plate.")

with tab_jh:
    run_jh = st.checkbox(RUN_PROMPT, value=False, key="run_jh")
    if run_jh:
        render_jump_height_compare()
    else:
        st.info("☝️ Spunta la casella per confrontare jump height FT vs NI.")

with tab_fc:
    run_fc = st.checkbox(RUN_PROMPT, value=False, key="run_fc")
    if run_fc:
        render_hr_summary(load_raw_only(uploaded_files))
    else:
        st.info("☝️ Spunta la casella per calcolare il riepilogo della frequenza cardiaca.")

with tab_uf:
    run_uf = st.checkbox(RUN_PROMPT, value=False, key="run_uf")
    if run_uf:
        render_uphill_flat(load_raw_only(uploaded_files), resample_step)
    else:
        st.info("☝️ Spunta la casella per analizzare i tratti in salita e in piano.")