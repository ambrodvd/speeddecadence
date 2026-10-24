"""Riepilogo frequenza cardiaca per file .fit.

Per ogni file: media FC totale, media della prima e della seconda metà
(divise a metà della durata) e differenza % seconda vs prima metà.

Uso nel main app:
    from hr_summary import render as render_hr_summary
    render_hr_summary(uploaded_files)
"""
import io

import pandas as pd
import streamlit as st
from fitparse import FitFile

SOGLIA_DIFF = 5.0  # |differenza %| oltre cui la cella diventa rossa

RIGA_MEDIA = "Media totale"
RIGA_PRIMA = "Prima metà"
RIGA_SECONDA = "Seconda metà"
RIGA_DIFF = "Differenza"


@st.cache_data(show_spinner=False)
def _leggi_fc(contenuto: bytes) -> pd.DataFrame:
    """Estrae timestamp e FC (bpm > 0) dai record del file .fit."""
    righe = []
    for rec in FitFile(io.BytesIO(contenuto)).get_messages("record"):
        v = rec.get_values()
        hr = v.get("heart_rate")
        if hr is not None and hr > 0:
            righe.append((v.get("timestamp"), float(hr)))
    df = pd.DataFrame(righe, columns=["timestamp", "hr"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def riepilogo_fc(df: pd.DataFrame) -> dict:
    """Medie FC totale, prima metà, seconda metà e differenza %."""
    if df.empty:
        return {RIGA_MEDIA: None, RIGA_PRIMA: None, RIGA_SECONDA: None, RIGA_DIFF: None}

    hr = df["hr"].reset_index(drop=True)
    ts = df["timestamp"].reset_index(drop=True)
    if ts.notna().all():
        # divisione a metà della durata (robusta a registrazione smart)
        t = (ts - ts.iloc[0]).dt.total_seconds()
        prima = t <= t.iloc[-1] / 2
    else:
        # senza timestamp: divisione a metà dei campioni
        prima = pd.Series(range(len(hr))) < len(hr) / 2

    m_tot = hr.mean()
    m1 = hr[prima].mean()
    m2 = hr[~prima].mean()
    diff = (m2 - m1) / m1 * 100 if pd.notna(m1) and pd.notna(m2) and m1 > 0 else None
    return {RIGA_MEDIA: m_tot, RIGA_PRIMA: m1, RIGA_SECONDA: m2, RIGA_DIFF: diff}


def costruisci_tabella(files) -> pd.DataFrame:
    """Una colonna per file, una riga per metrica."""
    colonne = {}
    for f in files:
        colonne[f.name] = riepilogo_fc(_leggi_fc(f.getvalue()))
    tab = pd.DataFrame(colonne, index=[RIGA_MEDIA, RIGA_PRIMA, RIGA_SECONDA, RIGA_DIFF])
    tab.index.name = "Metrica (bpm)"
    return tab.astype(float)


def _colore_diff(riga: pd.Series) -> list:
    if riga.name != RIGA_DIFF:
        return [""] * len(riga)
    stili = []
    for v in riga:
        if pd.isna(v):
            stili.append("")
        elif abs(v) <= SOGLIA_DIFF:
            stili.append("background-color: #dff3e4; color: #1e6b35")
        else:
            stili.append("background-color: #fbe3e3; color: #a12a2a")
    return stili


def stile_tabella(tab: pd.DataFrame):
    righe_bpm = [RIGA_MEDIA, RIGA_PRIMA, RIGA_SECONDA]
    return (
        tab.style
        .format("{:.0f}", subset=pd.IndexSlice[righe_bpm, :], na_rep="—")
        .format("{:+.1f}%", subset=pd.IndexSlice[[RIGA_DIFF], :], na_rep="—")
        .apply(_colore_diff, axis=1)
    )


def render(files) -> None:
    st.subheader("Riepilogo frequenza cardiaca")

    if not files:
        st.info("Carica uno o più file .fit per iniziare.")
        return

    # se cambiano i file caricati, il risultato precedente viene scartato
    firma = tuple((f.name, f.size) for f in files)
    if st.session_state.get("hrs_firma") != firma:
        st.session_state["hrs_firma"] = firma
        st.session_state.pop("hrs_tabella", None)

    st.write(f"{len(files)} file caricati. Vuoi calcolare il riepilogo della frequenza cardiaca?")
    if st.button("Avvia analisi", key="hrs_avvia"):
        with st.spinner("Analisi in corso…"):
            st.session_state["hrs_tabella"] = costruisci_tabella(files)

    tab = st.session_state.get("hrs_tabella")
    if tab is None:
        return

    st.dataframe(stile_tabella(tab), use_container_width=True)
    st.caption(
        "Differenza = media della seconda metà rispetto alla prima metà. "
        f"Verde se entro ±{SOGLIA_DIFF:.0f}%, rosso se oltre. "
        "“—” = nessun dato di frequenza cardiaca nel file."
    )