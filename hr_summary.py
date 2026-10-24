"""Riepilogo frequenza cardiaca per file .fit.

Per ogni file: media FC totale, media della prima e della seconda metà
(divise a metà della durata) e differenza % seconda vs prima metà.

Usa i record grezzi già parsati dal main app (per_file_raw), quindi
nessun nuovo parsing e i file .gz funzionano come nelle altre tab.

Uso nel main app:
    from hr_summary import render as render_hr_summary
    render_hr_summary(per_file_raw)
"""
import pandas as pd
import streamlit as st

SOGLIA_DIFF = 5.0  # |differenza %| oltre cui la cella diventa rossa

RIGA_MEDIA = "Media totale"
RIGA_PRIMA = "Prima metà"
RIGA_SECONDA = "Seconda metà"
RIGA_DIFF = "Differenza"
RIGHE = [RIGA_MEDIA, RIGA_PRIMA, RIGA_SECONDA, RIGA_DIFF]


def riepilogo_fc(raw: pd.DataFrame) -> dict:
    """Medie FC totale, prima metà, seconda metà e differenza %."""
    vuoto = {r: None for r in RIGHE}
    if raw is None or "hr" not in raw.columns or "timestamp" not in raw.columns:
        return vuoto

    df = raw[["timestamp", "hr"]].copy()
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    df = df[df["hr"] > 0].dropna().reset_index(drop=True)
    if df.empty:
        return vuoto

    # divisione a metà della durata (robusta a registrazione smart)
    t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    prima = t <= t.iloc[-1] / 2

    m_tot = df["hr"].mean()
    m1 = df.loc[prima, "hr"].mean()
    m2 = df.loc[~prima, "hr"].mean()
    diff = (m2 - m1) / m1 * 100 if pd.notna(m1) and pd.notna(m2) and m1 > 0 else None
    return {RIGA_MEDIA: m_tot, RIGA_PRIMA: m1, RIGA_SECONDA: m2, RIGA_DIFF: diff}


def costruisci_tabella(per_file_raw: dict) -> pd.DataFrame:
    """Una colonna per file, una riga per metrica."""
    colonne = {nome: riepilogo_fc(raw) for nome, raw in per_file_raw.items()}
    tab = pd.DataFrame(colonne, index=RIGHE)
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


def render(per_file_raw: dict) -> None:
    st.header("❤️ Riepilogo frequenza cardiaca")

    if not per_file_raw:
        st.info("Carica uno o più file .fit dalla sidebar per eseguire questa analisi.")
        return

    tab = costruisci_tabella(per_file_raw)
    st.dataframe(stile_tabella(tab), use_container_width=True)
    st.caption(
        "Differenza = media della seconda metà rispetto alla prima metà. "
        f"Verde se entro ±{SOGLIA_DIFF:.0f}%, rosso se oltre. "
        "“—” = nessun dato di frequenza cardiaca nel file."
    )
    st.download_button(
        "📥 Scarica riepilogo FC (CSV)",
        data=tab.round(1).to_csv().encode("utf-8"),
        file_name="riepilogo_fc.csv",
        mime="text/csv",
    )