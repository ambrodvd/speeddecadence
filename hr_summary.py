"""Riepilogo frequenza cardiaca per file .fit.

Per ogni file: media FC totale, media della prima e della seconda metà
(divise a metà della durata) e differenza % seconda vs prima metà.
Sotto, statistiche descrittive sull'insieme dei file (stesso schema
della tab Efficiency Factor: un file = un'osservazione).

Uso nel main app:
    from hr_summary import render as render_hr_summary
    render_hr_summary(fc_raw)   # dict nome_file -> dataframe con timestamp, hr
"""
import numpy as np
import pandas as pd
import streamlit as st

SOGLIA_DIFF = 5.0  # |differenza %| oltre cui la cella diventa rossa

RIGA_MEDIA = "Media totale"
RIGA_PRIMA = "Prima metà"
RIGA_SECONDA = "Seconda metà"
RIGA_DIFF = "Differenza"
RIGHE = [RIGA_MEDIA, RIGA_PRIMA, RIGA_SECONDA, RIGA_DIFF]

# Nomi delle metriche nella tabella statistiche (con unità)
NOMI_STAT = {
    RIGA_MEDIA: "FC media totale (bpm)",
    RIGA_PRIMA: "FC 1ª metà (bpm)",
    RIGA_SECONDA: "FC 2ª metà (bpm)",
    RIGA_DIFF: "Differenza (%)",
}


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


def statistiche_insieme(tab: pd.DataFrame) -> pd.DataFrame:
    """Statistiche descrittive per metrica, un file = un'osservazione.
    Stesse colonne della tab Efficiency Factor. La moda si calcola sui valori
    arrotondati a 0.1 e si riporta solo se almeno due file coincidono."""
    rows = []
    for metrica in RIGHE:
        s = tab.loc[metrica].dropna()
        if s.empty:
            continue

        counts = s.round(1).value_counts()
        moda = float(counts.index[0]) if len(counts) and counts.iloc[0] > 1 else np.nan

        sd = float(s.std(ddof=1)) if len(s) > 1 else np.nan
        mean = float(s.mean())
        rows.append({
            "Metrica": NOMI_STAT[metrica],
            "N": int(s.count()),
            "Min": float(s.min()),
            "Max": float(s.max()),
            "Range": float(s.max() - s.min()),
            "Media": mean,
            "Mediana": float(s.median()),
            "Moda (arr. 0.1)": moda,
            "Dev. std": sd,
            "CV %": (sd / mean * 100.0) if (np.isfinite(sd) and mean != 0) else np.nan,
            "p25": float(s.quantile(0.25)),
            "p75": float(s.quantile(0.75)),
            "IQR": float(s.quantile(0.75) - s.quantile(0.25)),
        })
    return pd.DataFrame(rows)


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

    # --- 1) tabella per file ---
    st.subheader("1. Riepilogo per file")
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

    # --- 2) statistiche sull'insieme dei file ---
    st.divider()
    st.subheader("2. Statistiche sull'insieme dei file")
    stats_df = statistiche_insieme(tab)
    if stats_df.empty:
        st.warning("Nessun file con dati di frequenza cardiaca: nessuna statistica da calcolare.")
        return

    n_con_fc = int(tab.loc[RIGA_MEDIA].notna().sum())
    n_senza_fc = len(tab.columns) - n_con_fc
    st.caption(
        f"**{n_con_fc} file nelle statistiche**"
        + (f" ({n_senza_fc} senza FC, esclusi)" if n_senza_fc else "")
        + ". Un file = un'osservazione. La moda è calcolata sui valori "
        "arrotondati a 0.1 e mostrata solo se almeno due file cadono sullo "
        "stesso valore: su misure continue una moda \"vera\" non esiste quasi mai."
    )
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Scarica statistiche FC (CSV)",
        data=stats_df.to_csv(index=False).encode("utf-8"),
        file_name="statistiche_fc.csv",
        mime="text/csv",
    )