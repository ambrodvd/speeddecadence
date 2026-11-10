"""EF a blocchi: metodo vecchio del Race Analyzer (d/t) vs metodo "moto".
Il calcolo sta in ef_core.py; qui solo tabella, grafico e UI."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ef_core import (EF_BLOCK_MIN_DEFAULT, MOVING_MIN_KMH_DEFAULT, MOVING_WIN_S,
                     compute_block_table, read_hr_zones_csv)

EF_BLOCK_MIN = EF_BLOCK_MIN_DEFAULT

# (colonna EF, colore, etichetta)
EF_METHOD_LINES = [
    ("EF Race Analyzer", "#e07b39", "Race Analyzer (d/t)"),
    ("EF moto", "#2a9d8f", "moto"),
]


def build_block_ef_figure(tbl, title):
    """Una linea per metodo. In x il centro del blocco in ore."""
    fig = go.Figure()
    for col, color, label in EF_METHOD_LINES:
        y = pd.to_numeric(tbl[col], errors="coerce")
        if not y.notna().any():
            continue
        fig.add_trace(go.Scatter(
            x=tbl["_mid_h"], y=y, mode="lines+markers", name=label,
            line=dict(color=color, width=2.4), marker=dict(size=7),
            customdata=tbl["Blocco"],
            hovertemplate="%{customdata}<br>EF: %{y:.2f}<extra>" + label + "</extra>",
        ))
    if not fig.data:
        return None
    fig.update_layout(
        title=title,
        xaxis=dict(title="Tempo trascorso (ore, centro del blocco)", rangemode="tozero"),
        yaxis=dict(title="EF (km/h a soglia)"),
        height=440, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=90),
    )
    return fig


def render_ef_blocks(per_file_raw):
    st.header(f"⏱️ EF a blocchi di {EF_BLOCK_MIN} minuti: Race Analyzer vs moto")
    st.caption(
        "EF = EFS ÷ (FC / soglia). **Race Analyzer (d/t)** = calcolo precedente del DU "
        "COACHING RACE Analyzer: EFD del blocco ÷ tempo del blocco, FC media dei "
        "record, soste incluse. **moto** = stessa distanza ed EFD, ma solo sui "
        "secondi in movimento, con la FC media sugli stessi secondi. Blocchi "
        f"consecutivi da {EF_BLOCK_MIN}'; l'ultimo si tiene se copre almeno metà "
        "blocco. Smoothing quota e passo di ricampionamento sono quelli fissi del "
        "Race Analyzer (9 punti, 20 m), non quelli della sidebar."
    )

    if not per_file_raw:
        st.info("Carica uno o più file .fit dalla sidebar per eseguire questa analisi.")
        return

    moving_min_kmh = st.slider(
        "Soglia di movimento (km/h): sotto questa velocità GPS l'atleta è considerato fermo",
        0.5, 3.0, MOVING_MIN_KMH_DEFAULT, step=0.25, key="blk_moving_kmh",
        help=f"Spostamento in linea d'aria su {MOVING_WIN_S} s dalla traccia GPS grezza. "
             "Troppo bassa: il jitter GPS da fermo conta come moto. "
             "Troppo alta: le salite ripidissime camminate vengono scambiate per soste.",
    )

    st.subheader("1. Zone FC per file")
    zones_by_file = {}
    for fname in per_file_raw:
        c_name, c_up = st.columns([1, 1.3])
        c_name.markdown(f"**{fname}**")
        csv_file = c_up.file_uploader(
            f"CSV zone FC — {fname}", type=["csv"],
            key=f"blk_zones_csv_{fname}", label_visibility="collapsed",
        )
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

    st.subheader("2. Tabelle e grafici per file")
    int_cols = ["D+ (m)", "Tempo fermo (s)", "FC media RA (bpm)", "FC media moto (bpm)",
                "% soglia RA", "% soglia moto"]
    export = []
    for fname, zones in zones_by_file.items():
        st.divider()
        st.markdown(f"#### {fname}")

        tbl, info = compute_block_table(per_file_raw[fname], zones["threshold"],
                                        moving_min_kmh=moving_min_kmh)
        if tbl is None:
            st.warning(f"⚠️ {fname}: {info}, file saltato.")
            continue

        st.caption(f"soglia FC **{zones['threshold']:.0f} bpm**"
                   + (f" · atleta **{zones['athlete']}**" if zones.get("athlete") else "")
                   + f" · distanza: **{info}**"
                   + (" ⚠️ il file non ha la distanza del dispositivo: i numeri "
                      "possono differire da quelli del Race Analyzer"
                      if info != "dispositivo" else ""))

        shown = tbl.drop(columns=["_mid_h"])
        fmt = {c: "{:.0f}" for c in int_cols}
        fmt.update({c: "{:.2f}" for c in shown.columns
                    if c not in int_cols and c != "Blocco"})
        fmt["Δ EF (moto − RA)"] = "{:+.2f}"
        st.dataframe(shown.style.format(fmt, na_rep="—"),
                     use_container_width=True, hide_index=True)

        fig = build_block_ef_figure(tbl, f"{fname} — EF per blocco")
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, key=f"blk_chart_{fname}")

        export.append(shown.assign(File=fname, **{"Soglia (bpm)": zones["threshold"],
                                                  "Soglia moto (km/h)": moving_min_kmh,
                                                  "Fonte distanza": info}))

    if export:
        export_df = pd.concat(export, ignore_index=True)
        lead = ["File", "Soglia (bpm)", "Soglia moto (km/h)", "Fonte distanza"]
        export_df = export_df[lead + [c for c in export_df.columns if c not in lead]]
        st.download_button(
            "📥 Scarica tabelle a blocchi, tutti i file (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="ef_blocchi_ra_vs_moto.csv",
            mime="text/csv",
        )