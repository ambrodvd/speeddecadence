"""Confronto jump height FT (flight time) vs NI (net impulse) dagli export
ForceMate (.xlsx, foglio "Jump results").

Layout dell'export: riga 0 contiene "Jump 1", "Jump 2", ... ; per ogni salto
la colonna dell'intestazione contiene le etichette delle metriche e quella
subito a destra i valori. I metadati (Name, Date) stanno nelle colonne 0-1.

FT = g·t²/8 (dal tempo di volo), NI = v_takeoff²/(2g) (dall'impulso netto).
"""
import io
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy import stats as _sps
except ImportError:  # p-value opzionale
    _sps = None

FT_LABEL = "jump height ft"
NI_LABEL = "jump height ni"
COLORS = {"FT": "#1f4e79", "NI": "#c1440e"}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Lettura export force plate...")
def parse_forcemate_xlsx(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    df = pd.read_excel(io.BytesIO(file_bytes), header=None)

    def meta(key):
        hit = df[df[0].astype(str).str.strip().str.lower() == key.lower()]
        return str(hit.iat[0, 1]).strip() if not hit.empty else "—"

    athlete, date = meta("Name"), meta("Date")

    jump_cols = [c for c in df.columns
                 if re.fullmatch(r"Jump \d+", str(df.iat[0, c]).strip())]
    rows = []
    for c in jump_cols:
        if c + 1 not in df.columns:
            continue
        labels = df[c].astype(str).str.strip().str.lower()
        values = df[c + 1]

        def get(label):
            m = labels == label
            return values[m].iloc[0] if m.any() else np.nan

        ft = pd.to_numeric(get(FT_LABEL), errors="coerce")
        ni = pd.to_numeric(get(NI_LABEL), errors="coerce")
        jtype = get("jump type")
        rows.append({
            "File": file_name,
            "Atleta": athlete,
            "Data": date,
            "Test": str(jtype).upper() if isinstance(jtype, str) else "—",
            "Salto": int(str(df.iat[0, c]).split()[-1]),
            "FT (cm)": float(ft) if np.isfinite(ft) else np.nan,
            "NI (cm)": float(ni) if np.isfinite(ni) else np.nan,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------
def describe_methods(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    groups = [("Tutti", df)] if by is None else list(df.groupby(by))
    out = []
    for gname, g in groups:
        for method in ("FT", "NI"):
            s = g[f"{method} (cm)"].dropna()
            if s.empty:
                continue
            sd = s.std(ddof=1) if len(s) > 1 else np.nan
            out.append({
                (by or "Gruppo"): gname, "Metodo": method, "N": len(s),
                "Media": s.mean(), "Mediana": s.median(), "Dev. std": sd,
                "CV %": sd / s.mean() * 100 if s.mean() else np.nan,
                "Min": s.min(), "Max": s.max(),
            })
    return pd.DataFrame(out)


def agreement(df: pd.DataFrame) -> dict:
    """Bland-Altman + correlazione + Lin CCC sulle coppie FT/NI."""
    d = df.dropna(subset=["FT (cm)", "NI (cm)"])
    ft, ni = d["FT (cm)"].to_numpy(), d["NI (cm)"].to_numpy()
    n = len(d)
    if n < 3:
        return {"N": n}
    diff = ft - ni
    bias, sd_diff = diff.mean(), diff.std(ddof=1)
    r = np.corrcoef(ft, ni)[0, 1]
    # Lin's concordance correlation coefficient
    ccc = (2 * np.cov(ft, ni, ddof=1)[0, 1]
           / (ft.var(ddof=1) + ni.var(ddof=1) + (ft.mean() - ni.mean()) ** 2))
    p = _sps.ttest_rel(ft, ni).pvalue if _sps is not None else np.nan
    return {
        "N": n, "Bias (FT−NI, cm)": bias, "SD differenze (cm)": sd_diff,
        "LoA inf (cm)": bias - 1.96 * sd_diff, "LoA sup (cm)": bias + 1.96 * sd_diff,
        "Diff. assoluta media (cm)": np.abs(diff).mean(),
        "r di Pearson": r, "Lin CCC": ccc, "p (t appaiato)": p,
    }


def within_session_cv(df: pd.DataFrame) -> pd.DataFrame:
    """CV intra-sessione per file e metodo: quale misura è più ripetibile."""
    rows = []
    for fname, g in df.groupby("File", sort=False):
        row = {"File": fname, "Atleta": g["Atleta"].iloc[0],
               "Test": g["Test"].iloc[0], "N salti": len(g)}
        for m in ("FT", "NI"):
            s = g[f"{m} (cm)"].dropna()
            row[f"CV {m} %"] = (s.std(ddof=1) / s.mean() * 100
                                if len(s) > 1 and s.mean() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Grafici
# ---------------------------------------------------------------------------
def fig_per_file(g: pd.DataFrame, title: str) -> go.Figure:
    x = [f"Salto {i}" for i in g["Salto"]]
    fig = go.Figure()
    for m in ("FT", "NI"):
        fig.add_trace(go.Bar(
            x=x, y=g[f"{m} (cm)"], name=m, marker_color=COLORS[m],
            text=[f"{v:.1f}" for v in g[f"{m} (cm)"]], textposition="outside",
        ))
    fig.update_layout(title=title, barmode="group", yaxis_title="Jump height (cm)",
                      height=360, margin=dict(t=60, b=30))
    return fig


def fig_scatter(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["FT (cm)", "NI (cm)"])
    lo = float(min(d["FT (cm)"].min(), d["NI (cm)"].min())) - 2
    hi = float(max(d["FT (cm)"].max(), d["NI (cm)"].max())) + 2
    fig = go.Figure()
    for test, g in d.groupby("Test"):
        fig.add_trace(go.Scatter(
            x=g["FT (cm)"], y=g["NI (cm)"], mode="markers", name=test,
            marker=dict(size=10),
            customdata=np.stack([g["Atleta"], g["Salto"]], axis=1),
            hovertemplate="%{customdata[0]} · salto %{customdata[1]}<br>"
                          "FT %{x:.1f} cm · NI %{y:.1f} cm<extra></extra>",
        ))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             name="Identità", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="FT vs NI (linea = accordo perfetto)",
                      xaxis_title="FT (cm)", yaxis_title="NI (cm)",
                      xaxis_range=[lo, hi], yaxis_range=[lo, hi], height=450)
    return fig


def fig_bland_altman(df: pd.DataFrame, ag: dict) -> go.Figure:
    d = df.dropna(subset=["FT (cm)", "NI (cm)"])
    mean = (d["FT (cm)"] + d["NI (cm)"]) / 2
    diff = d["FT (cm)"] - d["NI (cm)"]
    fig = go.Figure(go.Scatter(
        x=mean, y=diff, mode="markers", marker=dict(size=10, color="#2a9d8f"),
        customdata=np.stack([d["Atleta"], d["Test"], d["Salto"]], axis=1),
        hovertemplate="%{customdata[0]} · %{customdata[1]} salto %{customdata[2]}"
                      "<br>media %{x:.1f} · diff %{y:+.1f} cm<extra></extra>",
        showlegend=False,
    ))
    for y, lab, dash in [(ag["Bias (FT−NI, cm)"], "bias", "solid"),
                         (ag["LoA sup (cm)"], "+1.96 SD", "dash"),
                         (ag["LoA inf (cm)"], "−1.96 SD", "dash")]:
        fig.add_hline(y=y, line_dash=dash, line_color="#333",
                      annotation_text=f"{lab}: {y:+.1f}", annotation_position="right")
    fig.update_layout(title="Bland-Altman (FT − NI)",
                      xaxis_title="Media FT/NI (cm)", yaxis_title="FT − NI (cm)",
                      height=420, margin=dict(r=110))
    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_jump_height_compare():
    st.header("🦘 Jump height: Flight Time vs Net Impulse")
    st.caption(
        "Export ForceMate (.xlsx). **FT** = g·t²/8 dal tempo di volo; "
        "**NI** = v_takeoff²/(2g) dall'impulso netto. Per i CMRJ si usa il primo "
        "salto (non il rebound)."
    )

    uploads = st.file_uploader("Carica uno o più export force plate (.xlsx)",
                               type=["xlsx"], accept_multiple_files=True,
                               key="jh_compare_files")
    if not uploads:
        st.info("Carica almeno un file .xlsx per procedere.")
        return

    frames = []
    for f in uploads:
        try:
            jdf = parse_forcemate_xlsx(f.getvalue(), f.name)
        except ImportError as e:
            st.error(f"⚠️ Dipendenza mancante per leggere gli .xlsx: {e}. "
                     "Esegui `uv add openpyxl` e riavvia l'app.")
            return
        except Exception as e:
            st.warning(f"⚠️ {f.name}: file illeggibile ({e.__class__.__name__}: {e}), escluso.")
            continue
        if jdf.empty or jdf[["FT (cm)", "NI (cm)"]].isna().all().all():
            st.warning(f"⚠️ {f.name}: nessuna riga 'jump height ft/ni' trovata, escluso.")
            continue
        frames.append(jdf)
    if not frames:
        return

    data = pd.concat(frames, ignore_index=True)
    data["FT − NI (cm)"] = data["FT (cm)"] - data["NI (cm)"]
    data["FT − NI (%)"] = data["FT − NI (cm)"] / data["NI (cm)"] * 100

    # --- 1) misure per salto ---
    st.subheader("1. Misure per salto")
    for fname, g in data.groupby("File", sort=False):
        st.markdown(f"#### {g['Atleta'].iloc[0]} · {g['Test'].iloc[0]} · {g['Data'].iloc[0]}")
        st.caption(fname)
        c1, c2 = st.columns([1.3, 1])
        c1.plotly_chart(fig_per_file(g, "FT vs NI per salto"),
                        use_container_width=True, key=f"jh_bar_{fname}")
        c2.dataframe(
            g[["Salto", "FT (cm)", "NI (cm)", "FT − NI (cm)", "FT − NI (%)"]]
            .style.format({"FT (cm)": "{:.1f}", "NI (cm)": "{:.1f}",
                           "FT − NI (cm)": "{:+.1f}", "FT − NI (%)": "{:+.1f}"}),
            use_container_width=True, hide_index=True,
        )

    # --- 2) mash-up statistico ---
    st.divider()
    st.subheader("2. Statistiche su tutti i salti")

    ag = agreement(data)
    if ag.get("N", 0) < 3:
        st.info("Servono almeno 3 salti con entrambe le misure per le statistiche di accordo.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bias FT−NI", f"{ag['Bias (FT−NI, cm)']:+.1f} cm")
        m2.metric("Limiti di accordo 95%",
                  f"{ag['LoA inf (cm)']:+.1f} / {ag['LoA sup (cm)']:+.1f} cm")
        m3.metric("r di Pearson", f"{ag['r di Pearson']:.2f}")
        m4.metric("Lin CCC", f"{ag['Lin CCC']:.2f}")

        c1, c2 = st.columns(2)
        c1.plotly_chart(fig_scatter(data), use_container_width=True, key="jh_scatter")
        c2.plotly_chart(fig_bland_altman(data, ag), use_container_width=True, key="jh_ba")

        st.markdown("**Accordo tra metodi**")
        st.dataframe(pd.DataFrame([ag]), use_container_width=True, hide_index=True)
        st.caption(
            "Bias = differenza media FT−NI; LoA = bias ± 1.96·SD delle differenze. "
            "Lin CCC misura l'accordo (1 = identici), r solo la correlazione. "
            "I salti di uno stesso file non sono indipendenti: con pochi atleti "
            "p e LoA vanno letti come indicativi."
            + ("" if _sps else " (scipy non installato: p non calcolato)")
        )

        per_test = [(t, agreement(g)) for t, g in data.groupby("Test")]
        per_test = [{"Test": t, **a} for t, a in per_test if a.get("N", 0) >= 3]
        if len(per_test) > 1:
            st.markdown("**Accordo per tipo di test**")
            st.dataframe(pd.DataFrame(per_test), use_container_width=True, hide_index=True)

    st.markdown("**Descrittive per metodo**")
    st.dataframe(describe_methods(data), use_container_width=True, hide_index=True)
    if data["Test"].nunique() > 1:
        st.markdown("**Descrittive per metodo e tipo di test**")
        st.dataframe(describe_methods(data, by="Test"),
                     use_container_width=True, hide_index=True)

    st.markdown("**Ripetibilità intra-sessione (CV % per file)**")
    cv = within_session_cv(data)
    st.dataframe(cv, use_container_width=True, hide_index=True)
    st.caption(
        f"CV medio — FT: {cv['CV FT %'].mean():.1f}% · NI: {cv['CV NI %'].mean():.1f}%. "
        "A parità di atleta e sessione, il metodo con CV più basso è il più ripetibile."
    )

    st.download_button(
        "📥 Scarica tutti i salti (CSV)",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="jump_height_ft_vs_ni.csv", mime="text/csv",
    )