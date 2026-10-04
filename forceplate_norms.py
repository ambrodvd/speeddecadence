# -*- coding: utf-8 -*-
"""
Norme di popolazione Force Plate — modulo per AMBRO BIG DATA BROTHER

Legge gli export completi della Force Plate app (forceplate_fulldata_*.csv,
una riga per ripetizione) e calcola media e deviazione standard per sesso di
ogni metrica, restituendo testo pronto da incollare nel codice:

  un unico dict DEFAULT_POP: prima le 13 costanti già usate dalla Force
  Plate app (stesse chiavi), poi un divisore, poi tutte le altre metriche
  registrate con chiave "jump_type::variabile".

Scelte di metodo:
  - Unità di osservazione = una seduta (un CSV). Più sedute dello stesso
    atleta contano tutte. Ogni seduta viene prima ridotta alla media delle
    sue ripetizioni, poi si calcolano media e SD fra le sedute.
  - SD campionaria (ddof=1).
  - Se un sesso ha meno di N sedute: per DEFAULT_POP resta il valore di
    default attuale, per le altre metriche (che un default non ce l'hanno) None.
  - La stessa variabile grezza (es. "peak force") esiste in test diversi con
    significati diversi: la chiave è sempre tipo di test + variabile.
"""

from __future__ import annotations

import io
import math

import pandas as pd
import streamlit as st

# Copia dei default della Force Plate app: usati quando un sesso non ha
# abbastanza sedute. Se aggiorni DEFAULT_POP là, aggiornalo anche qui.
CURRENT_DEFAULT_POP = {
    "imtp_peak_force":        dict(mean_m=2606.8,      sd_m=646.1163194,  mean_f=1575.0,      sd_f=386.145544),
    "imtp_rel_peak_force":    dict(mean_m=34.56,        sd_m=5.27,         mean_f=34.56,        sd_f=5.27),
    "sj_mean_power":          dict(mean_m=1180.0,       sd_m=414.1638849,  mean_f=823.0,        sd_f=240.2447942),
    "sj_height":              dict(mean_m=26.9,         sd_m=6.57,         mean_f=19.35,        sd_f=5.51),
    "sj_contraction_time":    dict(mean_m=0.445,        sd_m=0.127835797,  mean_f=0.46,         sd_f=0.137032268),
    "cmj_height":             dict(mean_m=30.01,        sd_m=6.5,          mean_f=20.91,        sd_f=5.84),
    "mrsi_cmj":               dict(mean_m=0.419,        sd_m=0.098531539,  mean_f=0.308,        sd_f=0.093831019),
    "dsi":                    dict(mean_m=0.7,          sd_m=0.074074074,  mean_f=0.7,          sd_f=0.074074074),
    "eur":                    dict(mean_m=1.107708605,  sd_m=0.038034873,  mean_f=1.090563116,  sd_f=0.02656191),
    "cmj_re_rebound_height":  dict(mean_m=38.5,         sd_m=5.5996817,    mean_f=38.5,         sd_f=5.5996817),
    "cmj_re_contact_time":    dict(mean_m=0.25,         sd_m=0.148005087,  mean_f=0.25,         sd_f=0.148005087),
    "cmj_re_rebound_impulse": dict(mean_m=541.5,        sd_m=53.82161458,  mean_f=541.5,        sd_f=53.82161458),
    "mrsi_cmj_re":            dict(mean_m=1.37,         sd_m=0.331705729,  mean_f=1.37,         sd_f=0.331705729),
}

# Da dove viene ciascuna costante: (tipo di test, variabile grezza) per le
# metriche dirette. Le derivate (imtp_rel_peak_force, dsi, eur) sono
# calcolate in session_values() con le stesse formule della Force Plate app.
CURATED_RAW = {
    "imtp_peak_force":        ("imtp", "peak force"),
    "sj_mean_power":          ("sj", "avg. propulsive power"),
    "sj_height":              ("sj", "jump height ft"),
    "sj_contraction_time":    ("sj", "time to takeoff"),
    "cmj_height":             ("cmj", "jump height ft"),
    "mrsi_cmj":               ("cmj", "rsi modified"),
    "cmj_re_rebound_height":  ("cmrj", "rebound jump height ft"),
    "cmj_re_contact_time":    ("cmrj", "rebound contact time"),
    "cmj_re_rebound_impulse": ("cmrj", "rebound propulsive impulse"),
    "mrsi_cmj_re":            ("cmrj", "rebound rsi modified"),
}

META_COLS = {"Nome", "Sesso", "Data test", "File", "Jump Type", "Indice Ripetizione"}
DEFAULT_MIN_N = 5


# ---------------------------------------------------------------------------
# Lettura
# ---------------------------------------------------------------------------
def _sex_code(raw):
    """'M' / 'F' / None da ciò che c'è nella colonna Sesso."""
    s = str(raw or "").strip().upper()
    if s.startswith(("M", "U")):      # M, MALE, UOMO
        return "M"
    if s.startswith(("F", "D", "W")):  # F, FEMALE, DONNA, WOMAN
        return "F"
    return None


def _first(df, col):
    if col in df.columns and len(df):
        v = str(df[col].iloc[0]).strip()
        if v and v.lower() not in ("nan", "none", "-"):
            return v
    return None


def read_session(file_bytes, filename):
    """Un CSV -> dict con metadati e {(jump_type, var): media delle rep}.
    Solleva ValueError con un messaggio leggibile se il file non va."""
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    if "Jump Type" not in df.columns:
        raise ValueError("manca la colonna 'Jump Type' (non è un export completo)")

    sex = _sex_code(_first(df, "Sesso"))
    if sex is None:
        raise ValueError(f"sesso non riconosciuto ('{_first(df, 'Sesso')}')")

    df["Jump Type"] = df["Jump Type"].astype(str).str.strip().str.lower()
    df = df[~df["Jump Type"].isin(["", "—", "-", "nan", "none"])]
    if df.empty:
        raise ValueError("nessuna ripetizione con un tipo di test valido")

    var_cols = [c for c in df.columns if c not in META_COLS]
    values = session_values(df, var_cols)
    if not values:
        raise ValueError("nessun valore numerico trovato")

    return dict(file=filename, nome=_first(df, "Nome") or "—",
                data=_first(df, "Data test") or "—", sesso=sex,
                tests=sorted(df["Jump Type"].unique()), values=values)


def session_values(df, var_cols):
    """Media per ripetizione di ogni variabile, per tipo di test, più le tre
    derivate che servono a DEFAULT_POP."""
    out = {}
    for jt, g in df.groupby("Jump Type"):
        num = g[var_cols].apply(pd.to_numeric, errors="coerce")
        for var, mean in num.mean(skipna=True).items():
            if pd.notna(mean):
                out[(jt, var)] = float(mean)

        # IMTP Rel Peak Force: rapporto per ripetizione, poi media (come
        # il derive della Force Plate app, non rapporto delle medie).
        if jt == "imtp" and {"peak force", "body mass"} <= set(num.columns):
            rel = (num["peak force"] / num["body mass"]).replace([math.inf, -math.inf], pd.NA)
            rel = pd.to_numeric(rel, errors="coerce").dropna()
            if len(rel):
                out[("derived", "imtp_rel_peak_force")] = float(rel.mean())

    # Indici cross-test: rapporto delle medie di seduta, come build_results().
    cmj_pk, imtp_pk = out.get(("cmj", "peak propulsive force")), out.get(("imtp", "peak force"))
    cmj_h, sj_h = out.get(("cmj", "jump height ft")), out.get(("sj", "jump height ft"))
    if cmj_pk and imtp_pk:
        out[("derived", "dsi")] = cmj_pk / imtp_pk
    if cmj_h and sj_h:
        out[("derived", "eur")] = cmj_h / sj_h
    return out


# ---------------------------------------------------------------------------
# Statistiche
# ---------------------------------------------------------------------------
def _mean_sd(values):
    vals = [v for v in values if v is not None and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return 0, None, None
    m = sum(vals) / n
    if n < 2:
        return n, m, None
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return n, m, sd


def stats_by_sex(sessions, key):
    """{'M': (n, mean, sd), 'F': (n, mean, sd)} per una chiave (jt, var)."""
    return {sex: _mean_sd([s["values"].get(key) for s in sessions if s["sesso"] == sex])
            for sex in ("M", "F")}


def _curated_key(name):
    if name in CURATED_RAW:
        return CURATED_RAW[name]
    return ("derived", name)


def build_curated(sessions, min_n):
    """Righe per DEFAULT_POP: valore calcolato se n >= min_n, altrimenti
    il default attuale (per sesso, indipendentemente)."""
    rows = []
    for name, default in CURRENT_DEFAULT_POP.items():
        st_ = stats_by_sex(sessions, _curated_key(name))
        row = dict(key=name)
        for sex, suf in (("M", "m"), ("F", "f")):
            n, m, sd = st_[sex]
            ok = n >= min_n and m is not None and sd is not None
            row[f"n_{suf}"] = n
            row[f"mean_{suf}"] = m if ok else default[f"mean_{suf}"]
            row[f"sd_{suf}"] = sd if ok else default[f"sd_{suf}"]
            row[f"src_{suf}"] = "calcolato" if ok else "default"
        rows.append(row)
    return rows


def build_extra(sessions, min_n):
    """Righe per le altre metriche: tutte le variabili registrate tranne quelle già
    coperte da DEFAULT_POP. Sotto min_n il valore è None (nessun default)."""
    used = set(CURATED_RAW.values())
    keys = sorted({k for s in sessions for k in s["values"]
                   if k[0] != "derived" and k not in used})
    rows = []
    for jt, var in keys:
        st_ = stats_by_sex(sessions, (jt, var))
        row = dict(key=f"{jt}::{var}")
        for sex, suf in (("M", "m"), ("F", "f")):
            n, m, sd = st_[sex]
            ok = n >= min_n and m is not None and sd is not None
            row[f"n_{suf}"] = n
            row[f"mean_{suf}"] = m if ok else None
            row[f"sd_{suf}"] = sd if ok else None
        if row["n_m"] or row["n_f"]:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Testo per il codice
# ---------------------------------------------------------------------------
def _num(v):
    """Numero leggibile in Python: None resta None, altrimenti 10 cifre
    significative (stessa precisione delle costanti attuali)."""
    if v is None:
        return "None"
    return repr(float(f"{v:.10g}"))


def _format_line(r, key_w, comment):
    k = f'"{r["key"]}":'
    body = (f"dict(mean_m={_num(r['mean_m'])},".ljust(26)
            + f" sd_m={_num(r['sd_m'])},".ljust(21)
            + f" mean_f={_num(r['mean_f'])},".ljust(24)
            + f" sd_f={_num(r['sd_f'])})")
    return f"    {k:<{key_w}}{body},  # {comment}"


def format_pop_block(curated, extra):
    """Un unico dict DEFAULT_POP: prima le 13 costanti attuali, poi un
    divisore, poi tutte le altre metriche. Si copia e incolla in blocco al
    posto del DEFAULT_POP esistente: la Force Plate app legge solo le chiavi
    che conosce, le altre restano lì pronte per quando serviranno."""
    key_w = max(len(r["key"]) for r in curated + extra) + 4
    lines = ["DEFAULT_POP = {"]
    lines += [_format_line(r, key_w, _comment_curated(r)) for r in curated]
    if extra:
        lines.append("")
        lines.append("    # " + "=" * 70)
        lines.append("    # ALTRE METRICHE — chiave 'tipo di test::variabile'.")
        lines.append("    # None = troppe poche sedute per quel sesso (nessun default).")
        lines.append("    # " + "=" * 70)
        lines += [_format_line(r, key_w, _comment_extra(r)) for r in extra]
    lines.append("}")
    return "\n".join(lines)


def _comment_curated(r):
    def part(suf, lab):
        tag = "" if r[f"src_{suf}"] == "calcolato" else " default"
        return f"{lab} n={r[f'n_{suf}']}{tag}"
    return f"{part('m', 'M')}, {part('f', 'F')}"


def _comment_extra(r):
    return f"M n={r['n_m']}, F n={r['n_f']}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def render_forceplate_norms():
    st.header("🏋️ Norme di popolazione Force Plate")
    st.caption(
        "Carica gli export completi della Force Plate app (forceplate_fulldata_...csv). "
        "Ogni file è una seduta e conta come un'osservazione, anche se lo stesso atleta "
        "compare più volte. Per ogni seduta si fa prima la media delle ripetizioni, poi "
        "media e deviazione standard fra le sedute, separatamente per uomini e donne."
    )

    files = st.file_uploader("CSV delle sedute", type=["csv"],
                             accept_multiple_files=True, key="fpn_files")
    min_n = st.number_input(
        "Sedute minime per sesso", min_value=2, value=DEFAULT_MIN_N, step=1, key="fpn_min_n",
        help="Sotto questa soglia, per le 13 costanti attuali resta il valore di default; "
             "per le altre metriche il valore è None.",
    )

    if not files:
        st.info("Carica almeno un CSV per iniziare.")
        return

    sessions, seen = [], set()
    for f in files:
        try:
            s = read_session(f.getvalue(), f.name)
        except Exception as e:
            st.warning(f"⚠️ {f.name}: {e}. File escluso.")
            continue
        sig = (s["nome"], s["data"])
        if sig in seen:
            st.warning(f"⚠️ {f.name}: stessa persona e stessa data di un file già "
                       "caricato, probabilmente un doppione. File escluso.")
            continue
        seen.add(sig)
        sessions.append(s)

    if not sessions:
        st.error("Nessun file utilizzabile.")
        return

    n_m = sum(s["sesso"] == "M" for s in sessions)
    n_f = sum(s["sesso"] == "F" for s in sessions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sedute", len(sessions))
    c2.metric("Uomini", n_m)
    c3.metric("Donne", n_f)

    with st.expander("Sedute caricate"):
        st.dataframe(pd.DataFrame([
            dict(File=s["file"], Atleta=s["nome"], Sesso=s["sesso"], Data=s["data"],
                 Test=", ".join(s["tests"])) for s in sessions
        ]), use_container_width=True, hide_index=True)

    curated = build_curated(sessions, min_n)
    extra = build_extra(sessions, min_n)

    st.subheader("Riepilogo delle 13 costanti attuali")
    st.dataframe(pd.DataFrame([
        {"Costante": r["key"], "N uomini": r["n_m"], "Fonte uomini": r["src_m"],
         "Media uomini": r["mean_m"], "Dev.std uomini": r["sd_m"],
         "N donne": r["n_f"], "Fonte donne": r["src_f"],
         "Media donne": r["mean_f"], "Dev.std donne": r["sd_f"]} for r in curated
    ]), use_container_width=True, hide_index=True)

    st.subheader("DEFAULT_POP da incollare")
    st.caption(
        f"Un unico blocco: le 13 costanti attuali, un divisore, poi le altre "
        f"{len(extra)} metriche. Copialo con il pulsante in alto a destra e incollalo al "
        "posto del DEFAULT_POP della Force Plate app. Nel commento a fine riga: quante "
        "sedute per sesso, e 'default' dove il valore non è stato calcolato."
    )
    block = format_pop_block(curated, extra)
    st.code(block, language="python")

    st.download_button(
        "📥 Scarica il blocco (.py)",
        data=(block + "\n").encode("utf-8"),
        file_name="forceplate_population_norms.py", mime="text/x-python",
    )