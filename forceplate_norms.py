# -*- coding: utf-8 -*-
"""
Norme di popolazione Force Plate — modulo per AMBRO BIG DATA BROTHER

Legge due tipi di file, anche insieme:
  - gli export completi della Force Plate app (forceplate_fulldata_*.csv,
    una riga per ripetizione): ogni CSV è già una seduta;
  - gli XLSX grezzi di ForceMate (un esercizio o più per file): i file dello
    stesso atleta nello stesso giorno vengono uniti in una seduta.
Calcola media e deviazione standard per sesso di ogni metrica, restituendo
testo pronto da incollare nel codice:

  un unico dict DEFAULT_POP: prima le 13 costanti già usate dalla Force
  Plate app (stesse chiavi), poi un divisore, poi tutte le altre metriche
  registrate con chiave "jump_type::variabile".

Scelte di metodo:
  - Unità di osservazione = una seduta (un CSV, oppure tutti gli XLSX dello
    stesso atleta nello stesso giorno). Più sedute dello stesso
    atleta contano tutte. Ogni seduta viene prima ridotta alla media delle
    sue ripetizioni, poi si calcolano media e SD fra le sedute.
  - SD campionaria (ddof=1).
  - Se un sesso ha meno di N sedute: per DEFAULT_POP resta il valore di
    default attuale, per le altre metriche (che un default non ce l'hanno) None.
  - La stessa variabile grezza (es. "peak force") esiste in test diversi con
    significati diversi: la chiave è sempre tipo di test + variabile.
"""

from __future__ import annotations

import datetime as dt
import io
import math
import re

import openpyxl

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

# Metriche derivate per ripetizione della Force Plate app (campo "derive"
# in METRICS): rapporto calcolato su ogni ripetizione, poi media di seduta.
# Chiave = stessa chiave di METRICS, così potrà diventare un pop_key.
DERIVED_PER_REP = {
    "imtp_rel_peak_force": ("imtp", "peak force", "body mass"),
    "sj_net_rel_impulse":  ("sj", "net impulse", "body mass"),
    "cmj_net_rel_impulse": ("cmj", "net impulse", "body mass"),
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


def _parse_day(raw):
    """Data del test come datetime.date, da qualunque formato visto finora
    (ForceMate '11.02.2026 15:31:57', CSV '03/10/2026', ISO)."""
    if raw is None:
        return None
    if isinstance(raw, dt.datetime):
        return raw.date()
    if isinstance(raw, dt.date):
        return raw
    txt = str(raw).strip()
    for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(txt, fmt).date()
        except ValueError:
            continue
    return None


def _day_from_filename(filename):
    """ForceMate nomina i file Nome_Cognome_ddmmyy-hh_mm_ss_TIPO.xlsx."""
    m = re.search(r"_(\d{2})(\d{2})(\d{2})-\d{2}_\d{2}_\d{2}", filename)
    if not m:
        return None
    try:
        return dt.date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _name_from_filename(filename):
    m = re.match(r"(.+?)_\d{6}-\d{2}_\d{2}_\d{2}", filename)
    return m.group(1).replace("_", " ").strip() if m else None


def _guess_type_from_filename(filename):
    """Stessa regola della Force Plate app: usata solo per le ripetizioni
    senza tag 'jump type'."""
    fn = filename.upper()
    for t in ("CMRJ", "IMTP", "CMJ", "SJ", "DJ"):
        if t in fn:
            return t.lower()
    return None


def _to_num(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip().replace(",", "."))
        except ValueError:
            return None
    return None


def read_csv_record(file_bytes, filename):
    """Un export completo (CSV) -> record con le ripetizioni in un DataFrame
    ('Jump Type' + una colonna per variabile)."""
    df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
    if "Jump Type" not in df.columns:
        raise ValueError("manca la colonna 'Jump Type' (non è un export completo)")
    data_label = _first(df, "Data test")
    reps = df[["Jump Type"] + [c for c in df.columns if c not in META_COLS]]
    return dict(kind="csv", file=filename, nome=_first(df, "Nome"),
                sesso=_sex_code(_first(df, "Sesso")),
                giorno=_parse_day(data_label), data_label=data_label or "—", reps=reps)


def read_xlsx_record(file_bytes, filename):
    """Un XLSX ForceMate -> record nello stesso formato del CSV. Stessa
    lettura della Force Plate app: metadati nelle colonne A-F, unità in G,
    blocchi 'Jump N' (variabile, valore, spacer) da H in poi; il tipo di
    test è per ripetizione. Gestisce anche il layout IMTP 'Unit / Trial N'."""
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
    rows = [tuple(r) for r in wb.worksheets[0].iter_rows(values_only=True)]
    wb.close()
    if not rows:
        raise ValueError("file vuoto")
    max_col = max(len(r) for r in rows)

    def cell(r_idx, c_idx):  # 1-based come in Excel
        row = rows[r_idx - 1] if 0 < r_idx <= len(rows) else ()
        return row[c_idx - 1] if 0 < c_idx <= len(row) else None

    def header_starts(prefix, first_col):
        return [c for c in range(first_col, max_col + 1)
                if cell(1, c) and str(cell(1, c)).strip().lower().startswith(prefix)]

    imtp_layout = (str(cell(1, 7) or "").strip().lower() == "unit"
                   and bool(header_starts("trial", 8)))

    if imtp_layout:
        # Nessun metadato nel file: nome e data dal nome del file, sesso
        # da assegnare a mano; peso da 'system weight' come nella Force Plate app.
        meta = dict(nome=_name_from_filename(filename), sesso=None,
                    giorno=_day_from_filename(filename))
        blocks, unit_col, fallback_type = header_starts("trial", 2), 1, "imtp"
    else:
        def kv(col_key, col_val):
            out = {}
            for r in range(1, 16):
                k = cell(r, col_key)
                if k:
                    out[str(k).strip().lower()] = cell(r, col_val)
            return out
        gen, attr = kv(1, 2), kv(3, 4)
        nome = str(attr.get("name") or gen.get("name") or "").strip() or _name_from_filename(filename)
        meta = dict(nome=nome, sesso=_sex_code(attr.get("gender")),
                    giorno=_parse_day(gen.get("date")) or _day_from_filename(filename))
        blocks, unit_col, fallback_type = header_starts("jump", 8), 7, _guess_type_from_filename(filename)

    reps = []
    for bs in blocks:
        var_col, val_col = bs, bs + 1
        rep, jump_type = {}, None
        for r in range(2, len(rows) + 1):
            var = cell(r, var_col)
            if not var:
                continue
            var = str(var).strip().lower()
            val = cell(r, val_col)
            if var == "jump type":
                jump_type = str(val).strip().lower() if val else None
                continue
            if var == "tags":
                continue
            num = _to_num(val)
            if num is not None:
                rep[var] = num
        if imtp_layout and "body mass" not in rep and rep.get("system weight"):
            rep["body mass"] = rep["system weight"] / 9.80665
        if rep:
            rep["Jump Type"] = jump_type or fallback_type
            reps.append(rep)

    if not reps:
        raise ValueError("nessuna ripetizione con valori numerici")
    return dict(kind="xlsx", file=filename, nome=meta["nome"], sesso=meta["sesso"],
                giorno=meta["giorno"],
                data_label=meta["giorno"].strftime("%d/%m/%Y") if meta["giorno"] else "—",
                reps=pd.DataFrame(reps))


def _norm_name(nome):
    return re.sub(r"\s+", " ", str(nome or "").strip().lower().rstrip("."))


def build_sessions(records, sex_override):
    """Raggruppa i record in sedute e calcola i valori di seduta.
    - CSV: un file = una seduta.
    - XLSX: stesso atleta + stesso giorno = una seduta.
    Il sesso viene dal file; se manca, da un altro file dello stesso atleta;
    se manca ancora, da sex_override {nome normalizzato: 'M'/'F'}.
    Ritorna (sedute, avvisi)."""
    warnings = []

    # Sesso noto per atleta, da qualunque file
    known_sex = {}
    for r in records:
        if r["sesso"]:
            known_sex.setdefault(_norm_name(r["nome"]), r["sesso"])

    groups = {}
    for i, r in enumerate(records):
        key = (_norm_name(r["nome"]), r["giorno"])
        if r["kind"] == "csv" or r["giorno"] is None:
            key = key + (i,)  # CSV o data ignota: seduta a sé
        elif any(x["file"] == r["file"] for x in groups.get(key, [])):
            warnings.append(f"{r['file']}: caricato due volte. Copia esclusa.")
            continue
        groups.setdefault(key, []).append(r)

    sessions, seen = [], {}
    for key, recs in groups.items():
        nome_n, giorno = key[0], key[1]
        nome = recs[0]["nome"] or "—"
        files = ", ".join(r["file"] for r in recs)
        is_csv = recs[0]["kind"] == "csv"

        # Doppione: stesso atleta e giorno già coperto (es. CSV + i suoi XLSX)
        if giorno is not None:
            sig = (nome_n, giorno)
            if sig in seen:
                warnings.append(f"{files}: stesso atleta e stesso giorno di "
                                f"{seen[sig]}, probabilmente un doppione. Escluso.")
                continue
            seen[sig] = files

        sex = next((r["sesso"] for r in recs if r["sesso"]), None) \
            or known_sex.get(nome_n) or sex_override.get(nome_n)
        if sex is None:
            continue  # segnalato nella UI come 'sesso da assegnare'

        df = pd.concat([r["reps"] for r in recs], ignore_index=True, sort=False)
        df["Jump Type"] = df["Jump Type"].astype(str).str.strip().str.lower()
        df = df[~df["Jump Type"].isin(["", "—", "-", "nan", "none"])]
        if df.empty:
            warnings.append(f"{files}: nessuna ripetizione con un tipo di test valido. Escluso.")
            continue
        var_cols = [c for c in df.columns if c not in META_COLS]
        values = session_values(df, var_cols)
        if not values:
            warnings.append(f"{files}: nessun valore numerico. Escluso.")
            continue

        sessions.append(dict(file=files, fonte="CSV" if is_csv else "XLSX", nome=nome,
                             data=recs[0]["data_label"], sesso=sex,
                             tests=sorted(df["Jump Type"].unique()), values=values))
    return sessions, warnings


def athletes_without_sex(records):
    """Atleti per cui nessun file indica il sesso: {nome normalizzato: nome}."""
    known = {_norm_name(r["nome"]) for r in records if r["sesso"]}
    out = {}
    for r in records:
        n = _norm_name(r["nome"])
        if n not in known:
            out.setdefault(n, r["nome"] or "—")
    return out


def session_values(df, var_cols):
    """Media per ripetizione di ogni variabile, per tipo di test, più le tre
    derivate che servono a DEFAULT_POP."""
    out = {}
    for jt, g in df.groupby("Jump Type"):
        num = g[var_cols].apply(pd.to_numeric, errors="coerce")
        for var, mean in num.mean(skipna=True).items():
            if pd.notna(mean):
                out[(jt, var)] = float(mean)

        # Rapporti per ripetizione, poi media (come il derive della Force
        # Plate app, non rapporto delle medie).
        for key, (d_jt, num_var, den_var) in DERIVED_PER_REP.items():
            if jt != d_jt or not {num_var, den_var} <= set(num.columns):
                continue
            ratio = pd.to_numeric(
                (num[num_var] / num[den_var]).replace([math.inf, -math.inf], pd.NA),
                errors="coerce").dropna()
            if len(ratio):
                out[("derived", key)] = float(ratio.mean())

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
    raw_keys = sorted({k for s in sessions for k in s["values"]
                       if k[0] != "derived" and k not in used})
    # Derivate non presenti fra le 13 costanti (es. net rel impulse): in
    # testa, con la chiave di METRICS.
    derived_keys = [("derived", k) for k in DERIVED_PER_REP if k not in CURRENT_DEFAULT_POP]
    keys = derived_keys + raw_keys
    rows = []
    for jt, var in keys:
        st_ = stats_by_sex(sessions, (jt, var))
        row = dict(key=var if jt == "derived" else f"{jt}::{var}")
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
        lines.append("    # ALTRE METRICHE — derivate (chiave di METRICS), poi grezze")
        lines.append("    # con chiave 'tipo di test::variabile'.")
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
        "Carica gli export completi della Force Plate app (forceplate_fulldata_...csv) e/o "
        "gli XLSX grezzi di ForceMate, anche di un solo esercizio. Un CSV è una seduta; "
        "gli XLSX dello stesso atleta nello stesso giorno vengono uniti in una seduta. "
        "Ogni seduta conta come un'osservazione, anche se lo stesso atleta compare più "
        "volte. Per ogni seduta si fa prima la media delle ripetizioni, poi media e "
        "deviazione standard fra le sedute, separatamente per uomini e donne."
    )

    files = st.file_uploader("CSV e/o XLSX delle sedute", type=["csv", "xlsx"],
                             accept_multiple_files=True, key="fpn_files")
    min_n = st.number_input(
        "Sedute minime per sesso", min_value=2, value=DEFAULT_MIN_N, step=1, key="fpn_min_n",
        help="Sotto questa soglia, per le 13 costanti attuali resta il valore di default; "
             "per le altre metriche il valore è None.",
    )

    if not files:
        st.info("Carica almeno un file per iniziare.")
        return

    records = []
    for f in files:
        try:
            reader = read_xlsx_record if f.name.lower().endswith(".xlsx") else read_csv_record
            records.append(reader(f.getvalue(), f.name))
        except Exception as e:
            st.warning(f"⚠️ {f.name}: {e}. File escluso.")

    if not records:
        st.error("Nessun file utilizzabile.")
        return

    # Sesso mancante: lista degli atleti sconosciuti, assegnazione nell'app
    missing = athletes_without_sex(records)
    sex_override = {}
    if missing:
        st.subheader("Sesso da assegnare")
        slot = st.empty()  # riempito dopo il ciclo con il conteggio aggiornato
        h1, h2, h3 = st.columns([3, 4, 2])
        h1.markdown("**Atleta**")
        h2.markdown("**File / date**")
        h3.markdown("**Sesso**")
        for n_key, nome in sorted(missing.items(), key=lambda kv: kv[1]):
            recs = [r for r in records if _norm_name(r["nome"]) == n_key]
            c1, c2, c3 = st.columns([3, 4, 2])
            c1.write(nome)
            c2.caption("  \n".join(f"{r['file']} — {r['data_label']}" for r in recs))
            scelta = c3.radio(nome, ["—", "M", "F"], horizontal=True,
                              key=f"fpn_sex_{n_key}", label_visibility="collapsed")
            if scelta in ("M", "F"):
                sex_override[n_key] = scelta
        rimasti = len(missing) - len(sex_override)
        if rimasti:
            slot.warning(f"{rimasti} atleti su {len(missing)} senza sesso: le loro sedute "
                         "non vengono contate finché non lo assegni.")
        else:
            slot.success(f"Sesso assegnato a tutti i {len(missing)} atleti.")
        st.divider()

    sessions, warns = build_sessions(records, sex_override)
    for w in warns:
        st.warning(f"⚠️ {w}")

    if not sessions:
        st.error("Nessuna seduta utilizzabile: controlla il sesso degli atleti qui sopra.")
        return

    n_m = sum(s["sesso"] == "M" for s in sessions)
    n_f = sum(s["sesso"] == "F" for s in sessions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Sedute", len(sessions))
    c2.metric("Uomini", n_m)
    c3.metric("Donne", n_f)

    with st.expander("Sedute caricate"):
        st.dataframe(pd.DataFrame([
            dict(Atleta=s["nome"], Sesso=s["sesso"], Data=s["data"], Fonte=s["fonte"],
                 Test=", ".join(s["tests"]), File=s["file"]) for s in sessions
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