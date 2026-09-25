"""Dualer Simplex und Neuoptimierung – nach einer Änderung von der alten Basis weiterrechnen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Lineare-Programmierung-Reihe der "Konzepte"-Reihe: Ändert sich die rechte Seite oder kommt eine Nebenbedingung dazu, bleibt die alte Endbasis dual zulässig, aber primal unzulässig - der duale Simplex
rechnet von dort weiter. Ändern sich Kosten oder kommt ein Dienst dazu, rechnet der primale Simplex von der alten Basis weiter. Die Demo zählt die Pivots gegen den Neustart von Null.

Lauffähig mit: streamlit run app.py
"""

import math

import pandas as pd
import streamlit as st

import dsx_constants as C
import dsx_evaluation as ev
import dsx_scenario as S
from dsx_evaluation import Settings, analyse
from dsx_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    reset_level,
    store_from_widget,
    sync_query_params,
)
from dsx_visualization import (
    build_chain,
    build_chain_hist,
    build_parametric,
    build_progress,
    build_ratios,
    build_sweep,
    build_warm_cold,
    tableau_frame,
)

st.set_page_config(page_title="Dualer Simplex – Sebastian Hanisch", layout="wide")


def num(x, digits=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{0.0 if abs(x) < 5e-13 else x:.{digits}f}"


STATUS_TEXT = {"optimal": "Optimum", "infeasible": "unzulässig", "unbounded": "unbeschränkt", "limit": "Pivot-Grenze erreicht"}
METHOD_TEXT = {"dual": "dualer Simplex", "primal": "primaler Simplex", "none": "keine Pivots nötig", "cold": "Rückfall auf Neustart"}

st.title("🔁 Dualer Simplex und Neuoptimierung – von der alten Basis weiterrechnen")
st.markdown(
    """
**Sechstes Stück der Lineare-Programmierung-Reihe.** Das Stück davor hat die Wertfunktion und den Zukauf durch **Neulösung von Null** gerechnet. Muss man nach einer Änderung wirklich neu anfangen? Nein: ändert sich die **rechte Seite** oder
kommt eine **Nebenbedingung** dazu (Schnitt, Branching-Schranke), bleibt die alte Endbasis **dual zulässig** - alle reduzierten Kosten sind noch ≥ 0 -, aber sie ist **primal unzulässig**: ein x_B ist negativ. Für genau diese Lage ist der **duale Simplex** gebaut (Lemke 1954).
Ändern sich **Deckungsbeiträge** oder kommt ein **neuer Dienst** dazu, bleibt die Basis primal zulässig, und der **primale Simplex** rechnet weiter. Vier Fragen, alle gemessen: **(1) Ein dualer Pivot** - was passiert in einem Schritt? **(2) Warm gegen kalt** - wie viele Pivots spart der Warmstart, und wann nicht?
**(3) Ketten** - 50 Änderungen nacheinander (rollierender Horizont, Branching-Folge). **(4) Wertkurve** - die Kurve aus dem letzten Stück als warm fortgeführte Folge.
"""
)
st.caption("Kind von [Dualität und Sensitivität](https://github.com/sebastian-hanisch/lp-dualitaet-demo). Folgestücke (Innere Punkte, Ellipsoid, Präsolve) sind [noch nicht gebaut].")

with st.expander("So funktioniert die Neuoptimierung", expanded=True):
    st.markdown(
        """
1. **Änderung am Endtableau:** die rechte Seite wird um δ·B⁻¹e_i verschoben, eine neue Nebenbedingung als Zeile mit neuer Schlupfspalte angefügt, eine Kostenänderung ändert nur die Zielzeile, ein neuer Dienst bringt die Spalte B⁻¹a mit. Nichts wird neu gerechnet.
2. **Welcher Simplex?** Nur die rechte Seite verletzt (x_B < 0, alle r_j ≥ 0): **dualer Simplex**. Nur die Zielzeile verletzt (ein r_j < 0, alle x_B ≥ 0): **primaler Simplex**. Beides in Ordnung: die Basis ist noch optimal, **0 Pivots**.
3. **Ein dualer Pivot:** wähle eine Zeile mit negativer rechter Seite (Dantzig: die kleinste); der **duale Quotiententest** min r_j / |a_ij| über alle Einträge a_ij < 0 der Zeile wählt die eintretende Spalte; pivotiere. Der Zielwert **fällt** (die Basis ist eine obere Schranke), bis alle x_B ≥ 0 sind. Hat die Zeile keinen negativen Eintrag, ist die Instanz **unzulässig**.
4. **Bezug zum letzten Stück:** liegt die Änderung innerhalb des Ranging-Bereichs, bleibt die Basis optimal: 0 Pivots. Außerhalb sind es meist wenige.
5. **Neustart** heißt hier: der Zwei-Phasen-Simplex desselben Lösers von Null auf der geänderten Instanz. **Pivots** sind das Maß (ein Pivot kostet im dichten Tableau überall gleich viel).
        """
    )

if C.PRESETS:
    st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
    preset_names = list(C.PRESETS.keys())
    for row in (preset_names[:4], preset_names[4:7], preset_names[7:]):
        if not row:
            continue
        cols = st.columns(len(row))
        for col, name in zip(cols, row):
            with col:
                st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

ss = st.session_state
with st.sidebar:
    st.header("⚙️ Einstellungen")
    kind = st.selectbox("Instanz", options=list(S.KINDS), format_func=lambda v: S.KIND_LABELS[v], key="kind_select",
                        help="Lehrbuchbeispiel, Zentrum und entartete Ecke sind fest; Zufall (≤-Ressourcen) und Mischung (mit ≥) sind regelbar. Ohne Optimum gibt es nichts weiterzurechnen.")
    random_kind = kind not in S.FIXTURE_KINDS
    if random_kind:
        m = st.slider("Ressourcen m", *bounds("m_slider"), value=int(ss["m_slider"]), key="m_widget", on_change=store_from_widget, args=("m_slider",), help="Zahl der Bedingungen.")
        n = st.slider("Dienste n", *bounds("n_slider"), value=int(ss["n_slider"]), key="n_widget", on_change=store_from_widget, args=("n_slider",), help="Zahl der Variablen.")
        seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), value=int(ss["seed_input"]), key="seed_widget", step=1, on_change=store_from_widget, args=("seed_input",))
        st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)
    else:
        m, n, seed = C.DEFAULT_M, C.DEFAULT_N, C.DEFAULT_SEED
    change = st.radio("Änderung", options=list(C.CHANGES), format_func=lambda c: C.CHANGE_LABELS[c], key="change_select", on_change=reset_level,
                      help="Rechte Seite, Schnitt und Schranke lassen die Basis dual zulässig (dualer Simplex); Kosten und neuer Dienst lassen sie primal zulässig (primaler Simplex).")
    options = list(range(len(C.LEVELS[change])))
    ss["level_select"] = min(int(ss["level_select"]), len(options) - 1)
    ss["level_widget"] = ss["level_select"]
    level = st.select_slider("Größe der Änderung", options=options, key="level_widget", on_change=store_from_widget, args=("level_select",),
                             format_func=lambda i: C.level_label(change, i), help="Prozent bei rechter Seite und Kosten, λ bei Schnitt und Schranke (β = λ·a·x*), Faktor f beim neuen Dienst (Deckungsbeitrag = f · y·a).")
    tmp = S.generate(kind, int(m), int(n), C.DENSITY, int(seed))
    ss["resource_select"] = min(int(ss["resource_select"]), tmp.m - 1)
    ss["var_select"] = min(int(ss["var_select"]), tmp.n - 1)
    if change == "rhs":
        st.selectbox("Betroffene Ressource", options=list(range(tmp.m)), format_func=lambda i: tmp.row_names[i], key="resource_select")
    if change in ("cost", "bound", "column"):
        st.selectbox("Betroffener Dienst", options=list(range(tmp.n)), format_func=lambda j: tmp.names[j], key="var_select", help="Kosten und Schranke gelten für diesen Dienst; beim neuen Dienst ist er die Vorlage der Spalte (0.9 mal).")
    rule = st.radio("Zeilenwahl im dualen Simplex", options=list(C.RULES), format_func=lambda r: C.RULE_LABELS[r], key="rule_select", help="Wirkt nur, wenn der duale Simplex rechnet (Änderungsarten mit dual zulässiger Basis).")

sync_query_params({"kind_select": kind, "m_slider": int(ss["m_slider"]), "n_slider": int(ss["n_slider"]), "seed_input": int(ss["seed_input"]), "change_select": change, "level_select": int(ss["level_select"]),
                   "resource_select": int(ss["resource_select"]), "var_select": int(ss["var_select"]), "rule_select": rule, "dsx_step": int(ss["dsx_step"])})

settings = Settings(kind, int(m), int(n), int(seed), change, int(ss["level_select"]), int(ss["resource_select"]), int(ss["var_select"]), rule)
with st.spinner("Rechne..."):
    a = analyse(settings)

# --- In Aktion ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Nach der Änderung weiterrechnen statt neu anfangen")
step = st.select_slider("Schritt", options=list(C.STEPS), key="dsx_step", format_func=lambda s: C.STEPS[s])

if a is None:
    st.warning("Die Ausgangsinstanz hat kein Optimum (unzulässig oder unbeschränkt): es gibt keine Endbasis, von der aus man weiterrechnen könnte.")
else:
    inst, warm, cold = a.inst, a.warm, a.cold
    if not a.status_equal or not a.obj_equal:
        st.error("Warmstart und Neustart weichen ab - bitte melden.")
    else:
        st.success(f"✅ Warmstart und Neustart liefern dasselbe: **{STATUS_TEXT[warm.status]}**" + (f" {num(warm.obj)}" if warm.status == "optimal" else "") + f" ({a.desc}).")
    if warm.method == "cold":
        st.info(f"Der Warmstart ist hier nicht möglich ({warm.note}); die Demo fällt auf den Neustart zurück.")

    if step == 1:
        st.markdown(f"**Änderung:** {a.desc}. Basis nach der Änderung: primal {'zulässig' if a.new_tab.is_primal_feasible() else '**unzulässig**'}, dual {'zulässig' if a.new_tab.is_dual_feasible() else '**unzulässig**'} → **{METHOD_TEXT[warm.method]}**.")
        if inst.m > 12 or not warm.path:
            if warm.method == "none":
                st.info("Die alte Basis ist auch nach der Änderung optimal: 0 Pivots. Die Änderung liegt innerhalb des Bereichs, in dem die Basis optimal bleibt.")
            else:
                st.info("Für diese Instanz gibt es keinen Verlauf zum Nachvollziehen.")
        else:
            total = warm.pivots
            if "pivot_k" in ss:
                ss["pivot_k"] = min(max(0, int(ss["pivot_k"])), total)
            k = st.slider("Pivot", 0, total, key="pivot_k", help="0 = Tableau direkt nach der Änderung; danach Pivot für Pivot.") if total > 0 else 0
            st_ = warm.path[k]
            T, basis = st_["T"], st_["basis"]
            names, m_rows = a.new_tab.names, a.new_tab.m
            df = tableau_frame(T, basis, names, a.new_tab.allowed(), m_rows)
            neg = [i for i in range(m_rows) if T[i, -1] < -1e-9]
            nxt = warm.path[k + 1] if k < total else None

            def style(frame):
                out = pd.DataFrame("", index=frame.index, columns=frame.columns)
                for i in neg:
                    out.iloc[i, -1] = "background-color: #f8d0d0; font-weight: 600"
                out.iloc[m_rows, :] = "background-color: #eef3f2"
                if nxt is not None:
                    row_name, col_name = frame.index[nxt["row"]], names[nxt["col"]]
                    out.loc[row_name, col_name] = "background-color: #ffe08a; font-weight: 700"
                return out
            st.dataframe(df.style.apply(style, axis=None).format("{:.2f}"), width="stretch")
            st.caption("Rot: negative rechte Seite (primal unzulässig). Gelb: das Pivotelement des nächsten Schritts. Die letzte Zeile sind die reduzierten Kosten r_j (bleiben ≥ 0: die Basis bleibt dual zulässig).")
            if nxt is not None and warm.method == "dual":
                i = nxt["row"]
                cand = {j: max(T[m_rows, j], 0.0) / -T[i, j] for j in a.new_tab.allowed() if j not in basis and T[i, j] < -1e-9}
                st.markdown(f"**Pivot {k + 1}:** die Zeile von **{names[basis[i]]}** hat die rechte Seite {num(T[i, -1])} und verlässt die Basis; der duale Quotiententest r_j / |a_ij| über die Spalten mit negativem Eintrag wählt **{names[nxt['col']]}**.")
                st.plotly_chart(build_ratios(cand, nxt["col"], names), width="stretch", key=f"s1_ratios_{k}")
            elif nxt is not None:
                st.markdown(f"**Pivot {k + 1}:** die Spalte **{names[nxt['col']]}** (negative reduzierte Kosten) tritt ein, **{names[nxt['leave']]}** verlässt die Basis (primaler Quotiententest).")
            elif total > 0:
                st.markdown("**Fertig:** alle rechten Seiten ≥ 0 und alle reduzierten Kosten ≥ 0 - die neue Endbasis ist optimal.")
            if total > 0:
                st.plotly_chart(build_progress(warm.path), width="stretch", key="s1_progress")
                st.caption("Der Zielwert läuft von der alten Basis (nach der Änderung) zum neuen Optimum" + (": beim dualen Simplex von oben nach unten, die Unzulässigkeit sinkt auf 0." if warm.method == "dual" else ": beim primalen Simplex von unten nach oben."))
    elif step == 2:
        st.markdown("**Pivots dieser Änderung:**")
        st.plotly_chart(build_warm_cold(warm.pivots, cold.pivots, METHOD_TEXT[warm.method].capitalize()), width="stretch", key="s2_bars")
        st.caption(f"{a.desc}. Neustart = Zwei-Phasen-Simplex von Null auf der geänderten Instanz. Ein einzelner Fall; die Kurve unten und die Tabelle fassen alle Ressourcen bzw. Dienste (und die fünf festen Instanzen) zusammen.")
        st.markdown(f"**Über die Größe der Änderung** ({C.CHANGE_SHORT[change]}; Summe der Pivots über alle betroffenen Ressourcen bzw. Dienste, bei Zufall über 5 feste Instanzen; 🔬 auf Abruf):")
        if st.button("Kurve über die Größe berechnen", key="sweep_start"):
            ss["sweep_done"] = (settings.kind, settings.m, settings.n, settings.change, settings.rule)
        if ss.get("sweep_done") == (settings.kind, settings.m, settings.n, settings.change, settings.rule):
            with st.spinner("Rechne..."):
                rows = ev.sweep(settings, change)
            st.plotly_chart(build_sweep(rows, C.CHANGE_LABELS[change]), width="stretch", key="s2_sweep")
            st.dataframe(pd.DataFrame([{"Größe": r["label"], "Läufe": r["n"], "Pivots warm": r["warm"], "Pivots kalt": r["cold"], "warm / kalt": num(r["ratio"]), "warm ≥ kalt": f"{r['warm_ge_cold']:.0%}",
                                        "ohne Pivot": f"{r['zero']:.0%}"} for r in rows]), hide_index=True, width="stretch")
            st.caption("Je größer die Änderung, desto mehr Pivots braucht der Warmstart; bei sehr großen kann der Neustart nicht schlechter sein (Spalte 'warm ≥ kalt').")
        st.markdown("**Alle Änderungsarten** bei der gewählten Größenstufe (🔬 auf Abruf):")
        if st.button("Alle Änderungsarten berechnen", key="table_start"):
            ss["table_done"] = (settings.kind, settings.m, settings.n, settings.rule)
        if ss.get("table_done") == (settings.kind, settings.m, settings.n, settings.rule):
            with st.spinner("Rechne..."):
                trs = []
                for ch in C.CHANGES:
                    rows = ev.sweep(settings, ch)
                    row = rows[min(int(ss["level_select"]), len(rows) - 1)]
                    trs.append({"Änderung": C.CHANGE_SHORT[ch], "Simplex": C.CHANGE_METHOD[ch], "Größe": row["label"], "Läufe": row["n"], "Pivots warm": row["warm"], "Pivots kalt": row["cold"], "warm / kalt": num(row["ratio"]),
                                "ohne Pivot": f"{row['zero']:.0%}"})
            st.dataframe(pd.DataFrame(trs), hide_index=True, width="stretch")
    elif step == 3:
        ch = ev.chain(settings)
        st.markdown(f"**Rollierender Horizont:** {C.CHAIN_STEPS} zufällige Änderungen je einer rechten Seite (b_i wandert in ±{C.CHAIN_SPREAD:.0%} des Ausgangswerts). Warm: jeweils vom letzten Endtableau; kalt: jeder Schritt von Null.")
        if ch is None or not ch["warm"]:
            st.info("Für diese Instanz gibt es keine Kette.")
        else:
            st.plotly_chart(build_chain(ch, "Änderung Nr."), width="stretch", key="s3_chain")
            zero = sum(1 for w in ch["warm"] if w == 0)
            st.markdown(f"**Zusammen {sum(ch['warm'])} Pivots warm gegen {sum(ch['cold'])} kalt** ({sum(ch['warm']) / max(1, sum(ch['cold'])):.0%}); in {zero} von {len(ch['warm'])} Schritten war gar kein Pivot nötig.")
            st.plotly_chart(build_chain_hist(ch), width="stretch", key="s3_hist")
        dv = ev.dive(settings)
        st.markdown(f"**Branching-Folge:** nacheinander die Schranke x_j ≤ 0.8·x_j* für den Dienst mit der größten Menge (bis zu {C.DIVE_STEPS} Schritte; kalt: jeweils die ganze Instanz neu).")
        if dv is None or not dv["warm"]:
            st.info("Für diese Instanz gibt es keine Folge.")
        else:
            st.plotly_chart(build_chain(dv, "Schranke Nr."), width="stretch", key="s3_dive")
            st.caption(f"Warm {sum(dv['warm'])} Pivots, kalt {sum(dv['cold'])}; der Optimalwert fällt mit jeder Schranke von {num(dv['objs'][0])} auf {num(dv['objs'][-1])}.")
    else:
        pr = ev.parametric(Settings(kind, int(m), int(n), int(seed), "rhs", 4, int(ss["resource_select"]), int(ss["var_select"]), rule))
        st.markdown(f"**Wertkurve der Ressource {inst.row_names[settings.res_i]}** (rechte Seite von 60 % darunter bis 150 % darüber, {C.CURVE_POINTS} Punkte): kalt löst jeden Punkt von Null, warm setzt die Folge von der Ausgangsbasis nach oben und nach unten fort.")
        if change != "rhs":
            st.info("Die Ressource der Kurve wählt die Seitenleiste bei der Änderungsart 'Rechte Seite'.")
        if pr is None:
            st.info("Die Wertkurve gibt es nur für ≤-Ressourcen (Kapazitäten).")
        else:
            st.plotly_chart(build_parametric(pr, inst.row_names[settings.res_i], inst.b[settings.res_i]), width="stretch", key="s4_curve")
            st.markdown(f"**{pr['total_warm']} Pivots warm gegen {pr['total_cold']} kalt** für alle {C.CURVE_POINTS} Punkte; die Kurve hat entlang der Punkte **{pr['basis_changes']} Basiswechsel**: der Warmstart braucht je Basiswechsel einen Pivot.")

    st.markdown("---")
    st.markdown("## ⚙️ Der gewählte Fall")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pivots warm", str(warm.pivots), delta=METHOD_TEXT[warm.method][:22], delta_color="off")
    m2.metric("Pivots kalt", str(cold.pivots), delta="Neustart von Null", delta_color="off")
    m3.metric("Warm / kalt", f"{warm.pivots / cold.pivots:.2f}" if cold.pivots else "-", delta="Pivots", delta_color="off")
    m4.metric("Ergebnis", num(warm.obj) if warm.status == "optimal" else "-", delta=STATUS_TEXT[warm.status], delta_color="off")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Warmstart spart immer.** | Bei kleinen Änderungen fast immer stark; bei großen kann der Neustart nicht schlechter sein (Schnitt mit λ = 0.5 oder ein neuer Dienst mit f = 2 in einem Teil der Fälle). Die Spalte "warm ≥ kalt" zählt sie. | Größe der Änderung |
| **Der Neustart ist der Maßstab.** | Der Neustart ist hier der Zwei-Phasen-Simplex desselben Lösers; ein besserer Startpunkt (Crash-Basis, dualer Simplex von Null) würde die Lücke verkleinern. | Echte Löser |
| **Jede Änderung lässt sich warm starten.** | Bei einer künstlichen Variable in der Basis (redundante Gleichung) ist der Warmstart nicht möglich; die Demo fällt auf den Neustart zurück und meldet es. | Präsolve |
| **Der duale Simplex ist ein neues Verfahren.** | Er rechnet dasselbe Tableau mit vertauschten Rollen von Zeile und Spalte; Pivotregeln (Dantzig, Bland) gibt es auch dort. Duales Steepest Edge und Bound Flipping (in echten Löser Standard) sind nicht gebaut. | Forrest und Goldfarb |
| **Schranken sind Zeilen.** | Hier ist x_j ≤ k eine neue Nebenbedingung; echte Löser behandeln Variablenschranken direkt und sparen die Zeile. | Bound-Simplex |
| **Branching ist ein Baum.** | Die Demo zeigt einzelne Schritte und eine Folge (Tauchgang), keinen Branch-and-Bound-Baum. | Exakte Suche |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Endbasis.** $B$ optimal: $x_B = B^{-1}b \ge 0$ und $r_j = c_B^\top B^{-1}a_j - c_j \ge 0$ (Maximierung). **Änderungen:** $b \to b + \delta e_i$: $x_B \to x_B + \delta B^{-1}e_i$, $r$ bleibt (dual zulässig); neue Zeile $a^\top x \le \beta$: Schlupf $s$ als Basisvariable, rechte Seite $\beta - a^\top x^*$ (negativ, wenn $x^*$ verletzt);
$c_j \to c_j'$: nur $r$ ändert sich (primal zulässig); neue Spalte $a$: $B^{-1}a$, $r = y^\top a - c$.

**Dualer Simplex.** Zeile $i$ mit $x_{B,i} < 0$; Spalte $j$ mit $\alpha_{ij} < 0$ und $r_j/|\alpha_{ij}|$ minimal; nach dem Pivot bleibt $r \ge 0$, der Zielwert ist nicht steigend. Kein $\alpha_{ij} < 0$ in der Zeile beweist die Unzulässigkeit (Farkas). **Primaler Simplex:** wie in den Vorgängerstücken.

**Literatur.** Lemke, C. E. (1954). *The dual method of solving the linear programming problem.* Naval Research Logistics Quarterly 1(1), 36-47. Forrest, J. J., & Goldfarb, D. (1992). *Steepest-edge simplex algorithms for linear programming.* Mathematical Programming 57, 341-374 (nur genannt).

Implementiert in `dsx_reopt.py` (Änderungen am Endtableau, dualer und primaler Simplex, Wahl nach Zulässigkeit), `dsx_algorithm.py` (Zwei-Phasen-Simplex als Neustart), `dsx_evaluation.py`, `dsx_scenario.py`.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
