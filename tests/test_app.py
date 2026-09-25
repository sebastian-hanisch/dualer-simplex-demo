"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz und Änderungsart, Pivot-Regler, Ressourcen-/Dienst-Auswahl, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dsx_constants as C
import dsx_scenario as S

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(step=2, **state):
    at = AppTest.from_file(APP, default_timeout=240)
    state.setdefault("dsx_step", step)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m for m in at.metric if m.label == label)


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_default_run_shows_the_centre_case():
    at = _run()
    _ok(at)
    assert {"Pivots warm", "Pivots kalt", "Warm / kalt", "Ergebnis"} == {m.label for m in at.metric}
    assert any("Warmstart und Neustart liefern dasselbe" in s.value for s in at.success) and at.get("plotly_chart")
    warm, cold = int(_metric(at, "Pivots warm").value), int(_metric(at, "Pivots kalt").value)
    assert 0 <= warm <= cold


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    _click(at, f"preset_{name}")
    _ok(at)
    p = C.PRESETS[name]
    ss = at.session_state
    assert (ss["kind_select"], ss["change_select"], ss["level_select"], ss["dsx_step"], ss["resource_select"], ss["var_select"]) == (p["kind"], p["change"], p["level"], p["step"], p["res"], p["var"])
    assert not any("weichen ab" in e.value for e in at.error)


@pytest.mark.parametrize("step", [1, 2, 3, 4])
@pytest.mark.parametrize("kind", list(S.KINDS))
@pytest.mark.parametrize("change", C.CHANGES)
def test_every_step_runs_for_every_kind_and_change(step, kind, change):
    at = _run(step=step, kind_select=kind, change_select=change, level_select=C.DEFAULT_LEVEL[change], m_slider=5, n_slider=6)
    _ok(at)
    assert at.session_state["dsx_step"] == step
    if kind in ("infeasible", "unbounded"):
        assert any("kein Optimum" in w.value for w in at.warning)


@pytest.mark.parametrize("change", C.CHANGES)
def test_every_level_runs_on_every_change(change):
    for level in range(len(C.LEVELS[change])):
        for step in (1, 2):
            at = _run(step=step, change_select=change, level_select=level)
            _ok(at)
            assert not any("weichen ab" in e.value for e in at.error)


def test_pivot_slider_walks_through_the_dual_simplex():
    at = _run(step=1, kind_select="textbook", change_select="rhs", level_select=5, resource_select=2, pivot_k=0)
    _ok(at)
    slider = next(s for s in at.slider if s.key == "pivot_k")
    assert slider.max == 1 and any("dualer Simplex" in m.value for m in at.markdown)
    slider.set_value(1).run()
    _ok(at)
    assert any("Fertig" in m.value for m in at.markdown) and at.dataframe
    at2 = _run(step=1, kind_select="centre", change_select="rhs", level_select=5, resource_select=0)
    _ok(at2)
    total = int(next(s for s in at2.slider if s.key == "pivot_k").max)
    assert total == 2
    for k in range(total + 1):
        next(s for s in at2.slider if s.key == "pivot_k").set_value(k).run()
        _ok(at2)


def test_zero_pivot_case_shows_the_range_message():
    at = _run(step=1, change_select="rhs", level_select=2, resource_select=1)
    _ok(at)
    assert any("0 Pivots" in i.value for i in at.info) and not any(s.key == "pivot_k" for s in at.slider)


def test_primal_and_dual_paths_are_labelled():
    dual = _run(step=1, change_select="row", level_select=2)
    _ok(dual)
    assert any("dualer Simplex" in m.value for m in dual.markdown)
    primal = _run(step=1, change_select="cost", level_select=5, var_select=3)
    _ok(primal)
    assert any("primaler Simplex" in m.value for m in primal.markdown)


def test_step_two_sweep_and_table_on_demand():
    at = _run(step=2)
    _ok(at)
    assert len(at.get("plotly_chart")) == 1
    _click(at, "sweep_start")
    _ok(at)
    assert len(at.get("plotly_chart")) == 2 and at.dataframe
    table = at.dataframe[-1].value
    assert list(table["Größe"]) == [C.level_label("rhs", i) for i in range(len(C.LEVELS["rhs"]))]
    _click(at, "table_start")
    _ok(at)
    assert list(at.dataframe[-1].value["Änderung"]) == [C.CHANGE_SHORT[c] for c in C.CHANGES]


def test_step_three_shows_chain_and_dive():
    at = _run(step=3, kind_select="random", m_slider=10, n_slider=10, seed_input=35)
    _ok(at)
    assert len(at.get("plotly_chart")) == 3
    assert any("Zusammen 6 Pivots warm gegen 240 kalt" in m.value and "45 von 50" in m.value for m in at.markdown)
    assert any("Branching-Folge" in m.value for m in at.markdown)


def test_step_four_shows_the_value_curve_and_the_basis_changes():
    at = _run(step=4, resource_select=0)
    _ok(at)
    assert any("3 Pivots warm gegen 134 kalt" in m.value and "3 Basiswechsel" in m.value for m in at.markdown)
    other = _run(step=4, change_select="cost", var_select=1)
    _ok(other)
    assert any("Änderungsart 'Rechte Seite'" in i.value for i in other.info)
    mixed = _run(step=4, kind_select="mixed", m_slider=8, n_slider=8)
    _ok(mixed)


def test_rule_choice_does_not_break_any_step():
    for rule in C.RULES:
        for step in (1, 2, 3, 4):
            _ok(_run(step=step, rule_select=rule, change_select="row", level_select=3))


def test_resource_and_variable_can_be_selected():
    base = _run(kind_select="centre", change_select="rhs")
    n_res = len(next(s for s in base.selectbox if s.key == "resource_select").options)
    for i in range(n_res):
        for step in (1, 2, 4):
            _ok(_run(step=step, change_select="rhs", resource_select=i))
    var_base = _run(kind_select="centre", change_select="cost")
    n_var = len(next(s for s in var_base.selectbox if s.key == "var_select").options)
    for j in range(n_var):
        for change in ("cost", "bound", "column"):
            _ok(_run(step=1, change_select=change, var_select=j))


def test_resource_selection_is_clamped_when_the_instance_gets_smaller():
    at = _run(kind_select="random", m_slider=10, n_slider=10, resource_select=9, var_select=9)
    _ok(at)
    at.session_state["kind_select"] = "textbook"
    at.run()
    _ok(at)
    assert at.session_state["resource_select"] == 2 and at.session_state["var_select"] == 1


def test_changing_the_change_kind_resets_the_level_and_swaps_the_controls():
    at = _run(step=2)
    assert any(s.key == "resource_select" for s in at.selectbox) and not any(s.key == "var_select" for s in at.selectbox)
    at.radio(key="change_select").set_value("column").run()
    _ok(at)
    assert at.session_state["level_select"] == C.DEFAULT_LEVEL["column"] and any(s.key == "var_select" for s in at.selectbox) and not any(s.key == "resource_select" for s in at.selectbox)
    at.select_slider(key="level_widget").set_value(3).run()
    _ok(at)
    assert at.session_state["level_select"] == 3
    at.radio(key="change_select").set_value("row").run()
    _ok(at)
    assert at.session_state["level_select"] == C.DEFAULT_LEVEL["row"] and not any(s.key in ("resource_select", "var_select") for s in at.selectbox)


@pytest.mark.parametrize("kw", [dict(kind_select="random", m_slider=C.M_MAX, n_slider=C.N_MAX), dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MIN), dict(kind_select="mixed", m_slider=C.M_MAX, n_slider=C.N_MAX),
                                dict(kind_select="random", m_slider=C.M_MIN, n_slider=C.N_MAX), dict(kind_select="mixed", m_slider=C.M_MIN, n_slider=C.N_MIN)])
def test_extreme_settings_run_on_every_step_and_change(kw):
    for step in (1, 2, 3, 4):
        for change in C.CHANGES:
            _ok(_run(step=step, change_select=change, level_select=C.DEFAULT_LEVEL[change], **kw))


def test_dice_button_changes_the_seed():
    at = _run(kind_select="random")
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old and at.session_state["seed_widget"] == at.session_state["seed_input"]


def test_permalink_values_are_clamped_and_invalid_choices_fall_back_to_the_default():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(m="999", n="1", step="9", kind="nope", change="x", level="99", res="99", var="-3", rule="zufall", seed="-4").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["m_slider"], ss["n_slider"], ss["dsx_step"], ss["kind_select"], ss["change_select"], ss["resource_select"], ss["var_select"], ss["rule_select"], ss["seed_input"]) == (
        C.M_MAX, C.N_MIN, 2, "centre", "rhs", 3, 0, C.DEFAULT_RULE, 0)
    assert ss["level_select"] == len(C.LEVELS["rhs"]) - 1


def test_permalink_accepts_valid_values():
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in dict(kind="mixed", m="8", n="7", seed="7", change="column", level="1", var="3", rule="bland", step="3").items():
        at.query_params[k] = v
    at.run()
    _ok(at)
    ss = at.session_state
    assert (ss["kind_select"], ss["m_slider"], ss["n_slider"], ss["seed_input"], ss["change_select"], ss["level_select"], ss["var_select"], ss["rule_select"], ss["dsx_step"]) == ("mixed", 8, 7, 7, "column", 1, 3, "bland", 3)


def test_sidebar_shows_the_controls_that_belong_to_the_instance():
    fixed = _run()
    assert not any(w.key in ("m_widget", "n_widget") for w in fixed.slider) and not any(n.key == "seed_widget" for n in fixed.number_input)
    rnd = _run(kind_select="random")
    assert any(w.key == "m_widget" for w in rnd.slider) and any(w.key == "n_widget" for w in rnd.slider) and any(n.key == "seed_widget" for n in rnd.number_input)


def test_changing_kind_and_step_on_later_steps_does_not_crash():
    for step in (1, 2, 3, 4):
        at = _run(step=step)
        _ok(at)
        for kw in (dict(kind_select="mixed", m_slider=8, n_slider=8), dict(kind_select="infeasible"), dict(kind_select="unbounded"), dict(kind_select="degenerate"), dict(kind_select="textbook"),
                   dict(kind_select="random", m_slider=4, n_slider=3), dict(kind_select="centre", resource_select=3)):
            for k, v in kw.items():
                at.session_state[k] = v
            at.run()
            _ok(at)


def test_footer_limits_and_literature_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Lemke" in m.value and "Forrest" in m.value for e in at.expander for m in e.markdown)
