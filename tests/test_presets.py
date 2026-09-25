"""Presets: gültige Werte und jede Zahl im Hilfetext gegen die echten Auswertungsfunktionen."""

import pytest

import dsx_constants as C
import dsx_evaluation as ev
from dsx_presets import PRESET_KEYS, SETTING_SPECS


def _settings(name):
    p = C.PRESETS[name]
    return ev.Settings(p["kind"], p["m"], p["n"], p["seed"], p["change"], p["level"], p["res"], p["var"], p["rule"])


def _has(name, *values):
    for v in values:
        assert v in C.PRESET_HELP[name], (name, v)


def _path(name, key):
    return [p[key] for p in ev.analyse(_settings(name)).warm.path]


def test_every_preset_has_valid_values_and_a_help_text():
    assert list(C.PRESETS) == list(C.PRESET_HELP) and len(C.PRESETS) == 10
    for name, p in C.PRESETS.items():
        assert set(p) <= set(PRESET_KEYS) and {"kind", "step", "change", "level"} <= set(p), name
        for key, state_key in PRESET_KEYS.items():
            if key in p and state_key in SETTING_SPECS:
                spec = SETTING_SPECS[state_key]
                assert spec.caster(p[key]) == p[key], (name, key)
                if spec.lo is not None:
                    assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert C.PRESET_HELP[name].strip()
        assert p["level"] < len(C.LEVELS[p["change"]])
        if "pivot_k" in p:
            assert p["step"] == 1 and p["pivot_k"] > 0


def test_help_single_changes():
    a = ev.analyse(_settings("Lehrbuch: ein dualer Pivot"))
    assert (a.warm.method, a.warm.pivots, a.cold.pivots) == ("dual", 1, 2) and [round(v, 2) for v in _path("Lehrbuch: ein dualer Pivot", "obj")] == [45.0, 42.0] and a.new_tab.primal_infeasibility() == pytest.approx(1.0)
    _has("Lehrbuch: ein dualer Pivot", "+50 % (18 → 27)", "[12, 24]", "Summe der negativen rechten Seiten 1", "45 auf das neue Optimum 42", "2 Pivots")
    a = ev.analyse(_settings("Zentrum: Kommissionierstunden +50 %"))
    assert (a.warm.pivots, a.cold.pivots) == (2, 3) and [round(v, 2) for v in _path("Zentrum: Kommissionierstunden +50 %", "obj")] == [920.0, 893.75, 880.83]
    _has("Zentrum: Kommissionierstunden +50 %", "(80 → 120, Bereich [70, 90])", "2 duale Pivots", "920, 893.75, 880.83", "3 beim Neustart", "880.83 (vorher 720)")
    a = ev.analyse(_settings("Innerhalb des Bereichs: null Pivots"))
    assert (a.warm.method, a.warm.pivots, a.cold.pivots) == ("none", 0, 4) and a.warm.obj == pytest.approx(710.0)
    _has("Innerhalb des Bereichs: null Pivots", "(200 → 180)", "[172, 217.5]", "0 Pivots", "0.5 · 20", "710", "4 Pivots")
    a = ev.analyse(_settings("Schnitt schneidet den Optimalpunkt ab"))
    assert (a.warm.pivots, a.cold.pivots) == (1, 4) and a.warm.obj == pytest.approx(716.0996, abs=1e-3) and "20.54" in a.desc
    _has("Schnitt schneidet den Optimalpunkt ab", "0.75·a·x*", "20.54", "1 dualer Pivot", "720 → 716.10", "4 beim Neustart")
    a = ev.analyse(_settings("Schranke (Branching)"))
    assert (a.warm.pivots, a.cold.pivots) == (1, 4) and a.warm.obj == pytest.approx(716.5)
    _has("Schranke (Branching)", "0.75 · 20 = 15", "1 dualer Pivot", "720 → 716.5", "4 beim Neustart")
    a = ev.analyse(_settings("Kosten ändern sich: primaler Warmstart"))
    assert (a.warm.method, a.warm.pivots, a.cold.pivots) == ("primal", 2, 5) and [round(v, 2) for v in _path("Kosten ändern sich: primaler Warmstart", "obj")] == [720.0, 770.56, 770.77]
    _has("Kosten ändern sich: primaler Warmstart", "um 50 % (20 → 30", "Grenze 23.5", "2 primale Pivots", "720, 770.56, 770.77", "5 beim Neustart")
    a = ev.analyse(_settings("Neuer Dienst"))
    assert (a.warm.method, a.warm.pivots, a.cold.pivots) == ("primal", 2, 5) and [round(v, 2) for v in _path("Neuer Dienst", "obj")] == [720.0, 797.5, 840.38] and "17.44" in a.desc
    _has("Neuer Dienst", "0.9 mal", "17.44", "2 primale Pivots", "720, 797.5, 840.38", "5 beim Neustart")


def test_help_restart_not_worse_chain_and_curve():
    a = ev.analyse(_settings("Neustart nicht schlechter"))
    assert (a.warm.method, a.warm.pivots, a.cold.pivots) == ("dual", 4, 1)
    cut = ev.sweep(ev.Settings("random", 10, 10, 35), "row")[3]
    deg = ev.sweep(ev.Settings("degenerate", 6, 8, 35), "row")[3]
    assert (round(cut["warm_ge_cold"], 2), round(deg["warm_ge_cold"], 2)) == (0.26, 0.75)
    _has("Neustart nicht schlechter", "8 × 8 (Seed 22)", "λ = 0.5", "4 duale Pivots", "nur 1", "26 %", "75 %")
    c = ev.chain(_settings("Kette von 50 Änderungen"))
    assert (sum(c["warm"]), sum(c["cold"]), sum(1 for w in c["warm"] if w == 0)) == (6, 240, 45)
    _has("Kette von 50 Änderungen", "10 × 10", "±30 %", "6 Pivots warm gegen 240 kalt", "45 von 50")
    p = ev.parametric(_settings("Wertkurve: ein Pivot je Knickpunkt"))
    assert (p["total_warm"], p["total_cold"], p["basis_changes"]) == (3, 134, 3)
    _has("Wertkurve: ein Pivot je Knickpunkt", "41 Punkte", "warm 3 Pivots gegen 134 kalt", "3 Basiswechsel")
