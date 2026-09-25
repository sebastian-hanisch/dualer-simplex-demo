"""Auswertung: eine Änderung warm gegen kalt, Größe der Änderung, Ketten, Branching-Folge, parametrische Wertkurve."""

import pytest

import dsx_constants as C
import dsx_evaluation as ev
import dsx_reopt as R
from dsx_evaluation import Settings


def test_settings_clamp_level_resource_and_variable():
    s = Settings("textbook", change="row", level=6, resource=9, var=9)
    assert (s.level_i, s.level_value, s.res_i, s.var_j) == (3, 0.5, 2, 1)
    assert Settings("centre", change="rhs", level=6).level_value == 100 and Settings("centre", change="column", level=0).level_value == 0.8


def test_level_labels():
    assert C.level_label("rhs", 5) == "+50 %" and C.level_label("cost", 0) == "-50 %" and C.level_label("row", 3) == "λ = 0.5" and C.level_label("column", 2) == "f = 1.25" and C.level_label("bound", 9) == "λ = 0.5"


def test_analyse_compares_warm_and_cold_and_caches():
    a = ev.analyse(Settings("centre", change="rhs", level=5, resource=0))
    assert a is ev.analyse(Settings("centre", change="rhs", level=5, resource=0))
    assert a.status_equal and a.obj_equal and (a.warm.method, a.warm.pivots, a.cold.pivots) == ("dual", 2, 3) and a.warm.obj == pytest.approx(880.8333, abs=1e-3)
    assert len(a.warm.path) == 3 and a.warm.path[0]["start"] and "Kommissionierstunden: b +50 %" == a.desc
    for change, method in (("row", "dual"), ("bound", "dual"), ("cost", "primal"), ("column", "primal")):
        r = ev.analyse(Settings("centre", change=change, level=2, var=0))
        assert r.warm.method == method and r.status_equal and r.obj_equal and r.warm.pivots <= r.cold.pivots
    assert ev.analyse(Settings("infeasible")) is None and ev.analyse(Settings("unbounded")) is None


def test_analyse_keeps_the_tableau_history_only_for_small_instances():
    small = ev.analyse(Settings("random", 6, 6, 3, "rhs", 6, 0))
    assert all("T" in p for p in small.warm.path) if small.warm.path else True
    big = ev.analyse(Settings("random", 12, 12, 3, "rhs", 6, 0))
    assert big.status_equal and big.obj_equal


def test_ops_per_pivot_is_the_dense_tableau_model():
    a = ev.analyse(Settings("textbook", change="rhs", level=5, resource=2))
    t = a.new_tab
    assert ev.ops_per_pivot(t) == (t.ncols + 1) + 2 * t.m * (t.ncols + 1)


def test_sweep_rows_and_summary_fields():
    rows = ev.sweep(Settings("centre"), "rhs")
    assert [r["label"] for r in rows] == ["-50 %", "-25 %", "-10 %", "+10 %", "+25 %", "+50 %", "+100 %"] and [r["n"] for r in rows] == [4] * 7
    assert all(r["ratio"] == pytest.approx(r["warm"] / r["cold"]) and 0 <= r["zero"] <= 1 and 0 <= r["warm_ge_cold"] <= 1 for r in rows)
    assert rows[2]["warm"] == 1 and rows[2]["cold"] == 16 and rows[5]["warm"] == 5
    assert ev.sweep(Settings("centre"), "rhs") is rows
    bound = ev.sweep(Settings("centre"), "bound")
    assert all(r["n"] == 3 for r in bound)                                              # nur Dienste in der Lösung
    assert ev.sweep(Settings("random", 8, 8, 35), "row")[0]["n"] == 5 * 8


def test_warm_start_beats_the_restart_for_small_changes_and_loses_ground_for_large_ones():
    s = Settings("random", 10, 10, 35)
    rhs, cut, col = ev.sweep(s, "rhs"), ev.sweep(s, "row"), ev.sweep(s, "column")
    assert rhs[2]["ratio"] < 0.05 and rhs[2]["zero"] > 0.8 and rhs[0]["ratio"] > rhs[2]["ratio"]
    assert cut[3]["ratio"] > cut[0]["ratio"] and cut[3]["warm_ge_cold"] > 0.15 and col[0]["warm"] == 0 and col[0]["zero"] == 1.0 and col[3]["ratio"] > col[1]["ratio"] and col[3]["warm_ge_cold"] > 0.3


def test_chain_is_deterministic_and_the_warm_start_needs_few_pivots():
    c = ev.chain(Settings("random", 10, 10, 35))
    assert c == ev.chain(Settings("random", 10, 10, 35)) and len(c["warm"]) == C.CHAIN_STEPS == len(c["cum_cold"])
    assert c["cum_warm"][-1] == sum(c["warm"]) == 6 and c["cum_cold"][-1] == sum(c["cold"]) == 240 and sum(1 for w in c["warm"] if w == 0) == 45 and set(c["methods"]) == {"dual", "none"}
    assert all(b >= a for a, b in zip(c["cum_warm"], c["cum_warm"][1:]))
    assert ev.chain(Settings("infeasible")) is None


def test_dive_bounds_the_largest_service_step_by_step():
    d = ev.dive(Settings("centre"))
    assert d["warm"] == [1, 1, 2, 1, 1, 1, 1, 1] and sum(d["cold"]) == 41 and len(d["objs"]) == 9
    assert all(b <= a + 1e-9 for a, b in zip(d["objs"], d["objs"][1:])) and d["objs"][0] == pytest.approx(720.0)


def test_parametric_curve_needs_about_one_pivot_per_basis_change():
    p = ev.parametric(Settings("centre", resource=0))
    assert (p["total_warm"], p["total_cold"], p["basis_changes"]) == (3, 134, 3) and len(p["grid"]) == C.CURVE_POINTS and all(v is not None for v in p["z"])
    zs = [v for _b, v in sorted(zip(p["grid"], p["z"]))]
    slopes = [(z2 - z1) / (b2 - b1) for (b1, z1), (b2, z2) in zip(zip(sorted(p["grid"]), zs), zip(sorted(p["grid"])[1:], zs[1:]))]
    assert all(a >= b - 1e-7 for a, b in zip(slopes, slopes[1:]))                     # konkav
    q = ev.parametric(Settings("random", 10, 10, 35, resource=3))
    assert q["total_warm"] >= q["basis_changes"] and q["total_cold"] > 20 * q["total_warm"]
    assert ev.parametric(Settings("infeasible")) is None


def test_parametric_curve_equals_the_cold_values():
    s = Settings("centre", resource=1)
    p = ev.parametric(s)
    inst = ev.instance_of(s)
    _sol, tab = R.solve_cold(inst)
    for k in (0, 10, 20, 30, 40):
        cold, _ = R.solve_cold(R.apply_rhs(tab, 1, p["grid"][k] - inst.b[1]).inst)
        assert p["z"][k] == pytest.approx(cold.obj, rel=1e-8, abs=1e-6)
