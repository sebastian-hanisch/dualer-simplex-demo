"""Korrektheitskette: warm == kalt == HiGHS für jede Änderungsart, Tableau nach den Änderungen konsistent, Invarianten des dualen Simplex, Bezug zu den Bereichen aus Stück 5, Unzulässig/Unbeschränkt, Branching, Rückfall."""

import math
import random

import numpy as np
import pytest

import dsx_algorithm as A
import dsx_reopt as R
import dsx_scenario as S
from tests.test_scenario import _highs, reference_status


def _custom(rows, b, c, senses):
    n = len(c)
    return S.Instance(tuple(tuple(float(v) for v in r) for r in rows), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(f"x{j}" for j in range(n)), tuple(f"r{i}" for i in range(len(b))), "custom")


def _bases(seeds=8):
    """Ausgangsinstanzen (nur zulässig und beschränkt): Lehrbuch, Zentrum, entartete Ecke, Zufall (<=), Mischung (mit >=)."""
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()
    for seed in range(seeds):
        yield S.generate("random", 5, 6, 0.5, seed)
        yield S.generate("random", 8, 4, 0.5, seed)
        yield S.generate("mixed", 6, 6, 0.5, seed)


def _changes(tab, rng):
    """Alle Änderungsarten an einem optimalen Tab: (Art, Beschreibung, neuer Tab)."""
    inst, x, y = tab.inst, np.array(tab.x()), np.array(tab.y())
    out = []
    for i in range(inst.m):
        for p in (0.1, 0.5, 1.5, -0.1, -0.5, -1.5):
            out.append(("rhs", f"b{i} {p:+.0%}", R.apply_rhs(tab, i, p * max(abs(inst.b[i]), 1.0))))
    for j in range(inst.n):
        for p in (0.2, 1.0, -0.2, -0.6):
            out.append(("cost", f"c{j} {p:+.0%}", R.apply_cost(tab, j, inst.c[j] * (1 + p))))
    for lam in (1.2, 0.9, 0.5):
        a = np.array([rng.choice([0.0, 0.0, 1.0, 2.0, 3.5]) for _ in range(inst.n)]) + 0.1 * np.array([rng.random() for _ in range(inst.n)])
        out.append(("row", f"cut {lam}", R.add_row(tab, a, lam * float(a @ x) + 0.01)))
    for j in range(inst.n):
        out.append(("bound", f"x{j} <= 0.5 x*", R.add_bound(tab, j, upper=0.5 * x[j])))
        out.append(("bound", f"x{j} >= x*+1", R.add_bound(tab, j, lower=x[j] + 1.0)))
    for f in (0.8, 1.25, 2.0):
        j = rng.randrange(inst.n)
        col = 0.9 * np.array([inst.A[i][j] for i in range(inst.m)])
        out.append(("column", f"new {f}", R.add_column(tab, col, f * float(y @ col))))
    return out


def _scenarios(seeds=8):
    rng = random.Random(7)
    for inst in _bases(seeds):
        sol, tab = R.solve_cold(inst)
        assert tab is not None
        for kind, label, new in _changes(tab, rng):
            yield inst, kind, label, tab, new


def test_warm_equals_cold_equals_highs_for_every_change_kind():
    stats = {}
    count = 0
    for inst, kind, label, tab, new in _scenarios():
        warm = R.reoptimize(new)
        ref = reference_status(new.inst)
        cold, _ = R.solve_cold(new.inst)
        assert warm.status == cold.status == ref, (inst.kind, kind, label, warm.status, cold.status, ref)
        if ref == "optimal":
            h = _highs(new.inst)
            assert warm.obj == pytest.approx(-h.fun, rel=1e-7, abs=1e-6) and cold.obj == pytest.approx(warm.obj, rel=1e-7, abs=1e-6)
            assert A.primal_violation(new.inst, warm.x) < 1e-6 and A.dual_violation(new.inst, warm.y) < 1e-6
            assert float(np.dot(warm.y, new.inst.b)) == pytest.approx(warm.obj, rel=1e-7, abs=1e-6)
        stats.setdefault((kind, warm.method), 0)
        stats[(kind, warm.method)] += 1
        count += 1
    assert count >= 1000
    assert {"dual", "primal", "none"} <= {m for (_k, m) in stats} and all(stats.get((k, "dual"), 0) > 20 for k in ("rhs", "row", "bound")) and stats.get(("cost", "primal"), 0) > 20 and stats.get(("column", "primal"), 0) > 5
    assert sum(v for (_k, m), v in stats.items() if m == "cold") < 0.02 * count


def test_change_kinds_choose_the_expected_simplex():
    inst = S.centre_instance()
    _sol, tab = R.solve_cold(inst)
    assert R.reoptimize(R.apply_rhs(tab, 1, 60.0)).method == "dual" and R.reoptimize(R.apply_rhs(tab, 1, -60.0)).method == "dual"
    assert R.reoptimize(R.apply_cost(tab, 3, 30.0)).method == "primal" and R.reoptimize(R.apply_cost(tab, 0, 30.0)).method in ("primal", "none")
    assert R.reoptimize(R.add_row(tab, [1, 1, 1, 1, 1], 30.0)).method == "dual"
    assert R.reoptimize(R.add_bound(tab, 0, upper=10.0)).method == "dual"
    assert R.reoptimize(R.add_column(tab, [1, 1, 1, 1], 100.0)).method == "primal"


def test_tableau_is_consistent_with_the_changed_instance():
    """Basislösung, Schlupf und Zielzeile des geänderten Tableaus gegen die Instanz (nur <=-Zeilen: die Anfangsbasis besteht aus Schlupfspalten)."""
    checked = 0
    for inst, kind, label, tab, new in _scenarios(4):
        if S.GE in inst.senses or S.EQ in inst.senses:
            continue
        T, m = new.T, new.m
        A_, b, c = new.inst.arrays()
        assert (m, len(new.struct)) == (new.inst.m, new.inst.n)
        assert np.allclose(T[:m, new.basis], np.eye(m), atol=1e-9) and np.allclose(T[m, new.basis], 0.0, atol=1e-9)              # Einheitsspalten der Basis, Zielzeile 0 dort
        x = np.array(new.x())
        slack = b - A_ @ x
        for i in range(m):
            col = new.init_basis[i]
            value = T[new.basis.index(col), -1] if col in new.basis else 0.0
            assert value == pytest.approx(slack[i], abs=1e-7), (kind, label, i)                                               # Schlupf == b - A x, Nichtbasis-Schlupf == 0
        y = np.array(new.y())
        for k, col in enumerate(new.struct):
            assert T[m, col] == pytest.approx(float(y @ A_[:, k]) - c[k], abs=1e-7)                                          # reduzierte Kosten == y a - c
        for i in range(m):
            assert T[m, new.init_basis[i]] == pytest.approx(y[i], abs=1e-9)
        assert T[m, -1] == pytest.approx(float(c @ x), abs=1e-6)
        checked += 1
    assert checked > 300


def test_dual_simplex_keeps_dual_feasibility_and_never_increases_the_objective():
    steps = 0
    for inst, kind, label, tab, new in _scenarios(4):
        if kind not in ("rhs", "row", "bound") or new.is_primal_feasible() or not new.is_dual_feasible():
            continue
        res = R.dual_simplex(new, keep=True)
        assert res.pivots == len(res.path) - 1
        objs = [p["obj"] for p in res.path]
        assert all(b <= a + 1e-7 for a, b in zip(objs, objs[1:]))
        for p in res.path:
            T = p["T"]
            assert all(T[new.m, j] >= -1e-8 for j in new.allowed())
        if res.status == "optimal":
            assert res.path[-1]["infeasibility"] == pytest.approx(0.0, abs=1e-9) and res.obj == pytest.approx(objs[-1])
        steps += res.pivots
    assert steps > 100


def test_dual_simplex_refuses_a_basis_that_is_not_dual_feasible():
    _sol, tab = R.solve_cold(S.centre_instance())
    bad = R.apply_cost(tab, 3, 40.0)
    assert not bad.is_dual_feasible()
    with pytest.raises(ValueError):
        R.dual_simplex(bad)


def test_bland_stall_guard_and_rules_reach_the_same_optimum(monkeypatch):
    results = {}
    for name, limit in (("dantzig", 25), ("bland", 25), ("guard", 0)):
        monkeypatch.setattr(R, "STALL_LIMIT", limit)
        objs = []
        for inst, kind, label, tab, new in _scenarios(3):
            if kind not in ("rhs", "row", "bound") or new.is_primal_feasible():
                continue
            res = R.reoptimize(new, "bland" if name == "bland" else "dantzig")
            objs.append((res.status, None if res.status != "optimal" else round(res.obj, 6)))
        results[name] = objs
    assert results["dantzig"] == results["bland"] == results["guard"] and len(results["dantzig"]) > 100


# --- Bezug zu Stück 5 (Ranging) -----------------------------------------------------------------------------------------------------------------------


def test_changes_inside_the_ranges_need_no_pivot_and_outside_at_least_one():
    _sol, tab = R.solve_cold(S.centre_instance())
    for i, inside, outside in ((1, (10.0, -20.0), (30.0, -40.0)), (0, (5.0, -8.0), (15.0, -15.0)), (2, (2.0, -2.0), (5.0, -3.5))):
        for d in inside:
            r = R.reoptimize(R.apply_rhs(tab, i, d))
            assert (r.method, r.pivots) == ("none", 0) and r.y == pytest.approx(tab.y(), abs=1e-9)
        for d in outside:
            r = R.reoptimize(R.apply_rhs(tab, i, d))
            assert r.method == "dual" and r.pivots >= 1
    for j, inside, outside in ((0, 16.0, 17.5), (1, 14.0, 21.0), (2, 25.5, 28.0)):
        assert R.reoptimize(R.apply_cost(tab, j, inside)).pivots == 0 and R.reoptimize(R.apply_cost(tab, j, outside)).pivots >= 1
    assert R.reoptimize(R.apply_cost(tab, 3, 23.4)).pivots == 0 and R.reoptimize(R.apply_cost(tab, 3, 23.6)).pivots >= 1
    x = np.array(tab.x())
    a = np.array([1.0, 2.0, 0.5, 0.0, 1.0])
    assert R.reoptimize(R.add_row(tab, a, float(a @ x) + 1.0)).pivots == 0 and R.reoptimize(R.add_row(tab, a, float(a @ x) - 1.0)).pivots >= 1


# --- Unzulässig, unbeschränkt, Branching, Rückfall -----------------------------------------------------------------------------------------------------


def test_infeasible_changes_are_found_by_a_row_without_negative_entry():
    _sol, tab = R.solve_cold(S.centre_instance())
    r = R.reoptimize(R.apply_rhs(tab, 1, -300.0))
    assert (r.status, r.method) == ("infeasible", "dual") and "negativen Eintrag" in r.note
    assert reference_status(R.apply_rhs(tab, 1, -300.0).inst) == "infeasible" and A.solve(R.apply_rhs(tab, 1, -300.0).inst).status == "infeasible"
    tight = R.add_row(tab, [1.0, 1.0, 1.0, 1.0, 1.0], -5.0)
    assert R.reoptimize(tight).status == "infeasible" and reference_status(tight.inst) == "infeasible"
    boxed = R.add_bound(R.add_bound(tab, 0, upper=5.0), 0, lower=9.0)
    assert R.reoptimize(boxed).status == "infeasible" and reference_status(boxed.inst) == "infeasible"


def test_unbounded_column_is_found_by_the_primal_simplex():
    _sol, tab = R.solve_cold(S.centre_instance())
    free = R.add_column(tab, [0.0, 0.0, 0.0, 0.0], 3.0)
    r = R.reoptimize(free)
    assert (r.status, r.method) == ("unbounded", "primal") and reference_status(free.inst) == "unbounded" and A.solve(free.inst).status == "unbounded"


def test_branching_children_equal_highs_and_never_beat_the_parent():
    for inst in list(_bases(4)):
        _sol, tab = R.solve_cold(inst)
        x = tab.x()
        for j in range(inst.n):
            k = math.floor(x[j] * 0.7 + 0.2)
            for child in (R.add_bound(tab, j, upper=k), R.add_bound(tab, j, lower=k + 1)):
                r = R.reoptimize(child)
                assert r.status == reference_status(child.inst)
                if r.status == "optimal":
                    assert r.obj == pytest.approx(-_highs(child.inst).fun, rel=1e-7, abs=1e-6) and r.obj <= tab.obj() + 1e-7


def test_fallback_to_cold_start_is_reported_when_an_artificial_variable_stays_basic():
    inst = _custom([[1, 1, 0], [1, 1, 0], [0, 1, 1]], [4, 4, 6], [1, 2, 1], [S.EQ, S.EQ, S.LE])
    _sol, tab = R.solve_cold(inst)
    assert any(j in tab.art for j in tab.basis)
    warm_ok = R.reoptimize(R.apply_cost(tab, 0, 3.0))
    assert warm_ok.status == "optimal" and warm_ok.method in ("primal", "none")
    bad = R.apply_rhs(tab, 0, 1.0)
    r = R.reoptimize(bad)
    assert r.method == "cold" and r.note and r.status == A.solve(bad.inst).status == reference_status(bad.inst)


def test_no_change_bookkeeping_and_determinism():
    inst = S.centre_instance()
    _sol, tab = R.solve_cold(inst)
    same = R.reoptimize(R.apply_rhs(tab, 0, 0.0))
    assert (same.method, same.pivots, same.status) == ("none", 0, "optimal") and same.obj == pytest.approx(720.0)
    new = R.apply_rhs(tab, 1, 60.0)
    a, b = R.reoptimize(new, keep=True), R.reoptimize(new, keep=True)
    assert a.pivots == b.pivots == len(a.path) - 1 and [p.get("col") for p in a.path] == [p.get("col") for p in b.path]
    one = _custom([[2.0, 3.0]], [12.0], [4.0, 5.0], [S.LE])
    _s, t1 = R.solve_cold(one)
    r = R.reoptimize(R.apply_rhs(t1, 0, -6.0))
    assert r.status == "optimal" and r.obj == pytest.approx(12.0) and r.method in ("dual", "none")
