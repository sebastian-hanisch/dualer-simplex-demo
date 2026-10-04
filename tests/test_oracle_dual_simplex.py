"""Unabhängige Orakel für die Neuoptimierung: (1) ein exakter dualer Simplex in rationaler Arithmetik (Fraction, die Basis wird aus der alten Endbasis exakt neu berechnet) gegen den Pivotpfad
der Demo nach einer Änderung der rechten Seite bzw. einem Schnitt, (2) HiGHS gegen Status, Optimalwert und Dualvektor nach jeder Änderungsart, (3) die Knickpunkte der Wertkurve aus den Schattenpreisen von HiGHS."""

import random
from fractions import Fraction as F

import numpy as np
import pytest

import dsx_algorithm as A
import dsx_evaluation as E
import dsx_reopt as R
import dsx_scenario as S

linprog = pytest.importorskip("scipy.optimize").linprog
LE, GE, EQ = S.LE, S.GE, S.EQ


def _lp(inst):
    Am, b, c = inst.arrays()
    ub = [i for i, s in enumerate(inst.senses) if s != EQ]
    eq = [i for i, s in enumerate(inst.senses) if s == EQ]
    sg = np.array([1.0 if inst.senses[i] == LE else -1.0 for i in ub])
    return linprog(-c, A_ub=(Am[ub] * sg[:, None]) if ub else None, b_ub=(b[ub] * sg) if ub else None, A_eq=Am[eq] if eq else None, b_eq=b[eq] if eq else None, bounds=[(0, None)] * inst.n, method="highs")


def _status(inst):
    r = _lp(inst)
    if r.status == 0:
        return "optimal", -r.fun
    if r.status == 3:
        return "unbounded", None
    zero = S.Instance(inst.A, inst.b, tuple(0.0 for _ in inst.c), inst.senses, inst.names, inst.row_names, "x")      # HiGHS meldet "unzulässig" auch für unbeschränkte LPs: Zulässigkeit getrennt prüfen
    return ("unbounded" if _lp(zero).status == 0 else "infeasible"), None


def _exact_dual_simplex(inst, basis_cols, rule):
    """Exakter dualer Simplex für Standard-LPs (nur <=, Schlupfspalte i = n + i) ab der Basis `basis_cols`: Pfad als Liste (austretende, eintretende Spalte)."""
    def f(v):
        return F(str(float(v)))

    m, n = inst.m, inst.n
    Af = [[f(v) for v in inst.A[i]] + [F(1) if k == i else F(0) for k in range(m)] for i in range(m)]
    cost = [f(v) for v in inst.c] + [F(0)] * m
    nc, basis = n + m, list(basis_cols)
    M = [[Af[i][j] for j in basis] + Af[i] + [f(inst.b[i])] for i in range(m)]
    for col in range(m):
        p = next(r for r in range(col, m) if M[r][col] != 0)
        M[col], M[p] = M[p], M[col]
        pv = M[col][col]
        M[col] = [v / pv for v in M[col]]
        for r in range(m):
            if r != col and M[r][col] != 0:
                g = M[r][col]
                M[r] = [M[r][k] - g * M[col][k] for k in range(len(M[r]))]
    T = [row[m:] for row in M]
    path = []
    while True:
        red = [sum((cost[basis[i]] * T[i][j] for i in range(m)), F(0)) - cost[j] for j in range(nc)]
        assert all(v >= 0 for v in red)                                                     # die alte Endbasis ist dual zulässig
        rows = [i for i in range(m) if T[i][nc] < 0]
        if not rows:
            return "optimal", path
        if rule == "bland":
            i = min(rows, key=lambda q: basis[q])
        else:
            w = min(T[q][nc] for q in rows)
            i = min((q for q in rows if T[q][nc] == w), key=lambda q: basis[q])
        cand = [j for j in range(nc) if j not in basis and T[i][j] < 0]
        if not cand:
            return "infeasible", path
        rat = {j: red[j] / -T[i][j] for j in cand}
        e = min(j for j in cand if rat[j] == min(rat.values()))
        leave, pv = basis[i], T[i][e]
        T[i] = [v / pv for v in T[i]]
        for q in range(m):
            if q != i and T[q][e] != 0:
                g = T[q][e]
                T[q] = [T[q][k] - g * T[i][k] for k in range(nc + 1)]
        basis[i] = e
        path.append((leave, e))


def _le_instances():
    yield S.textbook_instance()
    yield S.centre_instance()
    yield S.degenerate_instance()
    for seed in range(14):
        yield S.generate("random", 3 + seed % 6, 3 + (seed * 2) % 6, 0.5, seed)


def test_dual_simplex_follows_the_exact_rational_path_after_rhs_changes_and_cuts():
    rng = random.Random(3)
    nonzero = 0
    for inst in _le_instances():
        _sol, tab = R.solve_cold(inst)
        m, n = inst.m, inst.n
        for _ in range(6):
            rule = rng.choice(["dantzig", "bland"])
            if rng.random() < 0.5:
                i = rng.randrange(m)
                d = rng.uniform(-1.0, 1.5) * abs(inst.b[i])
                new, basis, b = R.apply_rhs(tab, i, d), list(tab.basis), list(inst.b)
                b[i] += d
                changed = S.Instance(inst.A, tuple(b), inst.c, inst.senses, inst.names, inst.row_names, "x")
            else:
                x = np.array(tab.x())
                a = np.array([rng.random() if rng.random() < 0.6 else 0.0 for _ in range(n)]) + 0.05
                beta = rng.uniform(0.3, 1.1) * float(a @ x)
                new, basis = R.add_row(tab, a, beta), list(tab.basis) + [n + m]
                changed = S.Instance(tuple(list(inst.A) + [tuple(a)]), tuple(list(inst.b) + [beta]), inst.c, tuple(list(inst.senses) + [LE]), inst.names, tuple(list(inst.row_names) + ["r"]), "x")
            status, path = _exact_dual_simplex(changed, basis, rule)
            w = R.reoptimize(new, rule)
            if w.method == "none":
                assert status == "optimal" and path == []
                continue
            assert w.status == status and [(p["leave"], p["enter"]) for p in w.path] == path, (inst.kind, rule)
            nonzero += len(path)
    assert nonzero > 30


def test_warm_start_equals_highs_in_status_value_and_duals_for_every_change_kind():
    rng = random.Random(1)
    runs = 0
    for inst in list(_le_instances()) + [S.generate("mixed", 4 + s % 4, 3 + s % 4, 0.5, s) for s in range(10)]:
        _sol, tab = R.solve_cold(inst)
        if tab is None:
            continue
        A0, b0, c0, se = [list(r) for r in inst.A], list(inst.b), list(inst.c), list(inst.senses)
        m, n = inst.m, inst.n
        for _ in range(6):
            kind = rng.choice(["rhs", "cost", "cut", "bound", "column"])
            if kind == "rhs":
                i = rng.randrange(m)
                d = rng.uniform(-1.2, 2.0) * abs(b0[i])
                new, ref = R.apply_rhs(tab, i, d), (A0, [v + (d if k == i else 0.0) for k, v in enumerate(b0)], c0, se)
            elif kind == "cost":
                j = rng.randrange(n)
                v = c0[j] * rng.uniform(-0.5, 3.0)
                new, ref = R.apply_cost(tab, j, v), (A0, b0, [v if k == j else c for k, c in enumerate(c0)], se)
            elif kind in ("cut", "bound"):
                x = np.array(tab.x())
                if kind == "cut":
                    a = np.array([rng.random() if rng.random() < 0.6 else 0.0 for _ in range(n)]) + 0.05
                    beta = rng.uniform(0.2, 1.3) * float(a @ x)
                else:
                    j = rng.randrange(n)
                    a, beta = np.eye(n)[j], rng.uniform(0.2, 1.3) * x[j]
                new, ref = R.add_row(tab, a, beta), (A0 + [list(a)], b0 + [beta], c0, se + [LE])
            else:
                col = [rng.uniform(0, 3) for _ in range(m)]
                cost = rng.uniform(0.3, 25)
                new, ref = R.add_column(tab, col, cost), ([r + [col[i]] for i, r in enumerate(A0)], b0, c0 + [cost], se)
            changed = S.Instance(tuple(tuple(map(float, r)) for r in ref[0]), tuple(map(float, ref[1])), tuple(map(float, ref[2])), tuple(ref[3]), tuple(f"x{j}" for j in range(len(ref[2]))),
                                 tuple(f"r{i}" for i in range(len(ref[1]))), "x")
            w = R.reoptimize(new, rng.choice(["dantzig", "bland"]))
            status, opt = _status(changed)
            assert w.status == status, (inst.kind, kind)
            if status == "optimal":
                y, bb, cc = np.array(w.y), np.array(changed.b), np.array(changed.c)
                assert w.obj == pytest.approx(opt, rel=1e-6, abs=1e-6)
                assert A.primal_violation(changed, w.x) <= 1e-6 * (1 + np.abs(bb).max())
                assert A.dual_violation(changed, y) <= 1e-6 * (1 + np.abs(cc).max()) and float(y @ bb) == pytest.approx(opt, rel=1e-6, abs=1e-6)
            runs += 1
    assert runs > 150


@pytest.mark.parametrize("settings", [E.Settings("centre", resource=0), E.Settings("centre", resource=1), E.Settings("random", 10, 10, 35, resource=0)])
def test_value_curve_breakpoints_equal_the_changes_of_the_shadow_price_of_highs(settings):
    inst = E.instance_of(settings)
    i = settings.res_i
    pc = E.parametric(settings)
    prices = []
    for b in np.linspace(pc["grid"][0], pc["grid"][-1], 800):
        bb = list(inst.b)
        bb[i] = b
        r = _lp(S.Instance(inst.A, tuple(bb), inst.c, inst.senses, inst.names, inst.row_names, "x"))
        prices.append(-r.ineqlin.marginals[i])                                              # Schattenpreis = Steigung der Wertkurve
    kinks = sum(1 for a, b in zip(prices, prices[1:]) if abs(a - b) > 1e-6)
    assert pc["basis_changes"] == kinks and pc["total_warm"] >= kinks
