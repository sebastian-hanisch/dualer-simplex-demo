"""Auswertung: eine Änderung warm gegen kalt, Größe der Änderung, Ketten (rollierender Horizont, Branching-Folge) und die parametrische Wertkurve."""

import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dsx_algorithm as A
import dsx_constants as C
import dsx_reopt as R
import dsx_scenario as S


@dataclass(frozen=True)
class Settings:
    kind: str = "centre"
    m: int = C.DEFAULT_M
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    change: str = "rhs"
    level: int = 4
    resource: int = 0
    var: int = 0
    rule: str = C.DEFAULT_RULE

    @property
    def level_i(self):
        return min(self.level, len(C.LEVELS[self.change]) - 1)

    @property
    def level_value(self):
        return C.LEVELS[self.change][self.level_i]

    @property
    def res_i(self):
        return min(self.resource, instance_of(self).m - 1)

    @property
    def var_j(self):
        return min(self.var, instance_of(self).n - 1)


@lru_cache(maxsize=256)
def instance_of(settings):
    return S.generate(settings.kind, settings.m, settings.n, C.DENSITY, settings.seed)


def make_change(tab, change, level_value, i=0, j=0, rng=None):
    """Wendet eine Änderung auf das Endtableau an: gibt (neuer Tab, Beschreibung) zurück."""
    inst = tab.inst
    x, y = np.array(tab.x()), np.array(tab.y())
    if change == "rhs":
        return R.apply_rhs(tab, i, level_value / 100.0 * abs(inst.b[i])), f"{inst.row_names[i]}: b {level_value:+d} %"
    if change == "cost":
        return R.apply_cost(tab, j, inst.c[j] * (1 + level_value / 100.0)), f"{inst.names[j]}: c {level_value:+d} %"
    if change == "bound":
        return R.add_bound(tab, j, upper=level_value * x[j]), f"{inst.names[j]} ≤ {level_value:g}·x* = {level_value * x[j]:.2f}"
    if change == "row":
        rng = rng or random.Random(0)
        a = np.array([rng.random() if rng.random() < 0.6 else 0.0 for _ in range(inst.n)]) + 0.05
        return R.add_row(tab, a, level_value * float(a @ x)), f"Schnitt a·x ≤ {level_value:g}·a·x* = {level_value * float(a @ x):.2f}"
    col = 0.9 * np.array([inst.A[r][j] for r in range(inst.m)])
    return R.add_column(tab, col, level_value * float(y @ col)), f"Neuer Dienst wie {inst.names[j]} (0.9×), Deckungsbeitrag {level_value:g}·y·a = {level_value * float(y @ col):.2f}"


@dataclass
class Comparison:
    settings: Settings
    inst: object
    base_sol: object
    base_tab: object
    new_tab: object
    desc: str
    warm: object
    cold: object

    @property
    def status_equal(self):
        return self.warm.status == self.cold.status

    @property
    def obj_equal(self):
        if self.warm.status != "optimal" or self.cold.status != "optimal":
            return self.warm.status == self.cold.status
        return abs(self.warm.obj - self.cold.obj) <= 1e-6 * max(1.0, abs(self.cold.obj))


@lru_cache(maxsize=128)
def analyse(settings):
    """Grundlösung, Änderung, Warmstart (mit Tableau-Verlauf) und Neustart; None, wenn die Ausgangsinstanz kein Optimum hat."""
    inst = instance_of(settings)
    sol, tab = R.solve_cold(inst)
    if tab is None:
        return None
    rng = random.Random(f"dsx-cut-{settings.seed}-{settings.kind}")
    new, desc = make_change(tab, settings.change, settings.level_value, settings.res_i, settings.var_j, rng)
    warm = R.reoptimize(new, settings.rule, keep=inst.m <= 12)
    cold = A.solve(new.inst)
    return Comparison(settings, inst, sol, tab, new, desc, warm, cold)


def ops_per_pivot(tab):
    """Dichtes Tableau-Modell der Vorgängerstücke: (Spalten + 1) + 2 m (Spalten + 1) je Pivot."""
    return (tab.ncols + 1) + 2 * tab.m * (tab.ncols + 1)


def _instances(settings):
    if settings.kind in S.FIXTURE_KINDS:
        return [instance_of(settings)]
    return [S.generate(settings.kind, settings.m, settings.n, C.DENSITY, s) for s in C.SWEEP_SEEDS]


def _summ(rows):
    w, c = sum(r[0] for r in rows), sum(r[1] for r in rows)
    return {"warm": w, "cold": c, "ratio": w / c if c else float("nan"), "warm_ge_cold": float(np.mean([r[0] >= r[1] for r in rows])) if rows else float("nan"),
            "zero": float(np.mean([r[0] == 0 for r in rows])) if rows else float("nan"), "n": len(rows)}


@lru_cache(maxsize=128)
def sweep(settings, change):
    """Pivots warm und kalt (Summe über die Läufe) je Größe der Änderung: über die festen Instanzen (Seeds 100000-100004; Fixtures: die Instanz) und alle Ressourcen bzw. Dienste (Schnitt: je Ressource ein
    zufälliger Schnitt). Nur Läufe mit gleichem Ausgang von warm und kalt; Zeilen: Größe, Summe warm, Summe kalt, Verhältnis, Anteil warm >= kalt, Anteil ohne Pivot."""
    out = []
    for lv, value in enumerate(C.LEVELS[change]):
        rows = []
        for inst in _instances(settings):
            sol, tab = R.solve_cold(inst)
            if tab is None:
                continue
            rng = random.Random(f"dsx-sweep-{inst.kind}-{inst.m}-{inst.n}-{lv}")
            units = range(inst.m) if change == "rhs" else (range(inst.m) if change == "row" else range(inst.n))
            for u in units:
                if change == "bound" and tab.x()[u] < 1e-9:
                    continue
                i, j = (u, 0) if change == "rhs" else (0, u)
                new, _ = make_change(tab, change, value, i, j, rng)
                w, c = R.reoptimize(new, settings.rule), A.solve(new.inst)
                if w.status == c.status:
                    rows.append((w.pivots, c.pivots))
        row = _summ(rows)
        row.update({"level": lv, "value": value, "label": C.level_label(change, lv)})
        out.append(row)
    return out


@lru_cache(maxsize=128)
def chain(settings, steps=C.CHAIN_STEPS):
    """Rollierender Horizont: `steps` zufällige Änderungen einer rechten Seite (b_i wandert in ±30 % des Ausgangswerts), jeweils warm vom letzten Endtableau und kalt von Null. Gibt die Pivots je Schritt zurück."""
    inst = instance_of(settings)
    sol, tab = R.solve_cold(inst)
    if tab is None:
        return None
    rng = random.Random(f"dsx-chain-{settings.seed}-{settings.kind}")
    b0 = list(inst.b)
    cur, warm, cold, methods = tab, [], [], []
    for _ in range(steps):
        i = rng.randrange(inst.m)
        target = b0[i] * (1 + rng.uniform(-C.CHAIN_SPREAD, C.CHAIN_SPREAD))
        new = R.apply_rhs(cur, i, target - cur.inst.b[i])
        w, c = R.reoptimize(new, settings.rule), A.solve(new.inst)
        if w.status != "optimal" or c.status != "optimal":
            break
        warm.append(w.pivots)
        cold.append(c.pivots)
        methods.append(w.method)
        cur = w.tab
    return {"warm": warm, "cold": cold, "methods": methods, "cum_warm": list(np.cumsum(warm)), "cum_cold": list(np.cumsum(cold))}


@lru_cache(maxsize=128)
def dive(settings, steps=C.DIVE_STEPS):
    """Branching-Folge: nacheinander die Schranke x_j ≤ 0.8·x_j* für den Dienst mit der größten Menge (jeweils auf dem letzten Endtableau), warm gegen kalt (Neustart der kumulierten Instanz)."""
    inst = instance_of(settings)
    sol, tab = R.solve_cold(inst)
    if tab is None:
        return None
    cur, warm, cold, objs = tab, [], [], [tab.obj()]
    for _ in range(steps):
        x = np.array(cur.x())
        j = int(np.argmax(x))
        if x[j] < 1e-6:
            break
        new = R.add_bound(cur, j, upper=0.8 * x[j])
        w, c = R.reoptimize(new, settings.rule), A.solve(new.inst)
        if w.status != "optimal" or c.status != "optimal":
            break
        warm.append(w.pivots)
        cold.append(c.pivots)
        objs.append(w.obj)
        cur = w.tab
    return {"warm": warm, "cold": cold, "cum_warm": list(np.cumsum(warm)), "cum_cold": list(np.cumsum(cold)), "objs": objs}


@lru_cache(maxsize=64)
def parametric(settings, points=C.CURVE_POINTS):
    """Wertkurve z*(b_i) im Fenster [b - 0.6 s, b + 1.5 s] (s = max(1, |b|)) über `points` Punkte: kalt (jeder Punkt von Null) gegen warm (von der Ausgangsbasis aus nach oben und nach unten fortgeführt); dazu die Zahl der
    Basiswechsel entlang der Kurve."""
    inst = instance_of(settings)
    i = settings.res_i
    sol, tab = R.solve_cold(inst)
    if tab is None or inst.senses[i] != S.LE:
        return None
    b0 = inst.b[i]
    s = max(1.0, abs(b0))
    grid = np.linspace(b0 - 0.6 * s, b0 + 1.5 * s, points)
    order_up = [k for k in range(points) if grid[k] >= b0 - 1e-12]
    order_down = [k for k in range(points) if grid[k] < b0 - 1e-12][::-1]
    z, warm, cold, bases = [None] * points, [0] * points, [0] * points, {}
    for k, b in enumerate(grid):
        c = A.solve(R.apply_rhs(tab, i, b - b0).inst)
        cold[k] = c.pivots if c.status == "optimal" else 0
    for order in (order_up, order_down):
        cur, b_cur = tab, b0
        for k in order:
            new = R.apply_rhs(cur, i, grid[k] - b_cur)
            w = R.reoptimize(new, settings.rule)
            if w.status != "optimal":
                break
            z[k], warm[k] = w.obj, w.pivots
            bases[k] = tuple(sorted(w.tab.basis))
            cur, b_cur = w.tab, grid[k]
    seq = [bases[k] for k in sorted(bases)]
    changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    return {"grid": [float(v) for v in grid], "z": z, "warm": warm, "cold": cold, "basis_changes": changes, "total_warm": sum(warm), "total_cold": sum(cold)}
