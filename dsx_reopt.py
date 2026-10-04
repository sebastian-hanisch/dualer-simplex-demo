"""Neuoptimierung ab der alten Endbasis: Änderungen direkt am Endtableau (rechte Seite, Kosten, neue Nebenbedingung, neuer Dienst), dann dualer Simplex (Basis dual zulässig, primal unzulässig) oder primaler Simplex
(primal zulässig) statt Neustart von Null. Dichtes Tableau wie in den Vorgängerstücken; Pivots sind das Maß."""

from dataclasses import dataclass, field

import numpy as np

import dsx_algorithm as A
import dsx_scenario as S

TOL = 1e-9
STALL_LIMIT = 25
MAX_PIVOTS = 20_000


@dataclass
class Tab:
    T: object                                # (m + 1) x (ncols + 1): Bedingungen und Zielzeile, rechts die rechte Seite
    basis: list                              # Basisspalte je Zeile
    init_basis: list                         # Spalten der Anfangsbasis (Einheitsmatrix): darunter steht B^-1
    struct: list                             # Spaltenindizes der Entscheidungsvariablen (in der Reihenfolge der Instanz)
    art: set                                 # künstliche Spalten (nie eintretend)
    sign: list                               # Zeilenvorzeichen der Standardform
    cvec: object                             # Kostenvektor über alle Spalten (0 außer bei Entscheidungsvariablen)
    inst: object                             # die zugehörige Instanz (für Neustart und Gegenprobe)
    names: list = field(default_factory=list)  # Spaltennamen (Dienste, Schlupf s_i, Überschuss e_i, künstliche a_i)

    @property
    def m(self):
        return len(self.basis)

    @property
    def ncols(self):
        return self.T.shape[1] - 1

    def copy(self):
        return Tab(self.T.copy(), list(self.basis), list(self.init_basis), list(self.struct), set(self.art), list(self.sign), self.cvec.copy(), self.inst, list(self.names))

    def allowed(self):
        return [j for j in range(self.ncols) if j not in self.art]

    def x(self):
        xf = np.zeros(self.ncols)
        for i, j in enumerate(self.basis):
            xf[j] = self.T[i, -1]
        return tuple(float(xf[j]) for j in self.struct)

    def obj(self):
        return float(self.T[self.m, -1])

    def y(self):
        """Schattenpreise je ursprünglicher Zeile (Ableitung des Optimums nach b_i)."""
        return tuple(float(self.T[self.m, self.init_basis[i]] * self.sign[i]) for i in range(self.m))

    def binv(self):
        return self.T[:self.m, self.init_basis]

    def primal_infeasibility(self):
        """Summe der negativen rechten Seiten (0 = primal zulässig)."""
        rhs = self.T[:self.m, -1]
        return float(-np.minimum(rhs, 0.0).sum())

    def is_primal_feasible(self):
        rhs = self.T[:self.m, -1]
        return all((rhs[i] >= -TOL) if self.basis[i] not in self.art else abs(rhs[i]) <= TOL for i in range(self.m))

    def is_dual_feasible(self):
        return all(self.T[self.m, j] >= -TOL for j in self.allowed())


def tab_from_solution(inst, sol):
    """Endtableau einer Lösung als `Tab` (nur bei Optimum)."""
    info = sol.info
    n = info["n"]
    cvec = np.zeros(info["ncols"])
    cvec[:n] = info["c"]
    names = [nm if nm is not None else f"c{j}" for j, nm in enumerate(info["names"])]
    return Tab(sol.T.copy(), list(sol.basis), list(sol.init_basis), list(range(n)), set(info["art_col"].values()), list(info["sign"]), cvec, inst, names)


def solve_cold(inst):
    """Neustart: Zwei-Phasen-Simplex von Null; gibt (Solution, Tab oder None) zurück."""
    sol = A.solve(inst)
    return sol, (tab_from_solution(inst, sol) if sol.status == "optimal" else None)


def _zero_row(tab):
    T, m = tab.T, tab.m
    cB = tab.cvec[tab.basis]
    T[m, :-1] = cB @ T[:m, :-1] - tab.cvec
    T[m, -1] = float(cB @ T[:m, -1])


def _with_row(inst, a, beta, name):
    A_ = tuple(list(inst.A) + [tuple(float(v) for v in a)])
    return S.Instance(A_, tuple(list(inst.b) + [float(beta)]), inst.c, tuple(list(inst.senses) + [S.LE]), inst.names, tuple(list(inst.row_names) + [name]), inst.kind)


# --- Änderungen am Endtableau ------------------------------------------------------------------------------------------------------------------------


def apply_rhs(tab, i, delta):
    """b_i + delta: x_B ändert sich um sign_i·delta·B^-1 e_i; die Zielzeile bleibt (dual zulässig), die rechte Seite kann negativ werden."""
    new = tab.copy()
    new.T[:new.m, -1] += new.sign[i] * delta * tab.binv()[:, i]
    _zero_row(new)
    b = list(tab.inst.b)
    b[i] += delta
    new.inst = S.Instance(tab.inst.A, tuple(b), tab.inst.c, tab.inst.senses, tab.inst.names, tab.inst.row_names, tab.inst.kind)
    return new


def apply_cost(tab, j, value):
    """Deckungsbeitrag c_j = value: die rechte Seite bleibt (primal zulässig), die Zielzeile wird neu berechnet."""
    new = tab.copy()
    new.cvec[new.struct[j]] = value
    _zero_row(new)
    c = list(tab.inst.c)
    c[j] = float(value)
    new.inst = S.Instance(tab.inst.A, tab.inst.b, tuple(c), tab.inst.senses, tab.inst.names, tab.inst.row_names, tab.inst.kind)
    return new


def add_row(tab, a, beta, name="Schnitt"):
    """Neue Bedingung a·x <= beta (a über die Entscheidungsvariablen): neue Schlupfspalte als Basisvariable der neuen Zeile, die Zeile wird mit den Basiszeilen der Strukturvariablen ausgerechnet.
    Verletzt der alte Optimalpunkt die Bedingung, ist die neue rechte Seite negativ (primal unzulässig, dual zulässig)."""
    m, ncols = tab.m, tab.ncols
    T2 = np.zeros((m + 2, ncols + 2))
    T2[:m, :ncols] = tab.T[:m, :-1]
    T2[:m, -1] = tab.T[:m, -1]
    row = np.zeros(ncols + 2)
    for k, col in enumerate(tab.struct):
        row[col] = a[k]
    row[ncols] = 1.0
    row[-1] = beta
    for p, j in enumerate(tab.basis):
        if j in tab.struct and row[j] != 0.0:
            row -= row[j] * T2[p]
    T2[m] = row
    new = Tab(T2, list(tab.basis) + [ncols], list(tab.init_basis) + [ncols], list(tab.struct), set(tab.art), list(tab.sign) + [1], np.append(tab.cvec, 0.0), _with_row(tab.inst, a, beta, name), list(tab.names) + [f"s{m + 1}"])
    _zero_row(new)
    return new


def add_bound(tab, j, upper=None, lower=None):
    """Schranke x_j <= upper bzw. x_j >= lower als neue Zeile (Branching); gibt den neuen Tab zurück."""
    n = len(tab.struct)
    e = np.zeros(n)
    e[j] = 1.0
    if upper is not None:
        return add_row(tab, e, upper, f"{tab.inst.names[j]} <= {upper:g}")
    return add_row(tab, -e, -lower, f"{tab.inst.names[j]} >= {lower:g}")


def add_column(tab, a_col, cost, name="Neuer Dienst"):
    """Neuer Dienst mit Verbrauchsspalte a_col (je Zeile der Instanz) und Deckungsbeitrag cost: die Spalte im Tableau ist B^-1 (sign · a_col), die Zielzeile bekommt die reduzierten Kosten y·a − c."""
    m, ncols = tab.m, tab.ncols
    col = tab.binv() @ (np.array(tab.sign, dtype=float) * np.asarray(a_col, dtype=float))
    T2 = np.zeros((m + 1, ncols + 2))
    T2[:, :ncols] = tab.T[:, :-1]
    T2[:m, ncols] = col
    T2[:, -1] = tab.T[:, -1]
    new = Tab(T2, list(tab.basis), list(tab.init_basis), list(tab.struct) + [ncols], set(tab.art), list(tab.sign), np.append(tab.cvec, float(cost)), tab.inst, list(tab.names) + [name])
    _zero_row(new)
    inst = tab.inst
    A_ = tuple(tuple(list(row) + [float(a_col[i])]) for i, row in enumerate(inst.A))
    new.inst = S.Instance(A_, inst.b, tuple(list(inst.c) + [float(cost)]), inst.senses, tuple(list(inst.names) + [name]), inst.row_names, inst.kind)
    return new


# --- Pivot und Simplex-Schleifen -------------------------------------------------------------------------------------------------------------------


def _pivot(T, row, col):
    T[row] /= T[row, col]
    factors = T[:, col].copy()
    factors[row] = 0.0
    idx = np.nonzero(factors)[0]
    if len(idx):
        T[idx] -= np.outer(factors[idx], T[row])


@dataclass
class Reopt:
    status: str                              # "optimal" | "infeasible" | "unbounded" | "limit"
    method: str                              # "none" (schon optimal) | "dual" | "primal" | "cold" (Rückfall auf Neustart)
    pivots: int = 0
    path: list = field(default_factory=list)
    tab: object = None                       # Endtableau bei Optimum
    x: tuple = ()
    obj: float = float("nan")
    y: tuple = ()
    note: str = ""


def _record(path, tab, row, col, leave, keep, ratio):
    entry = {"row": row, "col": col, "leave": leave, "enter": col, "ratio": float(ratio), "obj": tab.obj(), "infeasibility": tab.primal_infeasibility()}
    if keep:
        entry["T"] = tab.T.copy()
        entry["basis"] = list(tab.basis)
    path.append(entry)


def dual_simplex(tab, rule="dantzig", keep=False, max_pivots=MAX_PIVOTS):
    """Dualer Simplex ab einer dual zulässigen Basis: Zeile mit negativer rechter Seite (dantzig: die kleinste; bland: kleinster Basisindex), Spalte nach dem dualen Quotiententest min r_j / (-a_ij) über a_ij < 0;
    keine Kandidaten in der Zeile heißt primal unzulässig. Bei mehr als 25 Nullschritten in Folge gilt bis zum nächsten echten Schritt Bland."""
    if not tab.is_dual_feasible():
        raise ValueError("Basis ist nicht dual zulässig")
    tab = tab.copy()
    T, m = tab.T, tab.m
    allowed = tab.allowed()
    path, stall = [], 0
    if keep:
        path.append({"T": T.copy(), "basis": list(tab.basis), "start": True, "obj": tab.obj(), "infeasibility": tab.primal_infeasibility()})
    while True:
        rows = [i for i in range(m) if tab.basis[i] not in tab.art and T[i, -1] < -TOL]
        if not rows:
            if any(tab.basis[i] in tab.art and abs(T[i, -1]) > TOL for i in range(m)):
                return Reopt("optimal", "cold", note="künstliche Variable ungleich 0 in der Basis")
            return Reopt("optimal", "dual", len(path) - (1 if keep else 0), path, tab, tab.x(), tab.obj(), tab.y())
        if rule == "bland" or stall >= STALL_LIMIT:
            i = min(rows, key=lambda r: tab.basis[r])
        else:
            worst = min(T[r, -1] for r in rows)
            i = min((r for r in rows if T[r, -1] <= worst + TOL), key=lambda r: tab.basis[r])
        basic = set(tab.basis)
        cand = [j for j in allowed if j not in basic and T[i, j] < -TOL]
        if not cand:
            return Reopt("infeasible", "dual", len(path) - (1 if keep else 0), path, None, (), float("nan"), (), "Zeile ohne negativen Eintrag")
        ratios = {j: max(T[m, j], 0.0) / -T[i, j] for j in cand}
        rmin = min(ratios.values())
        col = next(j for j in cand if ratios[j] <= rmin + TOL * max(1.0, abs(rmin)))
        leave = tab.basis[i]
        _pivot(T, i, col)
        tab.basis[i] = col
        stall = stall + 1 if rmin <= TOL else 0
        _record(path, tab, i, col, leave, keep, rmin)
        if len(path) > max_pivots:
            return Reopt("limit", "dual", len(path), path)


def primal_simplex(tab, keep=False, max_pivots=MAX_PIVOTS):
    """Primaler Simplex ab einer primal zulässigen Basis (Phase 2 der Vorgängerstücke): Dantzig-Regel, Quotiententest mit kleinstem Index, Bland-Notbremse nach 25 Nullschritten."""
    tab = tab.copy()
    T, m = tab.T, tab.m
    allowed = tab.allowed()
    path, stall = [], 0
    if keep:
        path.append({"T": T.copy(), "basis": list(tab.basis), "start": True, "obj": tab.obj(), "infeasibility": 0.0})
    while True:
        r = T[m, :-1]
        cand = [j for j in allowed if r[j] < -TOL]
        if not cand:
            return Reopt("optimal", "primal", len(path) - (1 if keep else 0), path, tab, tab.x(), tab.obj(), tab.y())
        if stall >= STALL_LIMIT:
            enter = cand[0]
        else:
            best = min(r[j] for j in cand)
            enter = next(j for j in cand if r[j] <= best + 1e-9 * max(1.0, abs(best)))
        col = T[:m, enter]
        pos = [i for i in range(m) if col[i] > TOL]
        if not pos:
            return Reopt("unbounded", "primal", len(path) - (1 if keep else 0), path, None, (), float("nan"), (), "Spalte ohne positiven Eintrag")
        ratios = {i: T[i, -1] / col[i] for i in pos}
        rmin = min(ratios.values())
        tied = [i for i in pos if ratios[i] <= rmin + TOL * max(1.0, abs(rmin))]
        leave_row = min(tied, key=lambda i: tab.basis[i])
        leave = tab.basis[leave_row]
        _pivot(T, leave_row, enter)
        tab.basis[leave_row] = enter
        stall = stall + 1 if max(rmin, 0.0) <= TOL else 0
        _record(path, tab, leave_row, enter, leave, keep, rmin)
        if len(path) > max_pivots:
            return Reopt("limit", "primal", len(path), path)


def reoptimize(tab, rule="dantzig", keep=False):
    """Wählt nach dem Zustand der Basis: schon optimal (0 Pivots), dual zulässig (dualer Simplex), primal zulässig (primaler Simplex); ist beides verletzt oder liegt eine künstliche Variable ungleich 0 in der Basis,
    Rückfall auf den Neustart (Methode "cold", nie stumm)."""
    art_bad = any(tab.basis[i] in tab.art and abs(tab.T[i, -1]) > TOL for i in range(tab.m))
    if art_bad:
        return _cold(tab, "künstliche Variable ungleich 0 in der Basis")
    pf, df = tab.is_primal_feasible(), tab.is_dual_feasible()
    if pf and df:
        return Reopt("optimal", "none", 0, [], tab, tab.x(), tab.obj(), tab.y())
    if df:
        res = dual_simplex(tab, rule, keep)
        return res if res.method != "cold" else _cold(tab, res.note)
    if pf:
        return primal_simplex(tab, keep)
    return _cold(tab, "weder primal noch dual zulässig")


def _cold(tab, note):
    sol, new = solve_cold(tab.inst)
    if sol.status != "optimal":
        return Reopt(sol.status, "cold", sol.pivots, [], None, (), float("nan"), (), note)
    return Reopt("optimal", "cold", sol.pivots, [], new, new.x(), new.obj(), new.y(), note)


def rhs_breakpoints(tab, i, b_to, rule="dantzig"):
    """Zahl der Knickpunkte der Wertkurve z*(b_i) zwischen dem Ausgangswert und `b_to` (nach oben oder unten): vom Endtableau aus bis an die Grenze des Bereichs, in dem die Basis zulässig bleibt (x_B + t * d x_B / d b >= 0),
    dann knapp darüber hinaus neu optimieren; ein Knick ist ein Wechsel des Schattenpreises y_i. Unabhängig von einem Raster der Kurve (zwei Knicke zwischen zwei Rasterpunkten zählen beide).
    Gibt None zurück, wenn eine künstliche Variable in der Basis steht (dann gibt es keine einfache Bereichsgrenze)."""
    cur, step = tab, (1.0 if b_to >= tab.inst.b[i] else -1.0)
    count = 0
    for _ in range(10_000):
        if any(cur.basis[r] in cur.art for r in range(cur.m)):
            return None
        b = cur.inst.b[i]
        d = step * cur.sign[i] * cur.binv()[:, i]                                          # Änderung von x_B je Einheit Fortschritt in Richtung b_to
        x = cur.T[:cur.m, -1]
        limits = [max(x[r], 0.0) / -d[r] for r in range(cur.m) if d[r] < -TOL]
        if not limits:
            return count
        b_lim = b + step * min(limits)
        if step * (b_lim - b_to) >= -TOL:
            return count
        nxt = b_lim + step * 1e-7 * max(1.0, abs(b_lim))
        res = reoptimize(apply_rhs(cur, i, nxt - b), rule)
        if res.status != "optimal" or res.method == "cold":
            return count
        count += abs(res.y[i] - cur.y()[i]) > 1e-9
        cur = res.tab
    return count
