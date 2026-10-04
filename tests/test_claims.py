"""Jede Zahl aus README und App über die echten Auswertungsfunktionen (ganzzahlige Pivotzahlen, Anteile als gerundete Prozent)."""

import pytest

import dsx_constants as C
import dsx_evaluation as ev
from dsx_evaluation import Settings

RANDOM10 = Settings("random", 10, 10, 35)
CENTRE = Settings("centre")


def _rows(settings, change):
    return [(r["label"], r["n"], r["warm"], r["cold"], round(r["ratio"], 2), round(100 * r["zero"]), round(100 * r["warm_ge_cold"])) for r in ev.sweep(settings, change)]


def test_readme_random10_rhs_sweep():
    assert _rows(RANDOM10, "rhs") == [("-50 %", 50, 56, 271, 0.21, 38, 8), ("-25 %", 50, 29, 281, 0.10, 52, 0), ("-10 %", 50, 6, 279, 0.02, 90, 0), ("+10 %", 50, 5, 278, 0.02, 90, 0),
                                      ("+25 %", 50, 21, 284, 0.07, 76, 2), ("+50 %", 50, 32, 286, 0.11, 66, 4), ("+100 %", 50, 52, 280, 0.19, 66, 6)]


def test_readme_random10_cut_and_bound_sweeps():
    assert _rows(RANDOM10, "row") == [("λ = 0.95", 50, 69, 300, 0.23, 0, 2), ("λ = 0.9", 50, 73, 292, 0.25, 0, 2), ("λ = 0.75", 50, 97, 286, 0.34, 0, 4), ("λ = 0.5", 50, 144, 239, 0.60, 0, 26)]
    assert _rows(RANDOM10, "bound") == [("λ = 0.95", 18, 18, 111, 0.16, 0, 0), ("λ = 0.9", 18, 18, 111, 0.16, 0, 0), ("λ = 0.75", 18, 24, 106, 0.23, 0, 0), ("λ = 0.5", 18, 30, 103, 0.29, 0, 0)]


def test_readme_random10_cost_and_column_sweeps():
    assert _rows(RANDOM10, "cost") == [("-50 %", 50, 33, 262, 0.13, 66, 6), ("-25 %", 50, 16, 265, 0.06, 76, 0), ("-10 %", 50, 8, 274, 0.03, 86, 0), ("+10 %", 50, 9, 273, 0.03, 82, 0),
                                       ("+25 %", 50, 18, 269, 0.07, 72, 0), ("+50 %", 50, 41, 264, 0.16, 54, 4), ("+100 %", 50, 67, 250, 0.27, 44, 14)]
    assert _rows(RANDOM10, "column") == [("f = 0.8", 50, 0, 289, 0.0, 100, 0), ("f = 1.05", 50, 57, 288, 0.20, 0, 2), ("f = 1.25", 50, 83, 257, 0.32, 0, 8), ("f = 2", 50, 151, 214, 0.71, 0, 40)]


def test_readme_centre_sweeps():
    assert _rows(CENTRE, "rhs") == [("-50 %", 4, 7, 13, 0.54, 0, 25), ("-25 %", 4, 4, 14, 0.29, 25, 0), ("-10 %", 4, 1, 16, 0.06, 75, 0), ("+10 %", 4, 2, 15, 0.13, 50, 0),
                                    ("+25 %", 4, 3, 15, 0.20, 25, 0), ("+50 %", 4, 5, 14, 0.36, 25, 0), ("+100 %", 4, 5, 14, 0.36, 25, 0)]
    assert _rows(CENTRE, "column") == [("f = 0.8", 5, 0, 20, 0.0, 100, 0), ("f = 1.05", 5, 6, 19, 0.32, 0, 0), ("f = 1.25", 5, 9, 20, 0.45, 0, 20), ("f = 2", 5, 13, 10, 1.30, 0, 60)]
    assert _rows(CENTRE, "row")[-1] == ("λ = 0.5", 4, 6, 13, 0.46, 0, 25)


def test_readme_other_instance_types():
    mixed, deg, small = Settings("mixed", 8, 8, 35), Settings("degenerate", 6, 8, 35), Settings("random", 6, 8, 35)
    assert _rows(mixed, "rhs")[2] == ("-10 %", 40, 5, 416, 0.01, 88, 0) and _rows(mixed, "row")[-1] == ("λ = 0.5", 40, 107, 379, 0.28, 0, 0) and _rows(mixed, "column")[-1] == ("f = 2", 40, 94, 410, 0.23, 0, 0)
    assert _rows(deg, "row")[2:] == [("λ = 0.75", 4, 6, 6, 1.0, 0, 50), ("λ = 0.5", 4, 7, 7, 1.0, 0, 75)]
    assert _rows(small, "rhs")[2] == ("-10 %", 30, 2, 103, 0.02, 93, 0) and _rows(small, "column")[-1] == ("f = 2", 40, 87, 121, 0.72, 0, 50)


def test_readme_chains():
    got = {}
    for name, s in (("centre", CENTRE), ("random10", RANDOM10), ("mixed8", Settings("mixed", 8, 8, 35)), ("random12", Settings("random", 12, 12, 35)), ("degenerate", Settings("degenerate", 6, 8, 35)),
                    ("textbook", Settings("textbook"))):
        c = ev.chain(s)
        got[name] = (sum(c["warm"]), sum(c["cold"]), sum(1 for w in c["warm"] if w == 0))
    assert got == {"centre": (20, 184, 36), "random10": (6, 240, 45), "mixed8": (18, 346, 40), "random12": (24, 309, 37), "degenerate": (15, 84, 36), "textbook": (4, 98, 46)}


def test_readme_branching_dive():
    d = ev.dive(CENTRE)
    assert d["warm"] == [1, 1, 2, 1, 1, 1, 1, 1] and sum(d["cold"]) == 41 and round(d["objs"][0], 2) == 720.0 and round(d["objs"][-1], 2) == 654.90
    r = ev.dive(RANDOM10)
    assert (sum(r["warm"]), sum(r["cold"])) == (11, 38)
    m = ev.dive(Settings("mixed", 8, 8, 35))
    assert (sum(m["warm"]), sum(m["cold"])) == (10, 73)


def test_readme_parametric_curves():
    got = {}
    for name, s in (("c0", Settings("centre", resource=0)), ("c1", Settings("centre", resource=1)), ("c2", Settings("centre", resource=2)), ("r0", Settings("random", 10, 10, 35, resource=0)),
                    ("r3", Settings("random", 10, 10, 35, resource=3))):
        p = ev.parametric(s)
        got[name] = (p["total_warm"], p["total_cold"], p["basis_changes"])
    assert got == {"c0": (3, 134, 3), "c1": (4, 155, 4), "c2": (3, 130, 3), "r0": (5, 207, 5), "r3": (4, 154, 4)}


def test_readme_dantzig_against_bland_for_the_dual_row_choice():
    dantzig = [(r["warm"]) for r in ev.sweep(Settings("random", 10, 10, 35, "rhs"), "rhs")]
    bland = [(r["warm"]) for r in ev.sweep(Settings("random", 10, 10, 35, "rhs", rule="bland"), "rhs")]
    assert dantzig == [56, 29, 6, 5, 21, 32, 52] and bland == [56, 31, 6, 5, 24, 37, 51]
    dantzig_cut = [r["warm"] for r in ev.sweep(Settings("random", 10, 10, 35, "row"), "row")]
    bland_cut = [r["warm"] for r in ev.sweep(Settings("random", 10, 10, 35, "row", rule="bland"), "row")]
    assert dantzig_cut == [69, 73, 97, 144] and bland_cut == [74, 74, 100, 147]


def test_all_change_types_use_the_documented_simplex():
    assert C.CHANGE_METHOD == {"rhs": "dual", "row": "dual", "bound": "dual", "cost": "primal", "column": "primal"}
    for change, method in C.CHANGE_METHOD.items():
        a = ev.analyse(Settings("centre", change=change, level=2, var=0))
        assert a.warm.method in (method, "none") and a.warm.pivots <= a.cold.pivots


@pytest.mark.parametrize("change", C.CHANGES)
def test_warm_never_disagrees_with_cold_in_the_sweeps(change):
    for s in (RANDOM10, Settings("mixed", 8, 8, 35), CENTRE):
        for level in range(len(C.LEVELS[change])):
            a = ev.analyse(Settings(s.kind, s.m, s.n, s.seed, change, level, 0, 0))
            assert a is None or (a.status_equal and a.obj_equal)
