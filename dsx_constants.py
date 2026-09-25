"""Konstanten der Demo Dualer Simplex und Neuoptimierung: Regler-Bereiche, Änderungsarten und Größen, feste Instanzen, Presets."""
M_MIN, M_MAX, DEFAULT_M = 2, 12, 6
N_MIN, N_MAX, DEFAULT_N = 2, 12, 8
DENSITY = 0.5                                        # Dichte der Zufalls- und Mischinstanzen (fest)
SEED_MAX = 999999
DEFAULT_SEED = 35
INDEX_MAX = 11                                       # Ressourcen- bzw. Dienst-Auswahl (wird auf die Instanzgröße begrenzt)
CHANGES = ("rhs", "row", "bound", "cost", "column")
CHANGE_LABELS = {"rhs": "Rechte Seite ändert sich (b_i)", "row": "Neue Nebenbedingung (Schnitt)", "bound": "Schranke x_j ≤ λ·x_j* (Branching)", "cost": "Deckungsbeitrag ändert sich (c_j)",
                 "column": "Neuer Dienst (neue Spalte)"}
CHANGE_SHORT = {"rhs": "Rechte Seite", "row": "Schnitt", "bound": "Schranke", "cost": "Kosten", "column": "Neuer Dienst"}
CHANGE_METHOD = {"rhs": "dual", "row": "dual", "bound": "dual", "cost": "primal", "column": "primal"}
# Größen der Änderung je Art (Index in den Regler): Prozent, λ (Schnitt: β = λ·a·x*), Faktor f (Deckungsbeitrag des neuen Dienstes = f · y·a)
LEVELS = {"rhs": (-50, -25, -10, 10, 25, 50, 100), "row": (0.95, 0.9, 0.75, 0.5), "bound": (0.95, 0.9, 0.75, 0.5), "cost": (-50, -25, -10, 10, 25, 50, 100), "column": (0.8, 1.05, 1.25, 2.0)}
DEFAULT_LEVEL = {"rhs": 4, "row": 2, "bound": 2, "cost": 4, "column": 2}
LEVEL_MAX = 6
RULES = ("dantzig", "bland")
RULE_LABELS = {"dantzig": "Dantzig (kleinste rechte Seite)", "bland": "Bland (kleinster Index)"}
DEFAULT_RULE = "dantzig"
STEPS = {1: "1 · Ein dualer Pivot", 2: "2 · Warm gegen kalt", 3: "3 · Ketten", 4: "4 · Wertkurve"}
SWEEP_SEEDS = tuple(range(100000, 100005))
CHAIN_STEPS = 50
CHAIN_SPREAD = 0.3                                   # rollierender Horizont: b_i wandert innerhalb von ±30 % des Ausgangswerts
DIVE_STEPS = 8
CURVE_POINTS = 41
_BASE = {"kind": "centre", "m": 6, "n": 8, "seed": 35, "change": "rhs", "level": 4, "res": 0, "var": 0, "rule": "dantzig", "step": 2}
PRESETS = {
    "Lehrbuch: ein dualer Pivot": {**_BASE, "kind": "textbook", "res": 2, "level": 5, "step": 1, "pivot_k": 1},
    "Zentrum: Kommissionierstunden +50 %": {**_BASE, "res": 0, "level": 5},
    "Innerhalb des Bereichs: null Pivots": {**_BASE, "res": 1, "level": 2},
    "Schnitt schneidet den Optimalpunkt ab": {**_BASE, "change": "row", "level": 2},
    "Schranke (Branching)": {**_BASE, "change": "bound", "level": 2, "var": 0},
    "Kosten ändern sich: primaler Warmstart": {**_BASE, "change": "cost", "level": 5, "var": 3},
    "Neuer Dienst": {**_BASE, "change": "column", "level": 2, "var": 0},
    "Neustart nicht schlechter": {**_BASE, "kind": "random", "m": 8, "n": 8, "seed": 22, "change": "row", "level": 3},
    "Kette von 50 Änderungen": {**_BASE, "kind": "random", "m": 10, "n": 10, "step": 3},
    "Wertkurve: ein Pivot je Knickpunkt": {**_BASE, "res": 0, "step": 4},
}
PRESET_HELP = {
    "Lehrbuch: ein dualer Pivot": "Lagerfläche +50 % (18 → 27) liegt außerhalb des Bereichs [12, 24]: die alte Basis ist dual zulässig, aber primal unzulässig (Summe der negativen rechten Seiten 1). Ein dualer Pivot senkt den Zielwert von 45 auf das neue Optimum 42; der Neustart braucht 2 Pivots.",
    "Zentrum: Kommissionierstunden +50 %": "Kommissionierstunden +50 % (80 → 120, Bereich [70, 90]): 2 duale Pivots (Zielwert 920, 893.75, 880.83) gegen 3 beim Neustart; das neue Optimum ist 880.83 (vorher 720).",
    "Innerhalb des Bereichs: null Pivots": "Lagerfläche −10 % (200 → 180) liegt im Bereich [172, 217.5]: die Basis bleibt optimal, 0 Pivots, der Optimalwert sinkt um Schattenpreis mal Änderung (0.5 · 20) auf 710. Der Neustart braucht 4 Pivots.",
    "Schnitt schneidet den Optimalpunkt ab": "Ein Schnitt a·x ≤ 0.75·a·x* (rechte Seite 20.54) schneidet den alten Optimalpunkt ab: 1 dualer Pivot (Zielwert 720 → 716.10) gegen 4 beim Neustart.",
    "Schranke (Branching)": "Express-Pakete ≤ 0.75 · 20 = 15 (wie ein Branching-Schritt): 1 dualer Pivot (720 → 716.5) gegen 4 beim Neustart.",
    "Kosten ändern sich: primaler Warmstart": "Der Deckungsbeitrag von Sperrgut steigt um 50 % (20 → 30, über der Grenze 23.5): die Basis bleibt primal zulässig, 2 primale Pivots (720, 770.56, 770.77) gegen 5 beim Neustart.",
    "Neuer Dienst": "Neuer Dienst nach dem Vorbild der Express-Pakete (0.9 mal) mit Deckungsbeitrag 1.25 · y·a = 17.44: 2 primale Pivots (720, 797.5, 840.38) gegen 5 beim Neustart.",
    "Neustart nicht schlechter": "Zufallsinstanz 8 × 8 (Seed 22): ein Schnitt mit λ = 0.5 braucht 4 duale Pivots, der Neustart nur 1. Bei großen Änderungen holt der Neustart auf: beim Schnitt mit λ = 0.5 ist der Warmstart bei Zufall 10 × 10 in 26 % der Fälle nicht besser, bei der entarteten Ecke in 75 %.",
    "Kette von 50 Änderungen": "Zufall 10 × 10: 50 zufällige Änderungen je einer rechten Seite (±30 %), jeweils vom letzten Endtableau: zusammen 6 Pivots warm gegen 240 kalt; in 45 von 50 Schritten war kein Pivot nötig.",
    "Wertkurve: ein Pivot je Knickpunkt": "Zentrum, Kommissionierstunden: die Wertkurve über 41 Punkte braucht warm 3 Pivots gegen 134 kalt und hat 3 Basiswechsel - ein Pivot je Knickpunkt.",
}


def level_label(change, level):
    v = LEVELS[change][min(level, len(LEVELS[change]) - 1)]
    if change in ("rhs", "cost"):
        return f"{v:+d} %"
    if change == "column":
        return f"f = {v:g}"
    return f"λ = {v:g}"
