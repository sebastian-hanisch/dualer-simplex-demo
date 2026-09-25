"""Instanzen der Demo Dualer Simplex und Neuoptimierung: Auslastungsplanung eines Distributionszentrums als LP  max c·x,  A x (<=|>=|=) b,  x >= 0."""

import random
from dataclasses import dataclass

import numpy as np

KINDS = ("textbook", "centre", "degenerate", "random", "mixed", "infeasible", "unbounded")
FIXTURE_KINDS = ("textbook", "centre", "degenerate", "infeasible", "unbounded")
KIND_LABELS = {"textbook": "Lehrbuchbeispiel (2 Dienste)", "centre": "Distributionszentrum (5 Dienste, 4 Ressourcen)", "degenerate": "Entartete Ecke (2 Dienste, 4 Ressourcen)", "random": "Zufall (alle Ressourcen begrenzt)",
               "mixed": "Mischung (mit Mindest- und Gleichungs-Bedingungen)", "infeasible": "Unzulässig (Widerspruch)", "unbounded": "Unbeschränkt (kein Ende)"}
LE, GE, EQ = "<=", ">=", "="


@dataclass(frozen=True)
class Instance:
    A: tuple                      # m Zeilen mit je n Koeffizienten (Tupel, damit die Instanz hashbar bleibt)
    b: tuple
    c: tuple
    senses: tuple                 # je Zeile "<=", ">=" oder "="
    names: tuple                  # Namen der n Entscheidungsvariablen (Dienste)
    row_names: tuple              # Namen der m Bedingungen (Ressourcen)
    kind: str = "custom"

    @property
    def m(self):
        return len(self.b)

    @property
    def n(self):
        return len(self.c)

    def arrays(self):
        return np.array(self.A, dtype=float).reshape(self.m, self.n), np.array(self.b, dtype=float), np.array(self.c, dtype=float)


def _inst(A, b, c, senses, names, row_names, kind):
    return Instance(tuple(tuple(float(v) for v in row) for row in A), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(names), tuple(row_names), kind)


def textbook_instance():
    """Zwei Dienste, drei Ressourcen: max 3 x1 + 5 x2;  x1 <= 4;  2 x2 <= 12;  3 x1 + 2 x2 <= 18. Optimum (2, 6) mit Wert 36, von Hand in zwei Pivots erreichbar."""
    return _inst([[1, 0], [0, 2], [3, 2]], [4, 12, 18], [3, 5], [LE] * 3, ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Lagerfläche"], "textbook")


def degenerate_instance():
    """Das Lehrbuchbeispiel mit einer vierten Ressource x1 + x2 <= 8, die durch das Optimum (2, 6) läuft: an der Ecke sind drei Bedingungen zugleich bindend (entartet), die Duale sind nicht eindeutig."""
    return _inst([[1, 0], [0, 2], [3, 2], [1, 1]], [4, 12, 18, 8], [3, 5], [LE] * 4, ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Lagerfläche", "Fahrzeugkapazität"], "degenerate")


def centre_instance():
    """Distributionszentrum mit 5 Diensten und 4 Ressourcen, rückwärts konstruiert: Optimum x = (20, 10, 10, 0, 0) mit den Schattenpreisen y = (5, 0.5, 4, 0) - drei bindende Ressourcen mit sehr verschiedenem Preis
    (Kommissionierstunden 5, Lagerfläche 0.5, Rampenzeit 4) und eine, die nie bindet (Fahrzeugkapazität, Schlupf 160). Sperrgut und Retouren lohnen sich nicht: ihre Deckungsbeiträge liegen unter dem Wert ihres Verbrauchs."""
    return _inst([[2, 1, 3, 1.5, 2.5], [3, 8, 6, 12, 2], [1, 1.5, 2, 2.5, 0.5], [4, 10, 6, 14, 3]], [80, 200, 55, 400], [15.5, 15, 26, 20, 12], [LE] * 4,
                 ["Express-Pakete", "Palettenversand", "Kühlware", "Sperrgut", "Retouren"], ["Kommissionierstunden", "Lagerfläche (m²)", "Rampenzeit (h)", "Fahrzeugkapazität"], "centre")


def infeasible_instance():
    """Widerspruch: x1 <= 4 (Rampenzeit) und x1 >= 6 (Mindestmenge) zugleich."""
    return _inst([[1, 0], [0, 2], [1, 0]], [4, 12, 6], [3, 5], [LE, LE, GE], ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Mindestmenge Express"], "infeasible")


def unbounded_instance():
    """Kein Ende in Sicht: max x1 + x2 unter x1 - x2 <= 2 und -x1 + x2 <= 3 wächst entlang der Richtung (1, 1) ohne Grenze."""
    return _inst([[1, -1], [-1, 1]], [2, 3], [1, 1], [LE, LE], ["Express-Pakete", "Palettenversand"], ["Bedingung 1", "Bedingung 2"], "unbounded")


def _names(m, n):
    return [f"Dienst {j + 1}" for j in range(n)], [f"Ressource {i + 1}" for i in range(m)]


def _coefficients(rng, m, n, density):
    """Nichtnegative Verbrauchskoeffizienten mit Dichte `density`; jede Zeile und Spalte hat mindestens einen Eintrag, Zeile 0 ist dicht (Gesamtkapazität), damit alles beschränkt bleibt."""
    A = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if rng.random() < density:
                A[i][j] = round(rng.uniform(0.5, 5.0), 2)
    for j in range(n):
        A[0][j] = round(rng.uniform(0.5, 5.0), 2)
    for i in range(m):
        if not any(A[i]):
            A[i][rng.randrange(n)] = round(rng.uniform(0.5, 5.0), 2)
    return A


def generate(kind, m, n, density, seed):
    """Deterministische Instanz je (kind, m, n, density, seed); Mersenne-Twister mit Zeichenketten-Seed, plattformstabil."""
    if kind in FIXTURE_KINDS:
        return {"textbook": textbook_instance, "centre": centre_instance, "degenerate": degenerate_instance, "infeasible": infeasible_instance, "unbounded": unbounded_instance}[kind]()
    rng = random.Random(f"dsx-{kind}-{m}-{n}-{density}-{seed}")
    names, row_names = _names(m, n)
    c = [round(rng.uniform(1.0, 10.0), 2) for _ in range(n)]
    A = _coefficients(rng, m, n, density)
    if kind == "random":
        b = [round(rng.uniform(40.0, 120.0), 1) for _ in range(m)]
        return _inst(A, b, c, [LE] * m, names, row_names, kind)
    if kind == "mixed":
        x0 = [rng.uniform(1.0, 8.0) for _ in range(n)]
        senses, b = [LE], []
        eq_left = n // 2
        for i in range(1, m):
            r = rng.random()
            sense = LE if r < 0.55 else (GE if r < 0.85 else EQ)
            if sense == EQ:
                eq_left -= 1
                if eq_left < 0:
                    sense = GE
            senses.append(sense)
        for i in range(m):
            act = sum(A[i][j] * x0[j] for j in range(n))
            slack = rng.uniform(0.5, 25.0)
            if senses[i] == EQ:
                b.append(act)                                                                    # nicht runden: sonst können Gleichungen mit gleichem Träger unvereinbar werden
                continue
            b.append(round(act + slack if senses[i] == LE else act - slack, 2))
            if senses[i] == GE and b[-1] < 0:
                b[-1] = 0.0
        return _inst(A, b, c, senses, names, row_names, kind)
    raise ValueError(kind)
