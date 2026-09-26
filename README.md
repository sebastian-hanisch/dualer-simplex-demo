# Dualer Simplex und Neuoptimierung – nach einer Änderung von der alten Basis weiterrechnen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-dualer-simplex-demo.streamlit.app/)**


Sechstes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind von [lp-dualitaet-demo](https://github.com/sebastian-hanisch/lp-dualitaet-demo). Das Stück davor hat die Wertfunktion und den Zukauf durch **Neulösung von Null** gerechnet. Muss man nach einer Änderung wirklich neu anfangen? Nein: ändert sich die **rechte Seite** (eine Kapazität) oder kommt eine **Nebenbedingung** dazu (Schnitt, Branching-Schranke), bleibt die alte Endbasis **dual zulässig** – alle reduzierten Kosten sind noch ≥ 0 –, aber sie ist **primal unzulässig**: ein x_B ist negativ. Für genau diese Lage ist der **duale Simplex** gebaut (Lemke 1954). Ändern sich **Deckungsbeiträge** oder kommt ein **neuer Dienst** dazu, bleibt die Basis primal zulässig, und der **primale Simplex** rechnet von der alten Basis weiter. Die Demo zählt die Pivots gegen den **Neustart von Null** (Zwei-Phasen-Simplex desselben Lösers). Vier Fragen, alle gemessen: **(1) Ein dualer Pivot** – was passiert in einem Schritt? **(2) Warm gegen kalt** – wie viele Pivots spart der Warmstart, und wann nicht? **(3) Ketten** – 50 Änderungen nacheinander (rollierender Horizont, Branching-Folge). **(4) Wertkurve** – die Kurve aus dem letzten Stück als warm fortgeführte Folge.

**Einordnung in die Reihe:** die Reihe hat elf Stücke, dies ist das sechste (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [gebaut: revised-simplex-demo]  →  [nicht gebaut]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [gebaut: lp-dualitaet-demo]  →  [DIESES STÜCK]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [nicht gebaut]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP  [nicht gebaut]
```

Ergebnis in Kürze: **Der Warmstart spart fast immer sehr viel, aber nicht immer, und je größer die Änderung, desto weniger.** Auf Zufallsinstanzen 10 × 10 braucht der duale Simplex nach einer Änderung der rechten Seite um 10 % **5 bis 6 Pivots statt 278 bis 279 beim Neustart** (2 % der Pivots, in 90 % der Fälle gar keiner); bei +100 % sind es 52 gegen 280 (19 %). Ein Schnitt, der den alten Optimalpunkt auf das 0.5-fache abschneidet, kostet 144 gegen 239 Pivots (60 %), und **in 26 % der Fälle ist der Neustart nicht schlechter**; ein neuer Dienst mit doppeltem Deckungsbeitrag kostet 151 gegen 214 (71 %), in 40 % der Fälle ist der Neustart nicht schlechter, im Zentrum sogar **13 gegen 10 Pivots (Warmstart schlechter)**. **Ketten** sind die eigentliche Stärke: 50 zufällige Änderungen einer rechten Seite kosten zusammen **6 Pivots warm gegen 240 kalt**, in 45 von 50 Schritten war kein Pivot nötig. Die **Wertkurve** des Zentrums (41 Punkte) braucht warm **3 Pivots gegen 134 kalt**, bei 3 Basiswechseln: ein Pivot je Knickpunkt.

| Frage | Ergebnis (Auslastungsplanung als Standard-LP max c·x; Kreuzprüfung mit HiGHS und Neustart; Zufallsinstanzen 10 × 10: je 5 bis 50 Läufe, Seed 35; vollständig deterministisch) |
|---|---|
| **Stimmt der Warmstart?** | ✅ Auf 1962 Änderungen (rechte Seite, Kosten, Schnitte, Schranken, neue Dienste; Lehrbuch, Zentrum, entartete Ecke, Zufall, Mischung mit ≥): Status, Optimalwert und Duale gleich Neustart und HiGHS, primal und dual zulässig, c·x = b·y. Bei unzulässigen Änderungen erkennt der duale Simplex die Unzulässigkeit an einer Zeile ohne negativen Eintrag; ein neuer Dienst ohne positiven Eintrag wird als unbeschränkt erkannt |
| **Ein dualer Pivot (Lehrbuch)** | Lagerfläche +50 % (18 → 27) liegt außerhalb des Bereichs [12, 24]: die Basis ist dual zulässig, primal unzulässig (Summe der negativen rechten Seiten 1). **Ein** dualer Pivot senkt den Zielwert von 45 auf das neue Optimum 42; der Neustart braucht 2 Pivots |
| **Rechte Seite, Zufall 10 × 10** | Pivots warm / kalt (Summe über 50 Läufe) bei −50 % / −25 % / −10 % / +10 % / +25 % / +50 % / +100 %: **56 / 29 / 6 / 5 / 21 / 32 / 52** gegen **271 / 281 / 279 / 278 / 284 / 286 / 280**; Anteil der Läufe ohne Pivot 38 / 52 / 90 / 90 / 76 / 66 / 66 %; Anteil "Neustart nicht schlechter" höchstens 8 % |
| **Schnitt und Schranke, Zufall 10 × 10** | Schnitt λ = 0.95 / 0.9 / 0.75 / 0.5: **69 / 73 / 97 / 144** gegen 300 / 292 / 286 / 239; Neustart nicht schlechter in 2 / 2 / 4 / **26 %** der Fälle. Schranke x_j ≤ λ·x_j* (18 Dienste in den Lösungen): 18 / 18 / 24 / 30 gegen 111 / 111 / 106 / 103 |
| **Kosten und neuer Dienst, Zufall 10 × 10** | Kosten −50 % … +100 %: 33 / 16 / 8 / 9 / 18 / 41 / 67 gegen 262 / 265 / 274 / 273 / 269 / 264 / 250 (bei +100 % in 14 % der Fälle nicht besser). Neuer Dienst f = 0.8 / 1.05 / 1.25 / 2: **0 / 57 / 83 / 151** gegen 289 / 288 / 257 / 214; f = 0.8 lohnt nie (0 Pivots, die Basis bleibt optimal), bei f = 2 ist der Neustart in **40 %** der Fälle nicht schlechter |
| **Zentrum (4 Ressourcen, 5 Dienste)** | Rechte Seite −10 %: 1 gegen 16 Pivots (4 Ressourcen); +50 % und +100 %: 5 gegen 14; neuer Dienst f = 2: **13 gegen 10** (60 % der Fälle nicht besser) |
| **Entartete Ecke** | Schnitt λ = 0.75 und 0.5: **6 gegen 6 und 7 gegen 7 Pivots**, der Neustart ist in 50 und 75 % der Fälle nicht schlechter; auf entarteten Instanzen spart der Warmstart bei großen Änderungen nichts |
| **Ketten (50 Änderungen)** | Rechte Seite einer Ressource wandert innerhalb von ±30 %; warm vom letzten Endtableau, kalt jedesmal von Null. Zentrum **20 gegen 184** (36 Schritte ohne Pivot), Zufall 10 × 10 **6 gegen 240** (45 ohne Pivot), Zufall 12 × 12 24 gegen 309 (37), Mischung 8 × 8 18 gegen 346 (40), entartet 15 gegen 84 (36), Lehrbuch 4 gegen 98 (46); höchstens 4 Pivots in einem Schritt |
| **Branching-Folge** | Nacheinander x_j ≤ 0.8·x_j* für den Dienst mit der größten Menge (8 Schritte): Zentrum **9 gegen 41 Pivots**, der Optimalwert fällt von 720 auf 654.90; Zufall 10 × 10 11 gegen 38; Mischung 8 × 8 10 gegen 73 |
| **Wertkurve (41 Punkte)** | Zentrum Kommissionierstunden **3 warm gegen 134 kalt, 3 Basiswechsel**; Lagerfläche 4 gegen 155 (4 Basiswechsel), Rampenzeit 3 gegen 130 (3). Zufall 10 × 10: 5 gegen 207 bei 4 Basiswechseln, 4 gegen 154 bei 4. **Ein Pivot je Basiswechsel in fünf von sechs Kurven, einmal ein Pivot mehr** |
| **Zeilenwahl** | Dantzig (kleinste rechte Seite) gegen Bland (kleinster Index) im dualen Simplex, Zufall 10 × 10 rechte Seite: 56 / 29 / 6 / 5 / 21 / 32 / 52 gegen 56 / 31 / 6 / 5 / 24 / 37 / 51; Schnitte 69 / 73 / 97 / 144 gegen 74 / 74 / 100 / 147: kaum ein Unterschied, Bland eher etwas mehr |

## Vorab-Hypothesen

| Hypothese (vor der Messung) | Ergebnis |
|---|---|
| Der Warmstart braucht 10 bis 30 % der Pivots des Neustarts | **Teils bestätigt:** bei kleinen Änderungen der rechten Seite deutlich weniger (2 % bei ±10 %), bei mittleren im erwarteten Bereich (7 bis 34 %), bei großen Schnitten und neuen Diensten 60 bis 71 % auf Zufallsinstanzen und im Zentrum 130 % (neuer Dienst f = 2) |
| Bei großen Änderungen holt der Neustart auf | **Bestätigt:** Schnitt λ = 0.5 26 %, neuer Dienst f = 2 40 % (Zentrum 60 %), entartete Ecke bis 75 % der Fälle mit Warmstart ≥ Neustart |
| Innerhalb des Ranging-Bereichs braucht der Warmstart 0 Pivots, außerhalb wenige | **Bestätigt:** innerhalb genau 0 (Test über Instanzen), außerhalb mindestens 1; "wenige" heißt in den Ketten höchstens 4 Pivots je Schritt |
| Die parametrische Folge braucht etwa einen Pivot je Knickpunkt | **Bestätigt bis auf einen Fall:** 5 von 6 Kurven genau ein Pivot je Basiswechsel, eine Kurve 5 Pivots bei 4 Basiswechseln |
| Kostenänderungen sind billiger als Änderungen der rechten Seite | **Nicht bestätigt:** bei kleinen Änderungen ähnlich (8 / 9 gegen 6 / 5), bei −50 % billiger (33 gegen 56), bei +100 % teurer (67 gegen 52) |
| Ketten profitieren stärker als Einzeländerungen | **Bestätigt:** 6 gegen 240 Pivots; in 45 von 50 Schritten kein Pivot |
| Der Warmstart ist nie schlechter als der Neustart | **Widerlegt** (siehe oben): bei großen Änderungen wird der Warmstart in einem Teil der Fälle übertroffen; die Ursache ist nicht isoliert (denkbar: die alte Basis liegt nach einer großen Änderung weit vom neuen Optimum, der Neustart nicht) |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Ein dualer Pivot** (das Tableau nach der Änderung mit rot markierten negativen rechten Seiten und gelbem Pivotelement, Slider über die Pivots, der duale Quotiententest r_j / |a_ij| als Balken, Verlauf von Zielwert und Unzulässigkeit) → **Warm gegen kalt** (Balken, Kurve über die Größe der Änderung, Tabelle über alle fünf Änderungsarten, beides auf Abruf) → **Ketten** (50 Änderungen einer rechten Seite, kumulierte Pivots, Histogramm der Pivots je Schritt, Branching-Folge) → **Wertkurve** (Kurve z*(b) über 41 Punkte, Pivots je Punkt warm gegen kalt).
2. **Fünf Änderungsarten:** rechte Seite (dual), Schnitt (dual), Schranke x_j ≤ λ·x_j* (dual), Deckungsbeitrag (primal), neuer Dienst (primal), je mit einer Größenstufe.
3. **Zeilenwahl** im dualen Simplex: Dantzig oder Bland (mit Notbremse nach 25 Nullschritten).
4. **Instanzen:** Lehrbuchbeispiel, Zentrum, entartete Ecke, Zufall, Mischung; Unzulässig und Unbeschränkt (ohne Endbasis, mit Erklärung). Kann der Warmstart nicht rechnen (künstliche Variable in der Basis), fällt die Demo auf den Neustart zurück und meldet es.

Presets (10): Lehrbuch: ein dualer Pivot, Zentrum: Kommissionierstunden +50 %, Innerhalb des Bereichs: null Pivots, Schnitt schneidet den Optimalpunkt ab, Schranke (Branching), Kosten ändern sich: primaler Warmstart, Neuer Dienst, Neustart nicht schlechter, Kette von 50 Änderungen, Wertkurve: ein Pivot je Knickpunkt.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Lehrbuch: ein dualer Pivot** | Lagerfläche +50 % | 1 dualer Pivot (Zielwert 45 → 42) gegen 2 beim Neustart |
| **Zentrum: Kommissionierstunden +50 %** | 80 → 120, Bereich [70, 90] | 2 duale Pivots (920, 893.75, 880.83) gegen 3; neues Optimum 880.83 (vorher 720) |
| **Innerhalb des Bereichs: null Pivots** | Lagerfläche −10 % (200 → 180) | Bereich [172, 217.5]: 0 Pivots, Optimalwert 720 − 0.5 · 20 = 710; Neustart 4 Pivots |
| **Schnitt schneidet den Optimalpunkt ab** | a·x ≤ 0.75·a·x* (rechte Seite 20.54) | 1 dualer Pivot (720 → 716.10) gegen 4 |
| **Schranke (Branching)** | Express-Pakete ≤ 0.75 · 20 = 15 | 1 dualer Pivot (720 → 716.5) gegen 4 |
| **Kosten ändern sich: primaler Warmstart** | Sperrgut +50 % (20 → 30, Grenze 23.5) | 2 primale Pivots (720, 770.56, 770.77) gegen 5 |
| **Neuer Dienst** | Vorbild Express-Pakete (0.9 mal), Deckungsbeitrag 1.25 · y·a = 17.44 | 2 primale Pivots (720, 797.5, 840.38) gegen 5 |
| **Neustart nicht schlechter** | Zufall 8 × 8, Seed 22, Schnitt λ = 0.5 | 4 duale Pivots gegen 1 beim Neustart; bei Zufall 10 × 10 ist der Warmstart bei diesem Schnitt in 26 % der Fälle nicht besser, bei der entarteten Ecke in 75 % |
| **Kette von 50 Änderungen** | Zufall 10 × 10 | 6 Pivots warm gegen 240 kalt; 45 von 50 Schritten ohne Pivot |
| **Wertkurve: ein Pivot je Knickpunkt** | Zentrum, Kommissionierstunden | 3 Pivots warm gegen 134 kalt, 3 Basiswechsel |

## Modell und Verfahren

- **Instanz** (`dsx_scenario.py`): die Auslastungsplanung der Vorgängerstücke (Zentrum, Lehrbuch, entartete Ecke, Zufall, Mischung, Unzulässig, Unbeschränkt); das Transportproblem entfällt, weil seine redundanten Gleichungen künstliche Variablen in der Basis lassen und den Warmstart unmöglich machen.
- **Neustart** (`dsx_algorithm.py`): der Zwei-Phasen-Tableau-Simplex der Stücke 1 bis 5, der das Endtableau liefert; die Pivotzahl des Neustarts zählt beide Phasen.
- **Änderungen am Endtableau** (`dsx_reopt.py`): rechte Seite b_i → b_i + δ verschiebt die rechte Seite um δ · B⁻¹e_i (B⁻¹ steht unter den Spalten der Anfangsbasis); Kostenänderung berechnet nur die Zielzeile neu; ein Schnitt bringt eine Zeile mit neuer Schlupfspalte als Basisvariable, die Zeile wird um die Basiszeilen der Strukturvariablen bereinigt; eine Schranke ist ein Schnitt mit Einheitsvektor (±e_j); ein neuer Dienst bringt die Spalte B⁻¹a. Alle fünf gegen ein frisch berechnetes Tableau mit derselben Basis geprüft.
- **Dualer Simplex:** Zeile mit x_B < 0 (Dantzig: die kleinste; Bland: kleinster Basisindex, Notbremse nach 25 Nullschritten), Spalte durch min r_j / |a_ij| über a_ij < 0, Pivot; keine Spalte mit negativem Eintrag beweist die Unzulässigkeit. **Primaler Simplex** von der alten Basis wie in den Vorgängerstücken. `reoptimize` wählt nach dem Zulässigkeitszustand: dual, primal oder "keine Pivots nötig"; bei einer künstlichen Variable in der Basis Rückfall auf den Neustart (Hinweis, nie stumm).
- **Auswertung** (`dsx_evaluation.py`): Einzelvergleich warm gegen kalt, Größenkurven (Summen über alle Ressourcen bzw. Dienste, bei Zufall über fünf feste Instanzen), Ketten, Branching-Folge, parametrische Wertkurve.

## Was nicht funktioniert hat / Grenzen

- **Der Neustart ist der Maßstab, aber kein starker.** Er ist der Zwei-Phasen-Simplex desselben Lösers; echte Löser starten von einer Crash-Basis oder rechnen den dualen Simplex von Null. Die Lücke zum Warmstart wäre dort kleiner.
- **Große Änderungen und entartete Instanzen:** der Warmstart kann den Neustart nicht schlagen (26 % bei Schnitt λ = 0.5, bis 75 % bei der entarteten Ecke; der Anteil "warm ≥ kalt" steht in jeder Tabelle).
- **Nur Dantzig und Bland** für die Zeilenwahl; duales Steepest Edge (Forrest und Goldfarb 1992, hier nur genannt) und Bound Flipping sind nicht gebaut.
- **Schranken als Zeilen.** x_j ≤ k ist hier eine neue Nebenbedingung; echte Löser behandeln Variablenschranken direkt.
- **Kein Branch-and-Bound-Baum:** nur einzelne Schritte und eine Folge (Tauchgang). Auch die Zeilenerzeugung bei Schnittebenen (Gomory) ist nicht gebaut, nur der einzelne Schnitt.
- **Kein Transportproblem** (künstliche Variablen in der Basis); der Rückfall auf den Neustart ist nur getestet, nicht als Preset gezeigt.
- **Pivots als Maß, nicht Laufzeit.** Im dichten Tableau kostet ein Pivot überall gleich viel; Stück 4 (Revised Simplex) zeigt, wo das nicht stimmt.
- **Synthetische, kleine Instanzen** (bis 12 Ressourcen und 12 Dienste). Änderungen sind einzeln und zufällig, nicht aus einem echten Planungsverlauf.

## Verifikation

- `tests/test_algorithm.py`: **Warm == Neustart == HiGHS** auf 1962 Änderungen aller fünf Arten (Status, Optimalwert, primal und dual zulässig, starke Dualität); Tableau nach jeder Änderung gegen die frisch berechnete Basislösung, Schlupf und Zielzeile; Invarianten des dualen Simplex (dual zulässig in jedem Pivot, Zielwert nicht steigend, Ablehnung einer nicht dual zulässigen Basis); Bland-Notbremse; **Bezug zu Stück 5** (innerhalb des Bereichs 0 Pivots, außerhalb mindestens 1); Unzulässig (Zeile ohne negativen Eintrag) und Unbeschränkt; Branching-Kinder gegen HiGHS (Zielwert höchstens Elternwert); Rückfall auf den Neustart; Buchführung und Determinismus.
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz und Änderungsart, jede Größenstufe, Pivot-Regler, Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy, pandas, plotly und streamlit.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Lemke, C. E. (1954). *The dual method of solving the linear programming problem.* Naval Research Logistics Quarterly 1(1), 36–47.
- Forrest, J. J., & Goldfarb, D. (1992). *Steepest-edge simplex algorithms for linear programming.* Mathematical Programming 57, 341–374 (nur genannt).
- Dantzig, G. B. (1963). *Linear Programming and Extensions.* Princeton University Press.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
