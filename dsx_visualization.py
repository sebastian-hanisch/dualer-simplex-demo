"""Plotly-Abbildungen: Verlauf des dualen Simplex, duale Quotienten, Pivots warm gegen kalt, Größe der Änderung, Ketten, parametrische Wertkurve. Achsen sind gesperrt (fixedrange), damit Touch-Geräte
beim Scrollen nicht zoomen."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEAL, ORANGE, RED, BLUE, GREY, PURPLE = "#2F6B65", "#e8a13a", "#d62728", "#1f4e9c", "#8a8f98", "#7b3fbf"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height, legend_y=-0.25):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def tableau_frame(T, basis, names, allowed, m, digits=2):
    """Tableau als Tabelle: eine Zeile je Basisvariable, dazu die Zielzeile (reduzierte Kosten); Spalten: alle nicht künstlichen Variablen und die rechte Seite."""
    cols = [j for j in allowed]
    data = []
    labels = []
    for i in range(m):
        labels.append(names[basis[i]])
        data.append([T[i, j] for j in cols] + [T[i, -1]])
    labels.append("Zielzeile r")
    data.append([T[m, j] for j in cols] + [T[m, -1]])
    df = pd.DataFrame(data, index=labels, columns=[names[j] for j in cols] + ["rechte Seite"]).round(digits)
    return df.map(lambda v: 0.0 if abs(v) < 5e-13 else v)


def build_progress(path):
    """Verlauf über die dualen (bzw. primalen) Pivots: Zielwert (fällt beim dualen Simplex, steigt beim primalen) und Summe der Unzulässigkeit der rechten Seiten (Balken)."""
    xs = list(range(len(path)))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[p["infeasibility"] for p in path], marker_color="rgba(214,39,40,0.35)", name="Summe der negativen rechten Seiten", yaxis="y2"))
    fig.add_trace(go.Scatter(x=xs, y=[p["obj"] for p in path], mode="lines+markers", line=dict(color=TEAL, width=3), marker=dict(size=8), name="Zielwert der Basis"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Unzulässigkeit", rangemode="tozero", showgrid=False, fixedrange=True), yaxis_title_text="Zielwert")
    fig.update_xaxes(title_text="Pivot (0 = Endbasis nach der Änderung)", dtick=1)
    return _base(fig, 300, legend_y=-0.4)


def build_ratios(ratios, chosen, names):
    """Dualer Quotiententest der gewählten Zeile: r_j / |a_ij| je Kandidatenspalte; der kleinste tritt ein."""
    xs = [names[j] for j in ratios]
    fig = go.Figure(go.Bar(x=xs, y=list(ratios.values()), marker_color=[ORANGE if j == chosen else GREY for j in ratios], text=[f"{v:.2f}" for v in ratios.values()], textposition="outside"))
    fig.update_yaxes(title_text="r_j / |a_ij|", rangemode="tozero")
    return _base(fig, 260)


def build_warm_cold(warm, cold, warm_label):
    fig = go.Figure(go.Bar(x=[warm_label, "Neustart von Null"], y=[warm, cold], marker_color=[TEAL, GREY], text=[str(warm), str(cold)], textposition="outside"))
    fig.update_yaxes(title_text="Pivots", rangemode="tozero")
    return _base(fig, 280)


def build_sweep(rows, change_label):
    """Pivots über die Größe der Änderung: Summe der Warmstarts (Balken), Summe der Neustarts (Balken) und das Verhältnis (Linie, rechte Achse)."""
    xs = [r["label"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=[r["warm"] for r in rows], name="Warmstart", marker_color=TEAL))
    fig.add_trace(go.Bar(x=xs, y=[r["cold"] for r in rows], name="Neustart", marker_color=GREY))
    fig.add_trace(go.Scatter(x=xs, y=[r["ratio"] for r in rows], name="Warm / kalt", mode="lines+markers", yaxis="y2", line=dict(color=ORANGE, width=2.5)))
    fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", title="Verhältnis", rangemode="tozero", showgrid=False, fixedrange=True), yaxis_title_text="Pivots (Summe)")
    fig.update_xaxes(title_text=change_label, type="category")
    return _base(fig, 340, legend_y=-0.4)


def build_chain(chain, title_x="Änderung"):
    """Kumulierte Pivots über die Kette: warm (vom letzten Endtableau) gegen kalt (jeder Schritt von Null)."""
    xs = list(range(1, len(chain["warm"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=chain["cum_cold"], mode="lines", line=dict(color=GREY, width=3), name="Neustart von Null"))
    fig.add_trace(go.Scatter(x=xs, y=chain["cum_warm"], mode="lines", line=dict(color=TEAL, width=3), name="Warmstart"))
    fig.update_xaxes(title_text=title_x)
    fig.update_yaxes(title_text="Pivots (kumuliert)", rangemode="tozero")
    return _base(fig, 320, legend_y=-0.35)


def build_chain_hist(chain):
    """Verteilung der Pivots je Schritt: warm (meist 0 oder 1) gegen kalt."""
    top = max(max(chain["cold"]), max(chain["warm"])) if chain["warm"] else 1
    bins = list(range(0, top + 2))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=chain["cold"], xbins=dict(start=-0.5, end=top + 1.5, size=1), marker_color=GREY, name="Neustart", opacity=0.75))
    fig.add_trace(go.Histogram(x=chain["warm"], xbins=dict(start=-0.5, end=top + 1.5, size=1), marker_color=TEAL, name="Warmstart", opacity=0.85))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Pivots je Schritt", tickmode="array", tickvals=bins)
    fig.update_yaxes(title_text="Schritte")
    return _base(fig, 280, legend_y=-0.4)


def build_parametric(p, resource_label, b0):
    """Wertkurve z*(b_i) (warm fortgeführt) und die Pivots je Punkt: kalt (jeder Punkt von Null) gegen warm."""
    grid = np.array(p["grid"])
    z = [v if v is not None else np.nan for v in p["z"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=grid, y=p["cold"], name="Pivots kalt", marker_color="rgba(138,143,152,0.55)", yaxis="y2", width=(grid[1] - grid[0]) * 0.45))
    fig.add_trace(go.Bar(x=grid, y=p["warm"], name="Pivots warm", marker_color=TEAL, yaxis="y2", width=(grid[1] - grid[0]) * 0.45))
    fig.add_trace(go.Scatter(x=grid, y=z, mode="lines+markers", line=dict(color=ORANGE, width=3), marker=dict(size=5), name="Optimalwert z*(b)"))
    fig.add_vline(x=b0, line=dict(color=BLUE, dash="dot"))
    fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", title="Pivots je Punkt", rangemode="tozero", showgrid=False, fixedrange=True), yaxis_title_text="Optimalwert")
    fig.update_xaxes(title_text=f"{resource_label}: rechte Seite b")
    return _base(fig, 380, legend_y=-0.4)
