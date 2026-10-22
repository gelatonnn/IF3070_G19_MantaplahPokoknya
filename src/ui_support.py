"""Data dan grafik interaktif untuk antarmuka Streamlit."""

from io import BytesIO
import json
from pathlib import Path
import re
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

from .Core.evaluation import evaluate
from .Core.fleet import placement_ship_id, ships_by_id
from .Core.state import footprint


NAVY = "#233867"
CREAM = "#E3CE96"
SAND = "#C69960"
SEA = "#3D607B"
PAPER = "#F8F5ED"
COLORS = (NAVY, SEA, SAND, CREAM)


def parse_seeds(raw: str) -> list[int]:
    parts = [part for part in re.split(r"[,\s]+", raw.strip()) if part]
    if not parts:
        raise ValueError("Isi sedikitnya satu seed.")
    try:
        seeds = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Seed harus berupa bilangan bulat, dipisahkan koma atau spasi.") from exc
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed tidak boleh berulang.")
    return seeds


def read_result(run_dir: Path, algorithm: str, seed: int) -> dict:
    path = run_dir / algorithm / f"seed_{seed}" / "result.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_bundle(run_dir: Path) -> bytes:
    """Bundel hanya berkas hasil pada direktori eksekusi yang dipilih."""
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    return buffer.getvalue()


def comparison_summary(rows: list[dict]) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["algorithm"], []).append(row)
    summary = []
    for name, items in grouped.items():
        valid_values = [item["objective_value"] for item in items if item["is_valid"]]
        summary.append({
            "Algoritma": name, "Run": len(items), "Valid": len(valid_values),
            "Rata-rata objective valid": round(mean(valid_values), 2) if valid_values else None,
            "Terbaik": round(max(valid_values), 2) if valid_values else None,
            "Rata-rata durasi (detik)": round(mean(item["duration_seconds"] for item in items), 3),
        })
    return summary


def _plotly_layout(fig, title: str, *, height: int = 450):
    fig.update_layout(title=title, height=height, paper_bgcolor=PAPER,
                      plot_bgcolor="#FFFFFF", font_color=NAVY,
                      margin=dict(l=45, r=30, t=65, b=55),
                      hoverlabel=dict(bgcolor=NAVY, font_color="#FFFFFF"))
    return fig


def deck_figure(state: dict, vehicles: dict, ship: dict, ship_id: str, stage: str):
    """Denah geladak Plotly; hover memperlihatkan atribut kendaraan."""
    import plotly.graph_objects as go

    deck = ships_by_id(ship)[ship_id]
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=deck["l"], y1=deck["w"],
                  line=dict(color=NAVY, width=2), fillcolor="#FFFFFF", layer="below")
    for index, (vid, placement) in enumerate(sorted(state.items())):
        if placement_ship_id(placement, ship) != ship_id:
            continue
        x, y = placement["position"]
        dx, dy = footprint(vehicles[vid], placement["orientation"])
        vehicle = vehicles[vid]
        fig.add_shape(type="rect", x0=x, y0=y, x1=x + dx, y1=y + dy,
                      line=dict(color=NAVY, width=1.5),
                      fillcolor=COLORS[index % len(COLORS)], opacity=0.82)
        fig.add_trace(go.Scatter(
            x=[x + dx / 2], y=[y + dy / 2], mode="markers+text",
            marker=dict(size=18, color="rgba(0,0,0,0)"),
            text=[f"{vid}<br>{placement['orientation']}"],
            textfont=dict(size=11, color="#FFFFFF" if index % 4 in (0, 1) else NAVY),
            hovertemplate=(f"<b>{vid}</b><br>Posisi: ({x}, {y})<br>"
                           f"Orientasi: {placement['orientation']}<br>"
                           f"Ukuran: {dx} × {dy}<br>Berat: {vehicle['weight']}<br>"
                           f"Shipping fee: {vehicle['shipping_fee']}<br>ETA: {vehicle['eta']}<extra></extra>"),
            showlegend=False,
        ))
    for y, row in enumerate(evaluate(state, vehicles, ship)["per_ship"][ship_id]["grid"]):
        for x, count in enumerate(row):
            if count > 1:
                fig.add_shape(type="rect", x0=x, y0=y, x1=x + 1, y1=y + 1,
                              line=dict(color="#B33636", width=2.5), fillcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, deck["l"]], dtick=2 if deck["l"] > 12 else 1, title="Kolom (x)",
                     showgrid=True, gridcolor="#DCD6C9", zeroline=False)
    fig.update_yaxes(range=[deck["w"], 0], dtick=1, title="Baris (y)",
                     showgrid=True, gridcolor="#DCD6C9", zeroline=False,
                     scaleanchor="x", scaleratio=1)
    return _plotly_layout(fig, f"{ship_id} · {stage}", height=500)


def history_figure(result: dict):
    """Pisahkan fitness berpenalti dan nilai objective agar keduanya terbaca."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if result["algorithm"] == "random_restart":
        for index, run in enumerate(result["runs"]):
            xs = [row["iteration"] for row in run["history"]]
            color = COLORS[index % len(COLORS)]
            for key, label, secondary, dash in (
                ("score", "score", False, "solid"),
                ("objective_value", "objective", True, "dot"),
            ):
                fig.add_trace(go.Scatter(x=xs, y=[row[key] for row in run["history"]],
                                         mode="lines+markers", name=f"Run {index} · {label}",
                                         line=dict(color=color, width=2, dash=dash),
                                         marker=dict(size=4)), secondary_y=secondary)
        x_label = "Iterasi per restart"
    elif result["algorithm"] == "genetic_algorithm":
        history = result["history"]
        xs = [row["generation"] for row in history]
        series = (("candidate_peak", "Fitness maksimum", NAVY, False),
                  ("mean_fitness", "Fitness rata-rata", SEA, False),
                  ("best_valid_objective_so_far", "Objective valid terbaik", SAND, True))
        x_label = "Generasi"
    else:
        history = result["history"]
        xs = [row["iteration"] for row in history]
        series = (("score", "Score (dengan penalti)", NAVY, False),
                  ("objective_value", "Nilai objective state", SAND, True))
        x_label = "Iterasi"
    if result["algorithm"] != "random_restart":
        for key, label, color, secondary in series:
            fig.add_trace(go.Scatter(x=xs, y=[row[key] for row in history],
                                     mode="lines+markers", name=label,
                                     line=dict(color=color, width=2), marker=dict(size=4)),
                          secondary_y=secondary)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text="Fitness / score", secondary_y=False)
    fig.update_yaxes(title_text="Nilai objective", secondary_y=True)
    fig.update_layout(legend=dict(orientation="h", y=-0.22), hovermode="x unified")
    _plotly_layout(fig, "Perkembangan optimasi", height=500)
    fig.update_layout(margin=dict(l=55, r=55, t=65, b=120))
    return fig


def comparison_figure(rows: list[dict], labels: dict[str, str]):
    import plotly.graph_objects as go

    fig = go.Figure()
    for index, seed in enumerate(sorted({row["seed"] for row in rows})):
        items = [row for row in rows if row["seed"] == seed]
        fig.add_bar(x=[labels[row["algorithm"]] for row in items],
                    y=[row["objective_value"] if row["is_valid"] else 0 for row in items],
                    name=f"Seed {seed}", marker_color=COLORS[index % len(COLORS)],
                    customdata=[[row["duration_seconds"], row["is_valid"]] for row in items],
                    hovertemplate="%{x}<br>Objective: %{y:.2f}<br>Durasi: %{customdata[0]:.2f} detik<br>Valid: %{customdata[1]}<extra></extra>")
    fig.update_layout(barmode="group", legend_title="Seed")
    fig.update_yaxes(title="Nilai objective solusi valid", rangemode="tozero")
    fig.update_xaxes(tickangle=-30)
    _plotly_layout(fig, "Perbandingan hasil per algoritma dan seed", height=540)
    fig.update_layout(margin=dict(l=55, r=30, t=65, b=120))
    return fig
