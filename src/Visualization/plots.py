"""Visualisasi Genetic Algorithm untuk kapal tunggal dan armada."""

from pathlib import Path
import textwrap

from ..Core.evaluation import evaluate
from ..Core.fleet import placement_ship_id, ships_by_id
from ..Core.state import footprint


PALETTE = (
    "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6", "#ffffcc",
    "#e5d8bd", "#fddaec", "#f2f2f2", "#fbb4ae", "#cbd5e8",
    "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02",
    "#a6761d", "#8dd3c7", "#ffffb3", "#bebada", "#fb8072",
)


def vehicle_colors(vehicles):
    """Berikan warna yang tetap untuk setiap ID pada gambar awal dan akhir."""
    return {vid: PALETTE[index % len(PALETTE)]
            for index, vid in enumerate(sorted(vehicles))}


def _draw_deck(ax, state, vehicles, ship, ship_id, deck, metrics, colors):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((0, 0), deck["l"], deck["w"],
                           facecolor="white", edgecolor="black", linewidth=1))
    for vid, placement in sorted(state.items()):
        if placement_ship_id(placement, ship) != ship_id:
            continue
        x, y = placement["position"]
        dx, dy = footprint(vehicles[vid], placement["orientation"])
        ax.add_patch(Rectangle((x, y), dx, dy, facecolor=colors[vid],
                               edgecolor="#20252b", linewidth=0.7, alpha=0.9))
        ax.text(x + dx / 2, y + dy / 2, f"{vid}\n{placement['orientation']}",
                ha="center", va="center", fontsize=6.5 if deck["l"] >= 15 else 8,
                color="#111111")
    for y, row in enumerate(metrics["grid"]):
        for x, count in enumerate(row):
            if count > 1:
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor="none",
                                       edgecolor="red", hatch="///", linewidth=0))
    ax.set_xlim(0, deck["l"])
    ax.set_ylim(deck["w"], 0)
    ax.set_aspect("equal")
    x_step = max(1, (deck["l"] + 9) // 10)
    y_step = max(1, (deck["w"] + 9) // 10)
    ax.set_xticks(sorted({*range(0, deck["l"] + 1, x_step), deck["l"]}))
    ax.set_yticks(sorted({*range(0, deck["w"] + 1, y_step), deck["w"]}))
    ax.set_xlabel("x / kolom", fontsize=9)
    ax.set_ylabel("y / baris", fontsize=9)
    ax.grid(True, color="#e2e2e2", linewidth=0.5, alpha=0.9)
    ax.tick_params(labelsize=8)
    ax.set_title(
        f"{ship_id} | Berat {metrics['total_weight']}/{deck['max_capacity']} | "
        f"Fee {metrics['revenue']} | Value {metrics['objective_value']:.2f}",
        fontsize=9.5,
    )


def draw_state(state, vehicles, ship, colors=None, *, stage="final",
               objective="shipping_fee", title=None):
    """Buat gambar state GA tanpa mengubah posisi atau ukuran kendaraan."""
    from matplotlib.figure import Figure

    metrics = evaluate(state, vehicles, ship, objective)
    fleet = ships_by_id(ship)
    columns = min(2, len(fleet))
    rows = (len(fleet) + columns - 1) // columns
    if columns == 1:
        width, height = 11, 8
        left, right, bottom_margin, top_margin, footer_top = 0.1, 0.93, 2, 0.7, 1.55
    else:
        width, height = 12, 4.4 * rows + 1.8
        left, right, bottom_margin, top_margin, footer_top = 0.08, 0.96, 1.6, 0.65, 1.42
    fig = Figure(figsize=(width, height), dpi=130)
    axes = fig.subplots(rows, columns, squeeze=False)
    fig.subplots_adjust(left=left, right=right, bottom=bottom_margin / height,
                        top=1 - top_margin / height, hspace=0.55, wspace=0.3)
    colors = colors if colors is not None else vehicle_colors(vehicles)
    for ax, (ship_id, deck) in zip(axes.flat, fleet.items()):
        _draw_deck(ax, state, vehicles, ship, ship_id, deck,
                   metrics["per_ship"][ship_id], colors)
    for ax in list(axes.flat)[len(fleet):]:
        ax.set_visible(False)

    if title is None:
        status = "VALID" if metrics["is_valid"] else "BELUM VALID"
        title = (f"Genetic Algorithm · {stage} | {status} | {objective}\n"
                 f"Score {metrics['score']:.2f} | Fee {metrics['revenue']} | "
                 f"Berat {metrics['total_weight']} | Overlap {metrics['overlap']} | "
                 f"Excess {metrics['excess_weight']}")
    fig.suptitle(title, fontsize=11, y=0.975)

    unloaded = [f"{vid}({placement['orientation']})"
                for vid, placement in sorted(state.items())
                if placement["position"] is None]
    outside = textwrap.wrap(", ".join(unloaded) or "(tidak ada)", width=110)
    fig.text(0.05, footer_top / height,
             f"Di dalam: {metrics['loaded_count']} | Di luar: {len(unloaded)}\n"
             "Kendaraan di luar (orientasi tetap disimpan):\n"
             + "\n".join(outside)
             + "\nArsir merah = sel overlap. Anchor = pojok kiri-atas.",
             fontsize=7.5, ha="left", va="top", linespacing=1.3)
    return fig


def history_figure(history, keys=None, ylabel="Nilai fitness", title=None, *,
                   final_score=None, duration=None, params=None, objective="shipping_fee"):
    """Grafik fitness populasi dan objective valid terbaik per generasi."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.5, 4.5), dpi=140)
    ax = fig.subplots()
    generations = [row["generation"] for row in history]
    if keys is None:
        keys = [
            ("candidate_peak", "Fitness maksimum populasi", "#1f77b4"),
            ("mean_fitness", "Fitness rata-rata populasi", "#2ca02c"),
            ("best_valid_objective_so_far", "Objective valid terbaik", "#d95f02"),
        ]
    for item in keys:
        key, label = item[:2]
        color = item[2] if len(item) > 2 else None
        ax.plot(generations, [row[key] for row in history], label=label,
                color=color, linewidth=1.5)
    if final_score is not None:
        ax.axhline(final_score, color="#a62727", linestyle="--", linewidth=1.1,
                   label=f"Final valid = {final_score:.2f}")
    if title is None and params is not None and duration is not None and final_score is not None:
        title = (f"Genetic Algorithm | {objective} | Final score {final_score:.2f} | "
                 f"{duration:.2f} detik\n"
                 f"Populasi {params['population_size']} · Generasi {params['generations']} · "
                 f"Crossover {params['crossover_rate']} · Mutasi {params['mutation_rate']} · "
                 f"Seed {params['seed']}")
    ax.set(xlabel="Generasi", ylabel=ylabel, title=title)
    ax.grid(True, color="#ebebeb", linewidth=0.7)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def _save_figures(result, vehicles, ship, initial_path, final_path, history_path):
    objective = result["parameters"]["objective"]
    colors = vehicle_colors(vehicles)
    for stage, path in (("initial", initial_path), ("final", final_path)):
        fig = draw_state(result[f"{stage}_state"], vehicles, ship, colors,
                         stage=stage, objective=objective)
        fig.savefig(path, dpi=130)
        fig.clear()
    fig = history_figure(result["history"],
                         ylabel=f"Nilai ({objective})",
                         final_score=result["final_metrics"]["score"],
                         duration=result["duration_seconds"],
                         params=result["parameters"], objective=objective)
    fig.savefig(history_path, dpi=130)
    fig.clear()


def save_result_plots(result, out_dir, stem):
    """Ekspor tiga gambar GA dengan nama yang dapat dipakai caller lama."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"initial_png": out / f"{stem}_initial.png",
             "final_png": out / f"{stem}_final.png",
             "fitness_png": out / f"{stem}_fitness.png"}
    _save_figures(result, result["vehicles"], result["ship"],
                  paths["initial_png"], paths["final_png"], paths["fitness_png"])
    return paths


def save_plots(result, vehicles, ship, output_dir):
    """Pertahankan nama gambar yang dipakai CLI dan Streamlit saat ini."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_figures(result, vehicles, ship, output_dir / "initial.png",
                  output_dir / "final.png", output_dir / "history.png")
