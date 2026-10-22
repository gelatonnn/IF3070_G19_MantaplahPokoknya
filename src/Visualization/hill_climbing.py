from pathlib import Path
import os
import textwrap

from ..Core.evaluation import evaluate
from ..Core.state import footprint
from ..Core.fleet import ships_by_id, placement_ship_id


def _pyplot():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _draw_deck(ax, state, vehicles, ship, sid, deck, metrics, palette):
    from matplotlib.patches import Rectangle
    for index, (vid, placement) in enumerate(state.items()):
        if placement_ship_id(placement, ship) != sid:
            continue
        x, y = placement["position"]
        dx, dy = footprint(vehicles[vid], placement["orientation"])
        ax.add_patch(Rectangle(
            (x, y), dx, dy, facecolor=palette(index % 20),
            edgecolor="#243247", linewidth=1.1, alpha=0.55,
        ))
        ax.text(x + dx / 2, y + dy / 2,
                f"{vid}\n{placement['orientation']}", ha="center", va="center",
                fontsize=7, color="#142238")
    for y, row in enumerate(metrics["grid"]):
        for x, count in enumerate(row):
            if count > 1:
                ax.add_patch(Rectangle((x, y), 1, 1, fill=False,
                                       hatch="///", edgecolor="#c62828", linewidth=0.5))
    ax.set(xlim=(0, deck["l"]), ylim=(deck["w"], 0), aspect="equal",
           xlabel="x / kolom", ylabel="y / baris")
    ax.set_xticks(range(deck["l"] + 1))
    ax.set_yticks(range(deck["w"] + 1))
    ax.grid(alpha=0.25)
    ax.set_title(f"{sid} | Berat {metrics['total_weight']}/{deck['max_capacity']}\n"
                 f"Fee {metrics['revenue']} | Value {metrics['objective_value']} | "
                 f"Overlap {metrics['overlap']} | Excess {metrics['excess_weight']}",
                 fontsize=10)


def plot_state(state, vehicles, ship, path, title, objective="shipping_fee"):
    plt = _pyplot()
    metrics = evaluate(state, vehicles, ship, objective)
    fleet = ships_by_id(ship)
    columns = min(2, len(fleet))
    rows = (len(fleet) + columns - 1) // columns
    height = 6 * rows + 3
    fig, axes = plt.subplots(rows, columns, figsize=(12, height), squeeze=False)
    fig.subplots_adjust(bottom=2.6 / height, top=1 - 1.3 / height,
                        hspace=0.55, wspace=0.3)
    palette = plt.get_cmap("tab20")
    for ax, (sid, deck) in zip(axes.flat, fleet.items()):
        _draw_deck(ax, state, vehicles, ship, sid, deck, metrics["per_ship"][sid], palette)
    for ax in list(axes.flat)[len(fleet):]:
        ax.set_visible(False)
    valid = "VALID" if metrics["is_valid"] else "BELUM VALID"
    fig.suptitle(
        f"{title} | {valid} | {len(fleet)} kapal | {objective}\n"
        f"Score {metrics['score']} | Value {metrics['objective_value']} | Fee {metrics['revenue']} | "
        f"Overlap {metrics['overlap']} | Excess {metrics['excess_weight']}",
        fontsize=13, y=0.97,
    )
    outside = [
        f"{vid}({p['orientation']})"
        for vid, p in state.items() if p["position"] is None
    ]
    lines = textwrap.wrap(", ".join(outside) or "(tidak ada)", width=110)
    fig.text(0.06, 2.1 / height,
             f"Di dalam: {metrics['loaded_count']} | Di luar: {len(outside)}\n"
             "Kendaraan di luar (orientasi tetap disimpan):\n" + "\n".join(lines)
             + "\nArsir merah = sel overlap. Anchor = pojok kiri-atas.",
             fontsize=9, va="top", linespacing=1.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(result, path):
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    runs = result.get("runs", [result])
    for index, run in enumerate(runs):
        history = run["history"]
        xs = [point["iteration"] for point in history]
        label = "Awal" if index == 0 else f"Restart {index}"
        axes[0].plot(xs, [p["score"] for p in history], label=label, marker=".", markersize=3)
        axes[1].plot(xs, [p["revenue"] for p in history], label=label, marker=".", markersize=3)
    objective = result["parameters"].get("objective", "shipping_fee")
    axes[0].set_ylabel(f"Score ({objective} - penalti)")
    axes[1].set_ylabel("Total shipping fee")
    axes[1].set_xlabel("Langkah diterima (0 = initial state; dihitung per pencarian)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    final = result["final_metrics"]
    fig.suptitle(f"{result['algorithm']} | {objective} | Final score {final['score']} | "
                 f"{result['duration_seconds']:.3f} detik")
    fig.text(0.07, 0.025,
             f"Stop: {result['stop_reason']} | Final source: {result['final_source']}\n"
             "Grafik merekam lintasan aktual; baseline valid tidak disisipkan sebagai langkah pencarian.",
             fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_plots(result, vehicles, ship, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = result["parameters"].get("objective", "shipping_fee")
    plot_state(result["initial_state"], vehicles, ship, output_dir / "initial.png",
               f"{result['algorithm']} - Initial", objective)
    plot_state(result["final_state"], vehicles, ship, output_dir / "final.png",
               f"{result['algorithm']} - Final", objective)
    plot_history(result, output_dir / "history.png")
    # Semua initial/final restart juga dapat diperiksa secara visual.
    for run in result.get("runs", []):
        folder = output_dir / f"restart_{run['restart_index']:02d}"
        save_plots(run, vehicles, ship, folder)
