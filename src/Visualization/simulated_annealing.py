"""Visualisasi Simulated Annealing yang memakai renderer state bersama."""

from pathlib import Path

from .hill_climbing import _pyplot, plot_state


def plot_history(result, trace, path):
    plt = _pyplot()
    scores = trace["scores"]
    xs = range(len(scores))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(xs, scores, linewidth=0.8, label="score state saat ini")
    best = result["final_metrics"]["score"]
    axes[0].axhline(best, color="#c62828", linestyle="--", label=f"best valid = {best:g}")
    axes[0].set_yscale("symlog")
    axes[0].set_ylabel("Score (symlog)")
    axes[0].legend()
    axes[1].plot(xs, trace["temperatures"], color="#ef6c00", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Suhu T (log)")
    axes[1].set_xlabel("Iterasi")
    for ax in axes:
        ax.grid(alpha=0.25)
    objective = result["parameters"]["objective"]
    fig.suptitle(f"Simulated Annealing | {objective} | final {best:g} | "
                 f"{result['duration_seconds']:.3f} detik")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_acceptance(result, trace, path):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(11, 5))
    if trace["acceptance"]:
        xs, ys = zip(*trace["acceptance"])
        ax.scatter(xs, ys, s=2, alpha=0.4)
    ax.set(xlabel="Iterasi", ylabel="exp(delta/T)", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.25)
    ax.set_title(f"Langkah memburuk: diterima {result['accepted_worse_moves']}, "
                 f"ditolak {result['rejected_moves']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_plots(result, trace, vehicles, ship, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = result["parameters"]["objective"]
    plot_state(result["initial_state"], vehicles, ship, output_dir / "initial.png",
               "simulated_annealing - Initial", objective)
    plot_state(result["final_state"], vehicles, ship, output_dir / "final.png",
               "simulated_annealing - Final", objective)
    plot_history(result, trace, output_dir / "history.png")
    plot_acceptance(result, trace, output_dir / "acceptance.png")
