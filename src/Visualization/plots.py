"""Visualisasi GA dengan renderer state bersama HC/SA."""

from pathlib import Path

from .hill_climbing import _pyplot, plot_state


def history_figure(history, keys, ylabel, title):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    generations = [row["generation"] for row in history]
    for key, label in keys:
        ax.plot(generations, [row[key] for row in history], label=label)
    ax.set(xlabel="Generasi", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def save_result_plots(result, out_dir, stem):
    out = Path(out_dir)
    objective = result["parameters"].get("objective", "shipping_fee")
    fig = history_figure(
        result["history"],
        [("candidate_peak", "Fitness maksimum"), ("mean_fitness", "Fitness rata-rata"),
         ("best_valid_objective_so_far", "Objective valid terbaik")],
        "Nilai", f"Genetic Algorithm | {objective}",
    )
    fitness_path = out / f"{stem}.png"
    fig.savefig(fitness_path, dpi=130)
    _pyplot().close(fig)
    plot_state(result["final_state"], result["vehicles"], result["ship"],
               out / f"{stem}_solution.png", "genetic_algorithm - Final", objective)
    return {"png": fitness_path, "solution_png": out / f"{stem}_solution.png"}


def save_plots(result, vehicles, ship, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objective = result["parameters"]["objective"]
    plot_state(result["initial_state"], vehicles, ship, output_dir / "initial.png",
               "genetic_algorithm - Initial", objective)
    plot_state(result["final_state"], vehicles, ship, output_dir / "final.png",
               "genetic_algorithm - Final", objective)
    fig = history_figure(
        result["history"],
        [("candidate_peak", "Fitness maksimum"), ("mean_fitness", "Fitness rata-rata"),
         ("best_valid_objective_so_far", "Objective valid terbaik")],
        "Nilai", f"Genetic Algorithm | {objective}",
    )
    fig.savefig(output_dir / "history.png", dpi=130)
    _pyplot().close(fig)
