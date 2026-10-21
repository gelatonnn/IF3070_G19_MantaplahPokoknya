"""plot khusus simulated annealing"""

#initial.png & final.png pake plot_state punya hill climbing biar tampilannya sama
#history.png    -> score sama suhu terhadap iterasi
#acceptance.png -> e^(delta/T) terhadap iterasi, cuma langkah yg lebih jelek

from pathlib import Path

from .hill_climbing import _pyplot, plot_state


def plot_history(result, trace, path):
    plt = _pyplot()
    scores = trace["scores"]
    iterations = range(len(scores))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    #atas: score state sekarang + garis best valid
    axes[0].plot(iterations, scores, linewidth=0.8, label="score state sekarang")
    best = result["final_metrics"]["score"]
    axes[0].axhline(best, color="#c62828", linestyle="--", linewidth=1,
                    label=f"best valid = {best}")
    #score awal bisa minus ratusan ribu gara2 penalti, jadi pake symlog
    #biar bagian akhir yg deket best valid tetep keliatan
    axes[0].set_yscale("symlog")
    axes[0].set_ylabel("score (fee - penalti), symlog")
    axes[0].legend()

    #bawah: suhu, pake skala log jadi pendinginan geometris keliatan lurus
    axes[1].plot(iterations, trace["temperatures"], color="#ef6c00", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("suhu T (log)")
    axes[1].set_xlabel("iterasi")

    for ax in axes:
        ax.grid(alpha=0.25)
    p = result["parameters"]
    fig.suptitle(f"simulated annealing | final score {best} | "
                 f"{result['duration_seconds']:.2f} detik\n"
                 f"T0={p['t0']}, alpha={p['alpha']}, Tmin={p['t_min']}, "
                 f"iterasi={result['iterations']}, stuck={result['stuck_count']}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_acceptance(result, trace, path):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(11, 5))
    points = trace["acceptance"]
    if points:
        xs = [it for it, _ in points]
        ys = [prob for _, prob in points]
        #titiknya puluhan ribu, jadi dibikin kecil & agak transparan
        ax.scatter(xs, ys, s=1, alpha=0.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("iterasi")
    ax.set_ylabel("e^(ΔE / T)")
    ax.grid(alpha=0.25)
    ax.set_title(f"peluang nerima langkah lebih jelek (ΔE < 0) | "
                 f"diterima {result['accepted_worse_moves']}, "
                 f"ditolak {result['rejected_moves']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_plots(result, trace, vehicles, ship, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_state(result["initial_state"], vehicles, ship, output_dir / "initial.png",
               "simulated_annealing - initial")
    plot_state(result["final_state"], vehicles, ship, output_dir / "final.png",
               "simulated_annealing - final")
    plot_history(result, trace, output_dir / "history.png")
    plot_acceptance(result, trace, output_dir / "acceptance.png")
