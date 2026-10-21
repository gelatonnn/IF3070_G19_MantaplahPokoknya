import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Core.evaluation import evaluate  # noqa: E402
from src.Core.state import footprint  # noqa: E402


def vehicle_colors(vehicles):
    from matplotlib import colormaps
    palette = [colormaps['tab20'](i) for i in range(20)]
    return {vid: palette[i % len(palette)] for i, vid in enumerate(vehicles)}


def draw_state(state, vehicles, ship, colors, title=''):
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    metrics = evaluate(state, vehicles, ship)
    cols, rows = ship['l'], ship['w']
    scale = max(0.45, min(0.8, 9 / max(cols, 1)))
    fig = Figure(figsize=(max(4, cols * scale + 1), max(3, rows * scale + 1)))
    ax = fig.subplots()
    ax.add_patch(Rectangle((0, 0), cols, rows, facecolor='#f4f4f4', edgecolor='black', lw=1.5))
    for vid, placement in state.items():
        if placement['position'] is None:
            continue
        x, y = placement['position']
        dx, dy = footprint(vehicles[vid], placement['orientation'])
        ax.add_patch(Rectangle((x, y), dx, dy, facecolor=colors[vid], alpha=0.85,
                               edgecolor='black', lw=1))
        ax.text(x + dx / 2, y + dy / 2, vid, ha='center', va='center',
                fontsize=8 if cols > 12 else 10, fontweight='bold')
    for yy, row in enumerate(metrics['grid']):
        for xx, count in enumerate(row):
            if count > 1:
                ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor='none', edgecolor='red',
                                       hatch='xxx', lw=0))
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, cols + 1, max(1, cols // 10)))
    ax.set_yticks(range(0, rows + 1, max(1, rows // 10)))
    ax.set_xlabel('x (kolom)')
    ax.set_ylabel('y (baris)')
    ax.grid(True, color='#bbbbbb', lw=0.4)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def history_figure(history, keys, ylabel, title):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(7, 3.6))
    ax = fig.subplots()
    gens = [h['generation'] for h in history]
    for key, label in keys:
        ax.plot(gens, [h[key] for h in history], label=label)
    ax.set_xlabel('Generasi')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def save_result_plots(result, out_dir, stem):
    out = Path(out_dir)
    fig = history_figure(result['history'],
                         [('candidate_peak', 'candidate_peak (fitness maksimum)'),
                          ('mean_fitness', 'mean_fitness')],
                         'Fitness', 'Fitness populasi per generasi')
    fitness_path = out / f'{stem}.png'
    fig.savefig(fitness_path, dpi=120)

    best = result['best_valid_metrics']
    title = f"Solusi valid terbaik (revenue {best['revenue']})"
    if result['best_valid_from_fallback']:
        title += ' - FALLBACK kapal kosong'
    fig = draw_state(result['best_valid_state'], result['vehicles'], result['ship'],
                     vehicle_colors(result['vehicles']), title)
    solution_path = out / f'{stem}_solution.png'
    fig.savefig(solution_path, dpi=120)
    return {'png': fitness_path, 'solution_png': solution_path}
