import csv
import io
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Core.evaluation import evaluate  # noqa: E402
from src.Core.state import footprint  # noqa: E402

HISTORY_FIELDS = [
    'generation',
    'candidate_peak',
    'mean_fitness',
    'valid_count',
    'best_valid_revenue_so_far'
]


def history_to_csv(history):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows([{k: row[k] for k in HISTORY_FIELDS} for row in history])
    return buffer.getvalue()


def vehicle_colors(vehicles):
    palette = [
        '#b3cde3', '#ccebc5', '#decbe4', '#fed9a6', '#ffffcc',
        '#e5d8bd', '#fddaec', '#f2f2f2', '#fbb4ae', '#cbd5e8',
        '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02',
        '#a6761d', '#666666', '#8dd3c7', '#ffffb3', '#bebada'
    ]
    return {vid: palette[i % len(palette)] for i, vid in enumerate(vehicles)}


def draw_state(state, vehicles, ship, colors, stage='final', algo_name='genetic_algorithm', title=None):
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    metrics = evaluate(state, vehicles, ship)
    cols, rows = ship['l'], ship['w']

    fig = Figure(figsize=(10, 7.2), dpi=130)
    ax = fig.add_axes([0.10, 0.26, 0.82, 0.62])

    if title is None:
        status_text = 'VALID' if metrics['is_valid'] else 'BELUM VALID'
        title_line1 = f"{algo_name} - {stage} | {status_text}"
        title_line2 = f"Score {metrics['score']} | Fee {metrics['revenue']} | Berat {metrics['total_weight']}/{ship['max_capacity']} | Overlap {metrics['overlap']} | Excess {metrics['excess_weight']}"
        fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=11, y=0.96, fontweight='normal')
    else:
        fig.suptitle(title, fontsize=11, y=0.96, fontweight='normal')

    ax.add_patch(Rectangle((0, 0), cols, rows, facecolor='white', edgecolor='black', lw=1.0))

    for vid, p in sorted(state.items()):
        if p['position'] is None:
            continue
        x, y = p['position']
        orient = p['orientation']
        dx, dy = footprint(vehicles[vid], orient)
        ax.add_patch(Rectangle((x, y), dx, dy, facecolor=colors[vid], edgecolor='black', lw=0.6, alpha=0.9))
        ax.text(x + dx / 2, y + dy / 2, f"{vid}\n{orient}", ha='center', va='center',
                fontsize=6.5 if cols >= 15 else 8, color='#111111')

    for yy, row in enumerate(metrics['grid']):
        for xx, count in enumerate(row):
            if count > 1:
                ax.add_patch(Rectangle((xx, yy), 1, 1, facecolor='none', edgecolor='red', hatch='///', lw=0))

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect('equal')
    ax.set_xticks(range(cols + 1))
    ax.set_yticks(range(rows + 1))
    ax.set_xlabel('x / kolom', fontsize=9)
    ax.set_ylabel('y / baris', fontsize=9)
    ax.grid(True, color='#e2e2e2', lw=0.5, alpha=0.9)
    ax.tick_params(labelsize=8)

    loaded_vids = [vid for vid, p in sorted(state.items()) if p['position'] is not None]
    unloaded_items = [f"{vid}({p['orientation']})" for vid, p in sorted(state.items()) if p['position'] is None]

    footer_lines = [
        f"Di dalam: {len(loaded_vids)} | Di luar: {len(unloaded_items)}",
        "Kendaraan di luar (orientasi tetap disimpan):"
    ]
    chunk_size = 12 if cols >= 15 else 8
    for i in range(0, len(unloaded_items), chunk_size):
        chunk = ", ".join(unloaded_items[i:i + chunk_size])
        if i + chunk_size < len(unloaded_items):
            chunk += ","
        footer_lines.append(chunk)
    footer_lines.append("Arsir merah = sel overlap. Anchor = pojok kiri-atas.")

    full_footer = "\n".join(footer_lines)
    fig.text(0.05, 0.03, full_footer, fontsize=7.2, ha='left', va='bottom', family='sans-serif', linespacing=1.35)

    return fig


def history_figure(history, keys=None, ylabel='Nilai fitness', title=None,
                   final_score=None, duration=None, params=None, algo_name='genetic algorithm'):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.5, 4.2), dpi=140)
    ax = fig.subplots()

    gens = [h['generation'] for h in history]

    if keys is None:
        keys = [
            ('candidate_peak', 'Fitness maksimum populasi (candidate_peak)', '#1f77b4'),
            ('mean_fitness', 'Fitness rata-rata populasi (mean_fitness)', '#2ca02c')
        ]

    for item in keys:
        if len(item) == 3:
            k, lbl, color = item
            ax.plot(gens, [h[k] for h in history], label=lbl, color=color, lw=1.5)
        else:
            k, lbl = item
            ax.plot(gens, [h[k] for h in history], label=lbl, lw=1.5)

    if final_score is not None:
        ax.axhline(final_score, color='red', linestyle='--', lw=1.2,
                   label=f'best valid = {final_score}')

    if params and duration is not None and final_score is not None:
        line1 = f"{algo_name} | final score {final_score} | {duration:.2f} detik"
        line2 = (f"pop_size={params.get('population_size')}, generations={params.get('generations')}, "
                 f"crossover={params.get('crossover_rate')}, mutation={params.get('mutation_rate')}, "
                 f"tournament={params.get('tournament_size')}, elitism={params.get('elite_count')}, "
                 f"seed={params.get('seed')}")
        ax.set_title(f"{line1}\n{line2}", fontsize=9.5)
    elif title:
        ax.set_title(title, fontsize=10)

    ax.set_xlabel('Generasi', fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.5)

    pos_means = [h['mean_fitness'] for h in history if h['mean_fitness'] > 0]
    peak_max = max(h['candidate_peak'] for h in history)
    ref_max = max(peak_max, final_score or 0)
    if pos_means:
        y_bot = max(0, min(pos_means) - 50)
        y_top = ref_max + max(40, (ref_max - y_bot) * 0.05)
        ax.set_ylim(y_bot, y_top)

    ax.grid(True, linestyle='-', color='#ebebeb', lw=0.7)
    ax.legend(loc='lower right', fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    return fig


def save_result_plots(result, out_dir, stem):
    out = Path(out_dir)
    colors = vehicle_colors(result['vehicles'])

    fig_init = draw_state(result['initial_best_state'], result['vehicles'], result['ship'],
                          colors, stage='initial')
    init_path = out / f'{stem}_initial.png'
    fig_init.savefig(init_path, dpi=130)

    fig_final = draw_state(result['best_valid_state'], result['vehicles'], result['ship'],
                           colors, stage='final')
    final_path = out / f'{stem}_final.png'
    fig_final.savefig(final_path, dpi=130)

    p = result.get('params', {})
    duration = result.get('duration_seconds')
    final_score = result.get('best_valid_metrics', {}).get('revenue')
    fig_hist = history_figure(
        result['history'],
        ylabel='Nilai fitness',
        final_score=final_score,
        duration=duration,
        params=p,
        algo_name='genetic algorithm'
    )
    fitness_path = out / f'{stem}_fitness.png'
    fig_hist.savefig(fitness_path, dpi=130)

    return {'initial_png': init_path, 'final_png': final_path, 'fitness_png': fitness_path}


def export_result(result, output_dir, dataset_name):
    p = result['params']
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    base = (f"ga_{dataset_name}_p{p['population_size']}_g{p['generations']}"
            f"_c{p['crossover_rate']}_m{p['mutation_rate']}_t{p['tournament_size']}"
            f"_e{p['elite_count']}_s{p['seed']}_{stamp}")
    suffix = ''
    counter = 1
    while any((out / f'{base}{suffix}{ext}').exists()
              for ext in ('.csv', '_initial.png', '_final.png', '_fitness.png')):
        suffix = f'_{counter}'
        counter += 1
    stem = f'{base}{suffix}'
    paths = {'csv': out / f'{stem}.csv'}
    with paths['csv'].open('x', encoding='utf-8', newline='') as f:
        f.write(history_to_csv(result['history']))
    try:
        paths.update(save_result_plots(result, out, stem))
    except ImportError:
        pass
    return paths
