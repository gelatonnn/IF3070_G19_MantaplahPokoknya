import csv
import io
from datetime import datetime
from pathlib import Path

HISTORY_FIELDS = ['generation', 'candidate_peak', 'mean_fitness', 'valid_count',
                  'best_valid_revenue_so_far']


def history_to_csv(history):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows([{k: row[k] for k in HISTORY_FIELDS} for row in history])
    return buffer.getvalue()


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
        from src.Visualization.plots import save_result_plots
        paths.update(save_result_plots(result, out, stem))
    except ImportError:
        pass
    return paths
