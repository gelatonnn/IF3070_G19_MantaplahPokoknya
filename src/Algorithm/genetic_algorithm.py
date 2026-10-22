import argparse
import random
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Core.data_loader import load_data  # noqa: E402
from src.Core.evaluation import evaluate  # noqa: E402
from src.Core.neighbors import random_neighbor  # noqa: E402
from src.Core.state import copy_state, empty_state, footprint, random_state  # noqa: E402

DEFAULT_PARAMS = {
    'population_size': 100,
    'generations': 300,
    'crossover_rate': 0.8,
    'mutation_rate': 0.5,
    'tournament_size': 3,
    'elite_count': 2,
    'seed': 42,
    'repair': False,
}


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_params(population_size, generations, crossover_rate, mutation_rate,
                    tournament_size, elite_count, seed=None, repair=False):
    if not isinstance(repair, bool):
        raise ValueError('repair harus berupa boolean (True atau False).')
    for name, value in (('population_size', population_size), ('generations', generations),
                        ('tournament_size', tournament_size), ('elite_count', elite_count)):
        if not _is_int(value):
            raise ValueError(f'{name} harus bilangan bulat.')
    if seed is not None and not _is_int(seed):
        raise ValueError('seed harus bilangan bulat (atau kosong).')
    if population_size < 2:
        raise ValueError('population_size minimal 2.')
    if generations < 1:
        raise ValueError('generations minimal 1.')
    for name, value in (('crossover_rate', crossover_rate), ('mutation_rate', mutation_rate)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'{name} harus berupa angka antara 0 dan 1.')
    if not 1 <= tournament_size <= population_size:
        raise ValueError('tournament_size harus antara 1 dan population_size '
                         f'({population_size}); populasi lebih kecil dari turnamen tidak dapat dipilih.')
    if not 0 <= elite_count <= population_size:
        raise ValueError(f'elite_count harus antara 0 dan population_size ({population_size}).')


def metrics_of(state, vehicles, ship):
    result = evaluate(state, vehicles, ship)
    result.pop('grid')
    result['fitness'] = result['score']
    return result


def tournament_select(population, fitnesses, tournament_size, rng):
    if not 1 <= tournament_size <= len(population):
        raise ValueError('tournament_size harus 1..ukuran populasi')
    contenders = rng.sample(range(len(population)), tournament_size)
    return max(contenders, key=lambda i: fitnesses[i])


def uniform_crossover(parent1, parent2, rng):
    child1, child2 = {}, {}
    for vid in parent1:
        if rng.random() < 0.5:
            child1[vid], child2[vid] = deepcopy(parent1[vid]), deepcopy(parent2[vid])
        else:
            child1[vid], child2[vid] = deepcopy(parent2[vid]), deepcopy(parent1[vid])
    return child1, child2


def mutate(state, vehicles, ship, rng):
    mutated = random_neighbor(state, vehicles, ship, rng)
    return state if mutated is None else mutated


def repair_state(state, vehicles, ship):
    w_ship, l_ship = ship['w'], ship['l']
    grid = [[0] * l_ship for _ in range(w_ship)]
    total_weight = 0
    overlap_cells = 0
    loaded = []

    for vid, p in state.items():
        if p['position'] is None:
            continue
        v = vehicles[vid]
        total_weight += v['weight']
        loaded.append(vid)
        x, y = p['position']
        dx, dy = footprint(v, p['orientation'])
        for yy in range(y, y + dy):
            for xx in range(x, x + dx):
                grid[yy][xx] += 1
                if grid[yy][xx] == 2:
                    overlap_cells += 1

    if overlap_cells == 0 and total_weight <= ship['max_capacity']:
        return state

    result = copy_state(state)

    if overlap_cells > 0:
        candidates = sorted(
            loaded,
            key=lambda vid: (
                vehicles[vid]['shipping_fee'] / max(1, vehicles[vid]['w'] * vehicles[vid]['l']),
                vehicles[vid]['shipping_fee']
            )
        )
        for vid in candidates:
            if overlap_cells == 0:
                break
            p = result[vid]
            if p['position'] is None:
                continue
            v = vehicles[vid]
            x, y = p['position']
            dx, dy = footprint(v, p['orientation'])
            in_overlap = any(grid[yy][xx] > 1 for yy in range(y, y + dy) for xx in range(x, x + dx))

            if in_overlap:
                result[vid]['position'] = None
                total_weight -= v['weight']
                for yy in range(y, y + dy):
                    for xx in range(x, x + dx):
                        if grid[yy][xx] == 2:
                            overlap_cells -= 1
                        grid[yy][xx] -= 1

    if total_weight > ship['max_capacity']:
        loaded_now = [vid for vid, p in result.items() if p['position'] is not None]
        loaded_now.sort(
            key=lambda vid: (
                vehicles[vid]['shipping_fee'] / max(1, vehicles[vid]['weight']),
                vehicles[vid]['shipping_fee']
            )
        )
        for vid in loaded_now:
            if total_weight <= ship['max_capacity']:
                break
            result[vid]['position'] = None
            total_weight -= vehicles[vid]['weight']

    return result


def run_ga(ship, vehicles, population_size=100, generations=300, crossover_rate=0.8,
           mutation_rate=0.5, tournament_size=3, elite_count=2, seed=42, repair=False):
    validate_params(population_size, generations, crossover_rate, mutation_rate,
                    tournament_size, elite_count, seed)
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 31)
    rng = random.Random(seed)

    start = perf_counter()

    population = [random_state(vehicles, ship, rng) for _ in range(population_size)]
    initial_population = [copy_state(s) for s in population]

    fitnesses = [metrics_of(s, vehicles, ship)['fitness'] for s in population]

    initial_best_index = max(range(population_size), key=lambda i: fitnesses[i])
    initial_best_state = copy_state(population[initial_best_index])

    best_fitness = {'fitness': None, 'state': None, 'generation': None}
    best_valid = {'revenue': None, 'state': None, 'generation': None}

    def observe(pop, fits, generation):
        valid_count = 0
        for state, fit in zip(pop, fits):
            if best_fitness['fitness'] is None or fit > best_fitness['fitness']:
                best_fitness.update(fitness=fit, state=copy_state(state), generation=generation)
            m = evaluate(state, vehicles, ship)
            if m['is_valid']:
                valid_count += 1
                if best_valid['revenue'] is None or m['revenue'] > best_valid['revenue']:
                    best_valid.update(revenue=m['revenue'], state=copy_state(state),
                                      generation=generation)
        so_far = best_valid['revenue'] if best_valid['revenue'] is not None else 0
        return {'generation': generation, 'candidate_peak': max(fits),
                'mean_fitness': sum(fits) / len(fits), 'valid_count': valid_count,
                'best_valid_revenue_so_far': so_far}

    history = [observe(population, fitnesses, 0)]

    for generation in range(1, generations + 1):
        order = sorted(range(population_size), key=lambda i: fitnesses[i], reverse=True)
        next_population = [copy_state(population[i]) for i in order[:elite_count]]
        next_fitnesses = [fitnesses[i] for i in order[:elite_count]]

        children = []
        slots = population_size - elite_count
        while len(children) < slots:
            p1 = population[tournament_select(population, fitnesses, tournament_size, rng)]
            p2 = population[tournament_select(population, fitnesses, tournament_size, rng)]
            if rng.random() < crossover_rate:
                c1, c2 = uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = copy_state(p1), copy_state(p2)
            for child in (c1, c2):
                if len(children) < slots:
                    if rng.random() < mutation_rate:
                        child = mutate(child, vehicles, ship, rng)
                    if repair:
                        child = repair_state(child, vehicles, ship)
                    children.append(child)
        for child in children:
            next_population.append(child)
            next_fitnesses.append(metrics_of(child, vehicles, ship)['fitness'])

        population, fitnesses = next_population, next_fitnesses
        history.append(observe(population, fitnesses, generation))

    duration = perf_counter() - start

    from_fallback = best_valid['state'] is None
    if from_fallback:
        final_valid_state = empty_state(vehicles)
        best_valid_generation = None
    else:
        final_valid_state = best_valid['state']
        best_valid_generation = best_valid['generation']
    final_valid_metrics = metrics_of(final_valid_state, vehicles, ship)
    if not final_valid_metrics['is_valid']:
        raise RuntimeError('Solusi akhir ternyata tidak valid menurut evaluate()')

    return {
        'params': {'population_size': population_size, 'generations': generations,
                   'crossover_rate': crossover_rate, 'mutation_rate': mutation_rate,
                   'tournament_size': tournament_size, 'elite_count': elite_count,
                   'seed': seed, 'repair': repair},
        'ship': dict(ship),
        'vehicles': deepcopy(vehicles),
        'initial_population': initial_population,
        'initial_best_state': initial_best_state,
        'initial_best_metrics': metrics_of(initial_best_state, vehicles, ship),
        'final_population': [copy_state(s) for s in population],
        'best_fitness_state': best_fitness['state'],
        'best_fitness_metrics': metrics_of(best_fitness['state'], vehicles, ship),
        'best_fitness_generation': best_fitness['generation'],
        'best_valid_state': copy_state(final_valid_state),
        'best_valid_metrics': final_valid_metrics,
        'best_valid_from_fallback': from_fallback,
        'best_valid_generation': best_valid_generation,
        'duration_seconds': duration,
        'generations_run': generations,
        'history': history,
    }


def build_parser():
    d = DEFAULT_PARAMS
    parser = argparse.ArgumentParser(description='Genetic Algorithm untuk geladak kapal')
    parser.add_argument('--data', default='data/mini_data.json', help='path dataset JSON')
    parser.add_argument('--population-size', type=int, default=d['population_size'])
    parser.add_argument('--generations', type=int, default=d['generations'])
    parser.add_argument('--crossover-rate', type=float, default=d['crossover_rate'])
    parser.add_argument('--mutation-rate', type=float, default=d['mutation_rate'])
    parser.add_argument('--tournament-size', type=int, default=d['tournament_size'])
    parser.add_argument('--elite-count', type=int, default=d['elite_count'])
    parser.add_argument('--seed', type=int, default=d['seed'])
    parser.add_argument('--repair', action='store_true',
                        help='aktifkan operator repair untuk memperbaiki overlap dan kelebihan muatan')
    parser.add_argument('--output-directory', default=str(ROOT / 'output' / 'genetic_algorithm'),
                        help='folder hasil CSV riwayat dan gambar PNG')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    data_path = Path(args.data)
    if not data_path.exists() and (ROOT / data_path).exists():
        data_path = ROOT / data_path
    ship, vehicles = load_data(data_path)
    try:
        result = run_ga(ship, vehicles, args.population_size, args.generations,
                        args.crossover_rate, args.mutation_rate, args.tournament_size,
                        args.elite_count, args.seed, repair=args.repair)
    except ValueError as error:
        print(f'Parameter tidak valid: {error}', file=sys.stderr)
        return 2

    from src.Visualization.export_result import export_result
    paths = export_result(result, args.output_directory, data_path.stem)

    bv, bf = result['best_valid_metrics'], result['best_fitness_metrics']
    print(f"Dataset          : {data_path} ({len(vehicles)} kendaraan)")
    print(f"Parameter        : {result['params']}")
    print(f"Durasi pencarian : {result['duration_seconds']:.3f} detik")
    print(f"Solusi valid terbaik: revenue={bv['revenue']} berat={bv['total_weight']}/"
          f"{ship['max_capacity']} dimuat={bv['loaded_count']} "
          f"(dari fallback kapal kosong: {result['best_valid_from_fallback']})")
    print(f"Fitness tertinggi (bisa melanggar): {bf['fitness']} valid={bf['is_valid']}")
    for kind, path in paths.items():
        print(f'File {kind}: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
