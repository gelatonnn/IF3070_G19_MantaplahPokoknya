"""Unit tests untuk Genetic Algorithm."""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Core.data_loader import load_data
from src.Core.evaluation import evaluate
from src.Core.state import random_state, validate_state, empty_state
from src.Algorithm import genetic_algorithm as ga
from src.Visualization import export_result as ex

SHIP, VEHICLES = load_data(ROOT / 'data' / 'mini_data.json')


def strip(result):
    return {k: v for k, v in result.items() if k != 'duration_seconds'}


def mutable_ids(state):
    ids = {id(state)}
    for placement in state.values():
        ids.add(id(placement))
    return ids


class GATests(unittest.TestCase):
    def test_crossover_keeps_ids_bounds_and_independence(self):
        rng = random.Random(1)
        for _ in range(200):
            p1 = random_state(VEHICLES, SHIP, rng)
            p2 = random_state(VEHICLES, SHIP, rng)
            c1, c2 = ga.uniform_crossover(p1, p2, rng)
            for child in (c1, c2):
                self.assertEqual(sorted(child), sorted(VEHICLES))
                validate_state(child, VEHICLES, SHIP)
                evaluate(child, VEHICLES, SHIP)
                self.assertFalse(mutable_ids(child) & (mutable_ids(p1) | mutable_ids(p2)))
            self.assertFalse(mutable_ids(c1) & mutable_ids(c2))
            for vid in VEHICLES:
                self.assertIn(c1[vid], (p1[vid], p2[vid]))

    def test_no_shared_objects_in_result(self):
        r = ga.run_ga(SHIP, VEHICLES, 7, 10, seed=3)
        seen = set()
        groups = (r['initial_population'] + r['final_population']
                  + [r['initial_best_state'], r['best_fitness_state'], r['best_valid_state']])
        total = 0
        for state in groups:
            ids = mutable_ids(state)
            total += len(ids)
            seen |= ids
        self.assertEqual(len(seen), total)

    def test_population_size_fixed_including_odd(self):
        for size, elite in ((5, 2), (7, 0), (3, 3), (2, 1), (9, 2)):
            r = ga.run_ga(SHIP, VEHICLES, size, 5, tournament_size=2, elite_count=elite, seed=1)
            self.assertEqual(len(r['initial_population']), size)
            self.assertEqual(len(r['final_population']), size)

    def test_negative_fitness_selection(self):
        pop = [{}, {}, {}]
        fits = [-500, -10, -300]
        rng = random.Random(0)
        self.assertEqual(ga.tournament_select(pop, fits, 3, rng), 1)
        identical = [{}] * 4
        self.assertIn(ga.tournament_select(identical, [-5] * 4, 3, rng), range(4))
        r = ga.run_ga(SHIP, VEHICLES, 20, 3, seed=5)
        self.assertLess(min(h['mean_fitness'] for h in r['history']), 0)

    def test_same_seed_same_result_different_seed_differs(self):
        a = ga.run_ga(SHIP, VEHICLES, 20, 15, seed=11)
        b = ga.run_ga(SHIP, VEHICLES, 20, 15, seed=11)
        self.assertEqual(strip(a), strip(b))
        c = ga.run_ga(SHIP, VEHICLES, 20, 15, seed=12)
        self.assertNotEqual(a['initial_population'], c['initial_population'])

    def test_history_and_elitism(self):
        r = ga.run_ga(SHIP, VEHICLES, 30, 40, seed=2)
        h = r['history']
        self.assertEqual(len(h), 41)
        self.assertEqual([x['generation'] for x in h], list(range(41)))
        peaks = [x['candidate_peak'] for x in h]
        self.assertTrue(all(b >= a for a, b in zip(peaks, peaks[1:])))
        best = [x['best_valid_revenue_so_far'] for x in h]
        self.assertTrue(all(b >= a for a, b in zip(best, best[1:])))
        self.assertEqual(r['generations_run'], 40)

    def test_without_elitism_runs(self):
        r = ga.run_ga(SHIP, VEHICLES, 10, 5, elite_count=0, seed=1)
        self.assertEqual(len(r['history']), 6)

    def test_final_solution_valid(self):
        for seed in range(5):
            r = ga.run_ga(SHIP, VEHICLES, 30, 30, seed=seed)
            m = evaluate(r['best_valid_state'], VEHICLES, SHIP)
            self.assertTrue(m['is_valid'])
            self.assertEqual(m['revenue'], r['best_valid_metrics']['revenue'])
            self.assertEqual(r['best_valid_metrics']['revenue'],
                             r['history'][-1]['best_valid_revenue_so_far'])

    def test_fallback_flag(self):
        vehicles = {'X': {'id': 'X', 'w': 1, 'l': 1, 'shipping_fee': 5, 'weight': 99, 'eta': 0}}
        r = ga.run_ga(SHIP, vehicles, 4, 3, seed=1)
        if r['best_valid_from_fallback']:
            self.assertEqual(r['best_valid_state'], empty_state(vehicles))
            self.assertEqual(r['best_valid_metrics']['revenue'], 0)

    def test_parameter_validation(self):
        bad = [dict(population_size=1), dict(generations=0), dict(crossover_rate=1.5),
                dict(mutation_rate=-0.1), dict(tournament_size=0), dict(tournament_size=51),
                dict(elite_count=-1), dict(elite_count=51), dict(population_size=2.5),
                dict(seed='a')]
        for override in bad:
            with self.assertRaises(ValueError, msg=str(override)):
                ga.run_ga(SHIP, VEHICLES, **{**dict(population_size=50, generations=2), **override})

    def test_export_does_not_overwrite(self):
        import tempfile
        r = ga.run_ga(SHIP, VEHICLES, 6, 3, seed=1)
        with tempfile.TemporaryDirectory() as folder:
            first = ex.export_result(r, folder, 'mini_data')
            second = ex.export_result(r, folder, 'mini_data')
            self.assertNotEqual(first['csv'], second['csv'])
            self.assertTrue(first['csv'].exists() and second['csv'].exists())

    def test_repair_operator_guarantees_validity(self):
        rng = random.Random(42)
        for _ in range(50):
            state = random_state(VEHICLES, SHIP, rng, load_probability=0.9)
            repaired = ga.repair_state(state, VEHICLES, SHIP)
            m = evaluate(repaired, VEHICLES, SHIP)
            self.assertTrue(m['is_valid'])
            self.assertEqual(m['overlap'], 0)
            self.assertEqual(m['excess_weight'], 0)

    def test_run_ga_with_repair(self):
        r = ga.run_ga(SHIP, VEHICLES, 10, 5, seed=1, repair=True)
        self.assertTrue(r['best_valid_metrics']['is_valid'])
        self.assertTrue(r['params']['repair'])

    def test_cli_does_not_import_streamlit(self):
        self.assertNotIn('streamlit', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
