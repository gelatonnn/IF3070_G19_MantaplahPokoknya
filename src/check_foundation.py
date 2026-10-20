import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from data_loader import load_data
from state import empty_state, random_state, validate_state, copy_state
from evaluation import evaluate
from neighbors import move, rotate, swap, iter_neighbors, random_neighbor

SHIP = {'w': 4, 'l': 6, 'max_capacity': 30}
VEHICLES = {
    'A': {'id': 'A', 'w': 2, 'l': 3, 'shipping_fee': 40, 'weight': 12, 'eta': 0},
    'B': {'id': 'B', 'w': 2, 'l': 2, 'shipping_fee': 25, 'weight': 8, 'eta': 0},
    'C': {'id': 'C', 'w': 1, 'l': 2, 'shipping_fee': 10, 'weight': 3, 'eta': 0},
    'D': {'id': 'D', 'w': 2, 'l': 4, 'shipping_fee': 50, 'weight': 15, 'eta': 0}}


def reference_state():
    s = empty_state(VEHICLES)
    s['A']['position'] = (0, 0)
    s['B']['position'] = (0, 2)
    s['C'] = {'position': (3, 0), 'orientation': 'V'}
    return s


class FoundationTests(unittest.TestCase):
    def test_manual_scores(self):
        s1 = reference_state()
        s2 = move(s1, 'C', (2, 1), VEHICLES, SHIP)
        s3 = swap(s1, 'B', 'D', VEHICLES, SHIP)
        s4 = move(s1, 'D', (2, 2), VEHICLES, SHIP)
        for state, expected in zip((s1, s2, s3, s4),
                                  ((75, 0, 0), (24, 1, 0), (100, 0, 0), (-283, 0, 8))):
            result = evaluate(state, VEHICLES, SHIP)
            self.assertEqual((result['score'], result['overlap'], result['excess_weight']), expected)
        self.assertEqual(evaluate(s3, VEHICLES, SHIP)['total_weight'], 30)

    def test_bounds_and_no_mutation(self):
        state = reference_state()
        before = copy_state(state)
        new = swap(state, 'B', 'D', VEHICLES, SHIP)
        new['A']['orientation'] = 'V'
        self.assertEqual(state, before)
        with self.assertRaises(ValueError):
            move(state, 'D', (3, 0), VEHICLES, SHIP)
        state4 = move(state, 'D', (2, 2), VEHICLES, SHIP)
        with self.assertRaises(ValueError):
            rotate(state4, 'D', VEHICLES, SHIP)
        invalid = copy_state(state)
        invalid['A']['position'] = (-1, 0)
        with self.assertRaises(ValueError):
            evaluate(invalid, VEHICLES, SHIP)
        self.assertEqual(state, before)

    def test_excess_occupants_not_pairs(self):
        state = empty_state(VEHICLES)
        for vid in ('A', 'B', 'D'):
            state[vid]['position'] = (0, 0)
        self.assertEqual(evaluate(state, VEHICLES, SHIP)['overlap'], 10)

    def test_random_and_all_neighbors(self):
        self.assertEqual(random_state(VEHICLES, SHIP, random.Random(10)),
                         random_state(VEHICLES, SHIP, random.Random(10)))
        state = reference_state()
        before = copy_state(state)
        candidates = list(iter_neighbors(state, VEHICLES, SHIP))
        keys = {repr(c) for c in candidates}
        self.assertEqual(len(keys), len(candidates))
        for candidate in candidates:
            validate_state(candidate, VEHICLES, SHIP)
            self.assertNotEqual(candidate, state)
        for vid in ('A', 'B', 'C'):
            self.assertIn(move(state, vid, None, VEHICLES, SHIP), candidates)
        for seed in range(100):
            initial = random_state(VEHICLES, SHIP, random.Random(seed))
            evaluate(initial, VEHICLES, SHIP)
            candidate = random_neighbor(initial, VEHICLES, SHIP, random.Random(seed))
            self.assertIsNotNone(candidate)
            evaluate(candidate, VEHICLES, SHIP)
        self.assertEqual(state, before)

    def test_loader_and_bad_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.json'
            payload = {'ship': SHIP, 'vehicles': list(VEHICLES.values())}
            path.write_text(json.dumps(payload), encoding='utf-8')
            self.assertEqual(load_data(path), (SHIP, VEHICLES))
            payload['vehicles'].append(dict(VEHICLES['A']))
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_data(path)
        state = empty_state(VEHICLES)
        del state['D']
        with self.assertRaises(ValueError):
            evaluate(state, VEHICLES, SHIP)

    def test_empty_and_oversized(self):
        self.assertTrue(evaluate({}, {}, SHIP)['is_valid'])
        self.assertIsNone(random_neighbor({}, {}, SHIP))
        huge = {'X': {'w': 99, 'l': 99, 'weight': 1, 'shipping_fee': 1}}
        state = random_state(huge, SHIP, random.Random(1), load_probability=1)
        self.assertIsNone(state['X']['position'])
        evaluate(state, huge, SHIP)


if __name__ == '__main__':
    dataset = sys.argv[1:]
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FoundationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if dataset:
        ship, vehicles = load_data(dataset[0])
        state = random_state(vehicles, ship, random.Random(42))
        metrics = evaluate(state, vehicles, ship)
        for seed in range(100):
            candidate = random_neighbor(state, vehicles, ship, random.Random(seed))
            if candidate is not None:
                evaluate(candidate, vehicles, ship)
        print('Dataset:', len(vehicles), 'vehicles; penalty:', metrics['penalty_coefficient'])
        print('Initial state (can violate overlap/weight):', {k:v for k,v in metrics.items() if k != 'grid'})
