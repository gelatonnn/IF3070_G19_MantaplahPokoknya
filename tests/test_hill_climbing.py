from copy import deepcopy
import random
import unittest

from src.Algorithm import (
    hill_climbing_sideways, random_restart_hill_climbing,
    steepest_ascent_hill_climbing, stochastic_hill_climbing,
)
from src.Algorithm._neighborhood import Neighborhood
from src.Core.evaluation import evaluate
from src.Core.neighbors import iter_neighbors
from src.Core.state import empty_state, random_state


SHIP = {"w": 2, "l": 3, "max_capacity": 5}
VEHICLES = {
    "A": {"w": 1, "l": 2, "weight": 3, "shipping_fee": 8},
    "B": {"w": 1, "l": 1, "weight": 2, "shipping_fee": 5},
    "C": {"w": 2, "l": 2, "weight": 4, "shipping_fee": 9},
}


def key(state):
    return tuple((vid, tuple(p["position"]) if p["position"] is not None else None,
                  p["orientation"]) for vid, p in sorted(state.items()))


class NeighborhoodTests(unittest.TestCase):
    def test_delta_and_neighborhood_equal_universal_core(self):
        cases = [(VEHICLES, SHIP)]
        cases.append(({"huge": {"w": 9, "l": 9, "weight": 1, "shipping_fee": 1}}, SHIP))
        cases.append(({}, SHIP))
        for vehicles, ship in cases:
            space = Neighborhood(vehicles, ship)
            for seed in range(12):
                with self.subTest(count=len(vehicles), seed=seed):
                    state = random_state(vehicles, ship, random.Random(seed), 0.8)
                    before = deepcopy(state)
                    space.bind(state)
                    expected = {key(n): evaluate(n, vehicles, ship)["score"]
                                for n in iter_neighbors(state, vehicles, ship)}
                    actual = {}
                    for changes in space.changes():
                        candidate = space.apply(changes)
                        candidate_key = key(candidate)
                        self.assertNotIn(candidate_key, actual)
                        actual[candidate_key] = space.score(changes)
                        self.assertEqual(actual[candidate_key],
                                         evaluate(candidate, vehicles, ship)["score"])
                    self.assertEqual(actual, expected)
                    self.assertEqual(state, before)

    def test_json_list_coordinates_work(self):
        state = empty_state(VEHICLES)
        state["A"]["position"] = [0, 0]
        space = Neighborhood(VEHICLES, SHIP)
        space.bind(state)
        for changes in space.changes():
            self.assertEqual(space.score(changes),
                             evaluate(space.apply(changes), VEHICLES, SHIP)["score"])


class HillClimbingTests(unittest.TestCase):
    def test_steepest_chooses_best_and_converges_to_valid_local_optimum(self):
        initial = random_state(VEHICLES, SHIP, random.Random(4), 1)
        before = deepcopy(initial)
        expected_best = max(evaluate(s, VEHICLES, SHIP)["score"]
                            for s in iter_neighbors(initial, VEHICLES, SHIP))
        result = steepest_ascent_hill_climbing(initial, VEHICLES, SHIP, max_iterations=1)
        self.assertEqual(result["terminal_metrics"]["score"], expected_best)
        result = steepest_ascent_hill_climbing(initial, VEHICLES, SHIP)
        self.assertTrue(result["final_metrics"]["is_valid"])
        self.assertEqual(result["stop_reason"], "local_optimum")
        self.assertTrue(all(evaluate(s, VEHICLES, SHIP)["score"] <= result["final_metrics"]["score"]
                            for s in iter_neighbors(result["final_state"], VEHICLES, SHIP)))
        self.assertEqual(initial, before)

    def test_stochastic_can_choose_nonbest_improving_neighbor(self):
        initial = empty_state(VEHICLES)
        observed = set()
        for seed in range(20):
            result = stochastic_hill_climbing(initial, VEHICLES, SHIP,
                                              seed=seed, max_iterations=1)
            observed.add(result["terminal_metrics"]["score"])
        self.assertTrue(all(score > 0 for score in observed))
        self.assertGreater(len(observed), 1)

    def test_sideways_escapes_orientation_plateau(self):
        ship = {"w": 1, "l": 2, "max_capacity": 10}
        vehicles = {"A": {"w": 2, "l": 1, "weight": 1, "shipping_fee": 7}}
        state = empty_state(vehicles)  # H tidak muat; rotasi V di luar punya score 0.
        strict = steepest_ascent_hill_climbing(state, vehicles, ship)
        self.assertEqual(strict["final_metrics"]["score"], 0)
        result = hill_climbing_sideways(state, vehicles, ship, max_sideways=3)
        self.assertEqual(result["final_metrics"]["score"], 7)
        self.assertEqual(result["sideways_moves"], 1)
        self.assertEqual([p["score"] for p in result["history"]], [0, 0, 7])

    def test_sideways_limit_and_zero(self):
        ship = {"w": 1, "l": 1, "max_capacity": 1}
        vehicles = {"A": {"w": 1, "l": 1, "weight": 1, "shipping_fee": 7}}
        state = {"A": {"position": (0, 0), "orientation": "H"}}
        result = hill_climbing_sideways(state, vehicles, ship, max_sideways=3)
        self.assertEqual(result["sideways_moves"], 3)
        self.assertEqual(result["iterations"], 3)
        self.assertEqual(result["stop_reason"], "sideways_limit")
        result = hill_climbing_sideways(empty_state(vehicles), vehicles, ship, max_sideways=0)
        self.assertEqual(result["sideways_moves"], 0)
        self.assertEqual(result["final_metrics"]["score"], 7)

    def test_sideways_budget_is_total_even_after_improvement(self):
        ship = {"w": 1, "l": 3, "max_capacity": 10}
        vehicles = {"A": {"w": 2, "l": 1, "weight": 1, "shipping_fee": 7}}
        # Rotasi di luar -> masuk -> geser setara. Budget tidak reset saat masuk.
        result = hill_climbing_sideways(empty_state(vehicles), vehicles, ship,
                                       max_sideways=2, seed=3)
        self.assertEqual([p["score"] for p in result["history"]], [0, 0, 7, 7])
        self.assertEqual(result["sideways_moves"], 2)
        self.assertEqual(result["stop_reason"], "sideways_limit")

    def test_capped_invalid_terminal_is_reported_with_separate_valid_solution(self):
        state = {vid: {"position": (0, 0), "orientation": "H"} for vid in VEHICLES}
        result = steepest_ascent_hill_climbing(state, VEHICLES, SHIP, max_iterations=0)
        self.assertEqual(result["terminal_state"], state)
        self.assertFalse(result["terminal_metrics"]["is_valid"])
        self.assertTrue(result["final_metrics"]["is_valid"])
        self.assertEqual(result["final_source"], "empty_baseline")
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["stop_reason"], "max_iterations")

    def test_restarts_count_and_best_state_match_same_run(self):
        initial = empty_state(VEHICLES)
        result = random_restart_hill_climbing(initial, VEHICLES, SHIP, max_restarts=2)
        self.assertEqual(result["restarts"], 2)
        self.assertEqual(result["searches"], 3)
        self.assertEqual(len(result["iterations_per_restart"]), 3)
        self.assertEqual(result["iterations"], sum(result["iterations_per_restart"]))
        chosen = result["runs"][result["best_run_index"]]
        self.assertEqual(result["final_state"], chosen["final_state"])
        self.assertEqual(result["best_run_initial_state"], chosen["initial_state"])
        self.assertEqual(result["initial_state"], initial)
        self.assertEqual(result["final_metrics"]["score"],
                         max(run["final_metrics"]["score"] for run in result["runs"]))
        zero = random_restart_hill_climbing(initial, VEHICLES, SHIP, max_restarts=0)
        self.assertEqual(zero["searches"], 1)
        self.assertEqual(zero["restarts"], 0)

    def test_seed_reproducibility_monotonic_trace_and_no_mutation(self):
        initial = random_state(VEHICLES, SHIP, random.Random(99), 1)
        before = deepcopy(initial)
        for algorithm in (steepest_ascent_hill_climbing, stochastic_hill_climbing,
                          hill_climbing_sideways, random_restart_hill_climbing):
            with self.subTest(algorithm=algorithm.__name__):
                first = algorithm(initial, VEHICLES, SHIP, seed=12)
                second = algorithm(initial, VEHICLES, SHIP, seed=12)
                self.assertEqual(first["final_state"], second["final_state"])
                for a, b in zip(first.get("runs", [first]), second.get("runs", [second])):
                    self.assertEqual(a["history"], b["history"])
                    scores = [p["score"] for p in a["history"]]
                    self.assertEqual(scores, sorted(scores))
                    self.assertEqual(len(scores), a["iterations"] + 1)
                    self.assertEqual(a["iterations"], a["improving_moves"] + a["sideways_moves"])
                self.assertTrue(first["final_metrics"]["is_valid"])
                self.assertEqual(initial, before)

    def test_empty_and_invalid_parameters(self):
        for algorithm in (steepest_ascent_hill_climbing, stochastic_hill_climbing,
                          hill_climbing_sideways, random_restart_hill_climbing):
            with self.subTest(algorithm=algorithm.__name__):
                self.assertEqual(algorithm({}, {}, SHIP)["final_metrics"]["score"], 0)
                for limit in (-1, True, 1.5):
                    with self.assertRaises(ValueError):
                        algorithm(empty_state(VEHICLES), VEHICLES, SHIP, max_iterations=limit)
                with self.assertRaises(ValueError):
                    algorithm({}, VEHICLES, SHIP)
        with self.assertRaises(ValueError):
            hill_climbing_sideways({}, {}, SHIP, max_sideways=-1)
        with self.assertRaises(ValueError):
            random_restart_hill_climbing({}, {}, SHIP, max_restarts=True)


if __name__ == "__main__":
    unittest.main()
