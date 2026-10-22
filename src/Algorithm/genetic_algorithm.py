"""Genetic Algorithm dengan kontrak Core yang sama seperti HC dan SA."""

import argparse
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from ..Core.data_loader import load_data, load_fleet
from ..Core.evaluation import evaluate
from ..Core.fleet import is_fleet, placement_ship_id
from ..Core.neighbors import random_neighbor
from ..Core.objectives import normalize_objective, vehicle_values
from ..Core.state import copy_state, empty_state, footprint, random_state

ROOT = Path(__file__).resolve().parents[2]


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_params(population_size, generations, crossover_rate, mutation_rate,
                    tournament_size, elite_count, seed=None, repair=False):
    if not isinstance(repair, bool):
        raise ValueError("repair harus berupa boolean")
    for name, value in (("population_size", population_size), ("generations", generations),
                        ("tournament_size", tournament_size), ("elite_count", elite_count)):
        if not _is_int(value):
            raise ValueError(f"{name} harus bilangan bulat")
    if seed is not None and not _is_int(seed):
        raise ValueError("seed harus bilangan bulat atau None")
    if population_size < 2 or generations < 1:
        raise ValueError("population_size minimal 2 dan generations minimal 1")
    for name, value in (("crossover_rate", crossover_rate), ("mutation_rate", mutation_rate)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} harus angka antara 0 dan 1")
    if not 1 <= tournament_size <= population_size:
        raise ValueError("tournament_size harus antara 1 dan population_size")
    if not 0 <= elite_count <= population_size:
        raise ValueError("elite_count harus antara 0 dan population_size")


def metrics_of(state, vehicles, ship, objective="shipping_fee"):
    result = evaluate(state, vehicles, ship, objective)
    result.pop("grid", None)
    result["fitness"] = result["score"]
    return result


def tournament_select(population, fitnesses, tournament_size, rng):
    if not 1 <= tournament_size <= len(population):
        raise ValueError("tournament_size harus 1..ukuran populasi")
    contenders = rng.sample(range(len(population)), tournament_size)
    return max(contenders, key=lambda index: fitnesses[index])


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


def repair_state(state, vehicles, ship, objective="shipping_fee"):
    """Repair asli GA: rasio value/luas untuk overlap, value/berat untuk kapasitas.

    Setiap kapal diperbaiki mandiri; nilai memakai objective pencarian aktif.
    """
    objective = normalize_objective(objective)
    values = vehicle_values(vehicles, objective)
    result = copy_state(state)
    metrics = evaluate(result, vehicles, ship, objective)
    for sid, row in metrics["per_ship"].items():
        grid = row["grid"]
        weight = row["total_weight"]
        overlap_cells = sum(count > 1 for line in grid for count in line)
        loaded = [vid for vid, placement in result.items()
                  if placement_ship_id(placement, ship) == sid]
        for vid in sorted(loaded, key=lambda item: (
                values[item] / max(1, vehicles[item]["w"] * vehicles[item]["l"]),
                values[item])):
            if not overlap_cells:
                break
            placement = result[vid]
            x, y = placement["position"]
            dx, dy = footprint(vehicles[vid], placement["orientation"])
            cells = [(xx, yy) for yy in range(y, y + dy) for xx in range(x, x + dx)]
            if any(grid[yy][xx] > 1 for xx, yy in cells):
                placement["position"] = None
                if is_fleet(ship):
                    placement["ship_id"] = None
                weight -= vehicles[vid]["weight"]
                for xx, yy in cells:
                    if grid[yy][xx] == 2:
                        overlap_cells -= 1
                    grid[yy][xx] -= 1
        remaining = [vid for vid in loaded if result[vid]["position"] is not None]
        for vid in sorted(remaining, key=lambda item: (
                values[item] / max(1, vehicles[item]["weight"]), values[item])):
            if weight <= row["max_capacity"]:
                break
            result[vid]["position"] = None
            if is_fleet(ship):
                result[vid]["ship_id"] = None
            weight -= vehicles[vid]["weight"]
    return result


def run_ga(ship, vehicles, population_size=100, generations=300, crossover_rate=0.8,
           mutation_rate=0.5, tournament_size=3, elite_count=2, seed=42,
           repair=False, objective="shipping_fee", load_probability=0.5):
    """Jalankan GA; urutan argumen kompatibel dengan branch asal."""
    validate_params(population_size, generations, crossover_rate, mutation_rate,
                    tournament_size, elite_count, seed, repair)
    objective = normalize_objective(objective)
    if seed is None:
        seed = random.SystemRandom().randrange(2 ** 31)
    rng = random.Random(seed)
    started = perf_counter()
    population = [random_state(vehicles, ship, rng, load_probability) for _ in range(population_size)]
    initial_population = [copy_state(state) for state in population]
    fitnesses = [metrics_of(state, vehicles, ship, objective)["fitness"] for state in population]
    initial_index = max(range(population_size), key=lambda index: fitnesses[index])
    initial_best_state = copy_state(population[initial_index])
    best_fitness = {"fitness": None, "state": None, "generation": None}
    best_valid = {"score": None, "state": None, "generation": None}

    def observe(pop, fits, generation):
        valid_count = 0
        for state, fitness in zip(pop, fits):
            if best_fitness["fitness"] is None or fitness > best_fitness["fitness"]:
                best_fitness.update(fitness=fitness, state=copy_state(state), generation=generation)
            metrics = evaluate(state, vehicles, ship, objective)
            if metrics["is_valid"]:
                valid_count += 1
                if best_valid["score"] is None or metrics["score"] > best_valid["score"]:
                    best_valid.update(score=metrics["score"], state=copy_state(state),
                                      generation=generation)
        if best_valid["state"] is None:
            best_score = best_revenue = 0
        else:
            selected = evaluate(best_valid["state"], vehicles, ship, objective)
            best_score, best_revenue = selected["score"], selected["revenue"]
        return {"generation": generation, "candidate_peak": max(fits),
                "mean_fitness": sum(fits) / len(fits), "valid_count": valid_count,
                "best_valid_objective_so_far": best_score,
                "best_valid_revenue_so_far": best_revenue}

    history = [observe(population, fitnesses, 0)]
    for generation in range(1, generations + 1):
        order = sorted(range(population_size), key=lambda index: fitnesses[index], reverse=True)
        next_population = [copy_state(population[index]) for index in order[:elite_count]]
        children = []
        while len(children) < population_size - elite_count:
            p1 = population[tournament_select(population, fitnesses, tournament_size, rng)]
            p2 = population[tournament_select(population, fitnesses, tournament_size, rng)]
            pair = (uniform_crossover(p1, p2, rng) if rng.random() < crossover_rate
                    else (copy_state(p1), copy_state(p2)))
            for child in pair:
                if len(children) >= population_size - elite_count:
                    break
                if rng.random() < mutation_rate:
                    child = mutate(child, vehicles, ship, rng)
                if repair:
                    child = repair_state(child, vehicles, ship, objective)
                children.append(child)
        population = next_population + children
        fitnesses = [metrics_of(state, vehicles, ship, objective)["fitness"]
                     for state in population]
        history.append(observe(population, fitnesses, generation))

    fallback = best_valid["state"] is None
    final_state = empty_state(vehicles, ship) if fallback else best_valid["state"]
    final_metrics = evaluate(final_state, vehicles, ship, objective)
    best_fitness_state = best_fitness["state"]
    params = {"population_size": population_size, "generations": generations,
              "crossover_rate": crossover_rate, "mutation_rate": mutation_rate,
              "tournament_size": tournament_size, "elite_count": elite_count,
              "seed": seed, "repair": repair, "objective": objective,
              "load_probability": load_probability}
    return {
        "algorithm": "genetic_algorithm", "parameters": params, "params": params,
        "initial_state": initial_best_state,
        "initial_metrics": evaluate(initial_best_state, vehicles, ship, objective),
        "final_state": copy_state(final_state), "final_metrics": final_metrics,
        "final_source": "empty_baseline" if fallback else "population",
        "iterations": generations, "generations_run": generations,
        "evaluated_neighbors": population_size * (generations + 1),
        "stop_reason": "max_generations", "duration_seconds": perf_counter() - started,
        "history": history, "initial_population": initial_population,
        "final_population": [copy_state(state) for state in population],
        "initial_best_state": initial_best_state,
        "initial_best_metrics": metrics_of(initial_best_state, vehicles, ship, objective),
        "best_fitness_state": copy_state(best_fitness_state),
        "best_fitness_metrics": metrics_of(best_fitness_state, vehicles, ship, objective),
        "best_fitness_generation": best_fitness["generation"],
        "best_valid_state": copy_state(final_state), "best_valid_metrics": metrics_of(final_state, vehicles, ship, objective),
        "best_valid_from_fallback": fallback,
        "best_valid_generation": best_valid["generation"],
        "ship": deepcopy(ship), "vehicles": deepcopy(vehicles),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data/data.json"))
    parser.add_argument("--fleet")
    parser.add_argument("--objective", choices=("shipping_fee", "eta_priority"),
                        default="shipping_fee")
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--crossover-rate", type=float, default=0.8)
    parser.add_argument("--mutation-rate", type=float, default=0.5)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--output-directory", default=str(ROOT / "output/genetic_algorithm"))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from ..main import main as run_experiment
    command = ["--algorithm", "genetic_algorithm", "--data", args.data,
               "--seeds", str(args.seed), "--objective", args.objective,
               "--output", args.output_directory]
    for name in ("population_size", "generations", "crossover_rate", "mutation_rate",
                 "tournament_size", "elite_count"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    if args.fleet:
        command.extend(["--fleet", args.fleet])
    if args.repair:
        command.append("--repair")
    run_experiment(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
