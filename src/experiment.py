"""Shared experiment runner for the CLI and a future frontend."""

import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Callable

from .Algorithm import (
    hill_climbing_sideways, random_restart_hill_climbing,
    steepest_ascent_hill_climbing, stochastic_hill_climbing,
    simulated_annealing, run_ga,
)
from .Core.data_loader import load_data, load_fleet
from .Core.fleet import is_fleet, ships_by_id
from .Core.objectives import OBJECTIVES, normalize_objective, vehicle_values
from .Core.state import random_state
from .Algorithm.genetic_algorithm import validate_params as validate_ga_params


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "steepest": steepest_ascent_hill_climbing,
    "stochastic": stochastic_hill_climbing,
    "sideways": hill_climbing_sideways,
    "random_restart": random_restart_hill_climbing,
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": run_ga,
}


@dataclass
class ExperimentConfig:
    data: Path = ROOT / "data/data.json"
    fleet: Path | None = None
    objective: str = "shipping_fee"
    algorithm: str = "all"
    selected_algorithms: list[str] | None = None
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    max_iterations: int | None = None
    max_sideways: int = 20
    max_restarts: int = 3
    load_probability: float = 0.5
    t0: float = 100.0
    alpha: float = 0.9999
    t_min: float = 0.1
    stuck_window: int = 1000
    history_step: int = 10
    population_size: int = 100
    generations: int = 300
    crossover_rate: float = 0.8
    mutation_rate: float = 0.5
    tournament_size: int = 3
    elite_count: int = 2
    repair: bool = False
    output: Path = ROOT / "output/experiments"
    no_plots: bool = False


def validate_config(config: ExperimentConfig) -> None:
    if config.algorithm not in {"all", *ALGORITHMS}:
        raise ValueError(f"Algoritma tidak dikenal: {config.algorithm}")
    if config.selected_algorithms is not None:
        names = config.selected_algorithms
        if (config.algorithm != "all" or not names or len(set(names)) != len(names)
                or any(name not in ALGORITHMS for name in names)):
            raise ValueError("selected_algorithms harus berisi algoritma unik yang dikenal pada mode all")
    if config.objective not in OBJECTIVES:
        raise ValueError(f"Objective tidak dikenal: {config.objective}")
    if not config.seeds or any(type(seed) is not int for seed in config.seeds):
        raise ValueError("--seeds harus berisi sedikitnya satu integer")
    if len(set(config.seeds)) != len(config.seeds):
        raise ValueError("--seeds harus unik")
    validate_ga_params(config.population_size, config.generations,
                       config.crossover_rate, config.mutation_rate,
                       config.tournament_size, config.elite_count,
                       repair=config.repair)
    for name in ("max_sideways", "max_restarts"):
        value = getattr(config, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} harus integer >= 0")
    if config.max_iterations is not None and (
        type(config.max_iterations) is not int or config.max_iterations < 0
    ):
        raise ValueError("max_iterations harus integer >= 0 atau None")
    for name in ("stuck_window", "history_step"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} harus integer >= 1")
    for name in ("load_probability",):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} harus angka antara 0 dan 1")
    for name in ("t0", "t_min"):
        value = getattr(config, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} harus angka > 0")
    if type(config.alpha) not in (int, float) or not math.isfinite(config.alpha) or not 0 < config.alpha < 1:
        raise ValueError("alpha harus angka di antara 0 dan 1")
    if config.output.exists() and (not config.output.is_dir() or any(config.output.iterdir())):
        raise ValueError("--output sudah berisi file; gunakan folder baru untuk menjaga hasil lama")


def run_experiments(
    config: ExperimentConfig,
    on_event: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run configured algorithms and persist their JSON, CSV, and optional plots.

    The callback receives a short message before and after each algorithm/seed run.
    Results are also returned as summary rows for callers that need an immediate view.
    """
    validate_config(config)
    data = Path(config.data)
    fleet = Path(config.fleet) if config.fleet is not None else None
    output = Path(config.output)
    objective = normalize_objective(config.objective)
    ship, vehicles = load_data(data)
    if fleet is not None:
        ship = load_fleet(fleet)
    vehicle_values(vehicles, objective)
    if not config.no_plots:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        import matplotlib  # noqa: F401; check dependency before long searches.

    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    fleet_digest = hashlib.sha256(fleet.read_bytes()).hexdigest() if fleet else None
    ship_count = len(ships_by_id(ship))
    output.mkdir(parents=True, exist_ok=True)
    names = (list(config.selected_algorithms) if config.selected_algorithms is not None
             else list(ALGORITHMS) if config.algorithm == "all" else [config.algorithm])
    rows = []
    announce = on_event or (lambda _message: None)
    for name in names:
        for seed in config.seeds:
            initial = random_state(vehicles, ship, random.Random(seed), config.load_probability)
            kwargs = {"seed": seed, "max_iterations": config.max_iterations,
                      "objective": objective}
            if name == "sideways":
                kwargs["max_sideways"] = config.max_sideways
            elif name == "random_restart":
                kwargs.update(max_restarts=config.max_restarts,
                              load_probability=config.load_probability)
            elif name == "simulated_annealing":
                kwargs.update(t0=config.t0, alpha=config.alpha, t_min=config.t_min,
                              stuck_window=config.stuck_window, history_step=config.history_step)
            elif name == "genetic_algorithm":
                kwargs = {"population_size": config.population_size,
                          "generations": config.generations,
                          "crossover_rate": config.crossover_rate,
                          "mutation_rate": config.mutation_rate,
                          "tournament_size": config.tournament_size,
                          "elite_count": config.elite_count, "seed": seed,
                          "repair": config.repair, "objective": objective,
                          "load_probability": config.load_probability}
            announce(f"Mulai {name}, seed={seed}, kendaraan={len(vehicles)}, "
                     f"kapal={ship_count}, objective={objective}")
            result = (run_ga(ship, vehicles, **kwargs) if name == "genetic_algorithm"
                      else ALGORITHMS[name](initial, vehicles, ship, **kwargs))
            trace = result.pop("trace", None)
            result["experiment"] = {
                "dataset": data.name, "dataset_sha256": digest,
                "initial_seed": seed, "search_seed": seed,
                "load_probability": config.load_probability,
                "python_version": sys.version.split()[0],
                "objective": objective, "ship_count": ship_count,
                "fleet_dataset": fleet.name if fleet else None,
                "fleet_sha256": fleet_digest,
            }
            if is_fleet(ship):
                result.pop("ship", None)
                result["ships"] = ship["ships"]
            else:
                result["ship"] = ship
            folder = output / name / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            omitted = {"initial_population", "final_population", "vehicles"}
            json_result = {key: value for key, value in result.items() if key not in omitted}
            (folder / "result.json").write_text(
                json.dumps(json_result, indent=2) + "\n", encoding="utf-8")
            if not config.no_plots:
                if name == "simulated_annealing":
                    from .Visualization.simulated_annealing import save_plots
                    save_plots(result, trace, vehicles, ship, folder)
                elif name == "genetic_algorithm":
                    from .Visualization.plots import save_plots
                    save_plots(result, vehicles, ship, folder)
                else:
                    from .Visualization.hill_climbing import save_plots
                    save_plots(result, vehicles, ship, folder)
            metrics = result["final_metrics"]
            rows.append({
                "algorithm": name, "seed": seed, "score": metrics["score"],
                "objective": objective, "objective_value": metrics["objective_value"],
                "ship_count": ship_count,
                "revenue": metrics["revenue"], "is_valid": metrics["is_valid"],
                "iterations": result["iterations"],
                "restarts": result.get("restarts", 0),
                "sideways_moves": result.get("sideways_moves", 0),
                "stuck_count": result.get("stuck_count", 0),
                "accepted_worse_moves": result.get("accepted_worse_moves", 0),
                "duration_seconds": result["duration_seconds"],
                "stop_reason": result["stop_reason"],
                "final_source": result["final_source"],
                "evaluated_neighbors": result["evaluated_neighbors"],
                "iterations_per_restart": json.dumps(result.get("iterations_per_restart", [])),
            })
            # Keep partial results available if a later run fails.
            with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            announce(f"Selesai {name} seed={seed}: score={metrics['score']}, "
                     f"valid={metrics['is_valid']}, iterasi={result['iterations']}, "
                     f"durasi={result['duration_seconds']:.2f}s")
    return rows
