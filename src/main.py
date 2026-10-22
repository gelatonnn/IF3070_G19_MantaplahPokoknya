"""CLI eksperimen HC, SA, dan GA. Jalankan: python -m src.main --help."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys

from .Algorithm import (
    hill_climbing_sideways, random_restart_hill_climbing,
    steepest_ascent_hill_climbing, stochastic_hill_climbing,
    simulated_annealing, run_ga,
)
from .Core.data_loader import load_data, load_fleet
from .Core.fleet import is_fleet, ships_by_id
from .Core.objectives import OBJECTIVES, normalize_objective, vehicle_values
from .Core.state import random_state


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "steepest": steepest_ascent_hill_climbing,
    "stochastic": stochastic_hill_climbing,
    "sideways": hill_climbing_sideways,
    "random_restart": random_restart_hill_climbing,
    "simulated_annealing": simulated_annealing,
    "genetic_algorithm": run_ga,
}


def _nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("Harus >= 0")
    return value


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Harus >= 1")
    return value


def _positive_float(text):
    value = float(text)
    if not 0 < value < float("inf"):
        raise argparse.ArgumentTypeError("Harus > 0")
    return value


def _probability(text):
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("Harus antara 0 dan 1")
    return value


def _alpha(text):
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("Harus di antara 0 dan 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/data.json")
    parser.add_argument("--fleet", type=Path,
                        help="JSON ships terpisah, menggantikan konfigurasi kapal dari --data.")
    parser.add_argument("--objective", choices=OBJECTIVES,
                        default="shipping_fee",
                        help="shipping_fee (wajib) atau eta_priority (bonus).")
    parser.add_argument("--algorithm", choices=["all", *ALGORITHMS], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--max-iterations", type=_nonnegative, default=None,
                        help="Batas iterasi HC/SA. GA memakai --generations. Default HC/SA: berhenti alami.")
    parser.add_argument("--max-sideways", type=_nonnegative, default=20)
    parser.add_argument("--max-restarts", type=_nonnegative, default=3,
                        help="Restart tambahan setelah pencarian awal.")
    parser.add_argument("--load-probability", type=_probability, default=0.5)
    parser.add_argument("--t0", type=_positive_float, default=100.0)
    parser.add_argument("--alpha", type=_alpha, default=0.9999)
    parser.add_argument("--t-min", type=_positive_float, default=0.1)
    parser.add_argument("--stuck-window", type=_positive_int, default=1000)
    parser.add_argument("--history-step", type=_positive_int, default=10)
    parser.add_argument("--population-size", type=_positive_int, default=100)
    parser.add_argument("--generations", type=_positive_int, default=300)
    parser.add_argument("--crossover-rate", type=_probability, default=0.8)
    parser.add_argument("--mutation-rate", type=_probability, default=0.5)
    parser.add_argument("--tournament-size", type=_positive_int, default=3)
    parser.add_argument("--elite-count", type=_nonnegative, default=2)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "output/experiments")
    parser.add_argument("--no-plots", action="store_true", help="Hanya simpan JSON dan CSV.")
    args = parser.parse_args(argv)
    args.objective = normalize_objective(args.objective)
    if args.population_size < 2:
        parser.error("--population-size minimal 2")
    if args.tournament_size > args.population_size:
        parser.error("--tournament-size tidak boleh melebihi --population-size")
    if args.elite_count > args.population_size:
        parser.error("--elite-count tidak boleh melebihi --population-size")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds harus unik")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output sudah berisi file; gunakan folder baru untuk menjaga hasil lama")
    try:
        ship, vehicles = load_data(args.data)
        if args.fleet is not None:
            ship = load_fleet(args.fleet)
        vehicle_values(vehicles, args.objective)
        if not args.no_plots:
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            import matplotlib  # noqa: F401; periksa dependency sebelum pencarian panjang.
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    fleet_digest = hashlib.sha256(args.fleet.read_bytes()).hexdigest() if args.fleet else None
    ship_count = len(ships_by_id(ship))
    args.output.mkdir(parents=True, exist_ok=True)
    names = list(ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
    rows = []
    for name in names:
        for seed in args.seeds:
            initial = random_state(vehicles, ship, random.Random(seed), args.load_probability)
            kwargs = {"seed": seed, "max_iterations": args.max_iterations,
                      "objective": args.objective}
            if name == "sideways":
                kwargs["max_sideways"] = args.max_sideways
            elif name == "random_restart":
                kwargs.update(max_restarts=args.max_restarts,
                              load_probability=args.load_probability)
            elif name == "simulated_annealing":
                kwargs.update(t0=args.t0, alpha=args.alpha, t_min=args.t_min,
                              stuck_window=args.stuck_window, history_step=args.history_step)
            elif name == "genetic_algorithm":
                kwargs = {"population_size": args.population_size,
                          "generations": args.generations,
                          "crossover_rate": args.crossover_rate,
                          "mutation_rate": args.mutation_rate,
                          "tournament_size": args.tournament_size,
                          "elite_count": args.elite_count, "seed": seed,
                          "repair": args.repair, "objective": args.objective,
                          "load_probability": args.load_probability}
            print(f"Mulai {name}, seed={seed}, kendaraan={len(vehicles)}, "
                  f"kapal={ship_count}, objective={args.objective}", flush=True)
            result = (run_ga(ship, vehicles, **kwargs) if name == "genetic_algorithm"
                      else ALGORITHMS[name](initial, vehicles, ship, **kwargs))
            trace = result.pop("trace", None)
            result["experiment"] = {
                "dataset": args.data.name, "dataset_sha256": digest,
                "initial_seed": seed, "search_seed": seed,
                "load_probability": args.load_probability,
                "python_version": sys.version.split()[0],
                "objective": args.objective, "ship_count": ship_count,
                "fleet_dataset": args.fleet.name if args.fleet else None,
                "fleet_sha256": fleet_digest,
            }
            if is_fleet(ship):
                result.pop("ship", None)
                result["ships"] = ship["ships"]
            else:
                result["ship"] = ship
            folder = args.output / name / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            omitted = {"initial_population", "final_population", "vehicles"}
            json_result = {key: value for key, value in result.items() if key not in omitted}
            (folder / "result.json").write_text(
                json.dumps(json_result, indent=2) + "\n", encoding="utf-8")
            if not args.no_plots:
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
                "objective": args.objective, "objective_value": metrics["objective_value"],
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
            # Simpan hasil parsial setiap run selesai.
            with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Selesai {name} seed={seed}: score={metrics['score']}, "
                  f"valid={metrics['is_valid']}, iterasi={result['iterations']}, "
                  f"durasi={result['duration_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
