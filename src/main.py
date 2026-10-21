"""CLI eksperimen HC dan SA. Jalankan dari repo: python -m src.main --help."""

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
)
from .Algorithm.simulated_annealing import simulated_annealing
from .Core.data_loader import load_data
from .Core.state import random_state


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "steepest": steepest_ascent_hill_climbing,
    "stochastic": stochastic_hill_climbing,
    "sideways": hill_climbing_sideways,
    "random_restart": random_restart_hill_climbing,
    "simulated_annealing": simulated_annealing,
}


def _nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("Harus >= 0")
    return value


#cek argumen sa dari terminal biar ga jalan pake nilai aneh
def _positive_float(text):
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError("harus > 0")
    return value


def _alpha(text):
    value = float(text)
    if not 0 < value < 1:
        raise argparse.ArgumentTypeError("harus di antara 0 dan 1")
    return value


def _positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("harus >= 1")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/data.json")
    parser.add_argument("--algorithm", choices=["all", *ALGORITHMS], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-iterations", type=_nonnegative, default=None,
                        help="Batas langkah per pencarian. Default: hingga berhenti alami.")
    parser.add_argument("--max-sideways", type=_nonnegative, default=20)
    parser.add_argument("--max-restarts", type=_nonnegative, default=3,
                        help="Restart tambahan setelah pencarian awal.")
    parser.add_argument("--load-probability", type=float, default=0.5)
    #parameter khusus simulated annealing
    parser.add_argument("--t0", type=_positive_float, default=100.0,
                        help="sa: suhu awal")
    parser.add_argument("--alpha", type=_alpha, default=0.9999,
                        help="sa: faktor pendinginan, T = T * alpha")
    parser.add_argument("--t-min", type=_positive_float, default=0.1,
                        help="sa: berhenti kalo T <= t-min")
    parser.add_argument("--stuck-window", type=_positive_int, default=1000,
                        help="sa: berapa iterasi tanpa perbaikan baru diitung 1x stuck")
    #bonus 1: objective alternatif, cuma buat sa
    parser.add_argument("--objective", choices=["fee", "eta"], default="fee",
                        help="sa: fee = objective biasa, eta = fee dikali bobot urgensi (bonus 1)")
    parser.add_argument("--output", type=Path, default=ROOT / "output/hill_climbing")
    parser.add_argument("--no-plots", action="store_true", help="Hanya simpan JSON dan CSV.")
    args = parser.parse_args(argv)
    if not 0 <= args.load_probability <= 1:
        parser.error("--load-probability harus antara 0 dan 1")
    #hc pake delta score dari evaluate() biasa, jadi objective eta cuma bisa di sa
    if args.objective == "eta" and args.algorithm != "simulated_annealing":
        parser.error("--objective eta cuma bisa dipake bareng --algorithm simulated_annealing")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds harus unik")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output sudah berisi file; gunakan folder baru untuk menjaga hasil lama")
    try:
        ship, vehicles = load_data(args.data)
        if not args.no_plots:
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            import matplotlib  # noqa: F401; periksa dependency sebelum pencarian panjang.
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    names = list(ALGORITHMS) if args.algorithm == "all" else [args.algorithm]
    rows = []
    for name in names:
        for seed in args.seeds:
            initial = random_state(vehicles, ship, random.Random(seed), args.load_probability)
            kwargs = {"seed": seed, "max_iterations": args.max_iterations}
            if name == "sideways":
                kwargs["max_sideways"] = args.max_sideways
            elif name == "random_restart":
                kwargs.update(max_restarts=args.max_restarts,
                              load_probability=args.load_probability)
            elif name == "simulated_annealing":
                #sa ga pake max_iterations, berhentinya dari suhu
                kwargs = {"seed": seed, "t0": args.t0, "alpha": args.alpha,
                          "t_min": args.t_min, "stuck_window": args.stuck_window,
                          "objective": args.objective}
            print(f"Mulai {name}, seed={seed}, kendaraan={len(vehicles)}", flush=True)
            result = ALGORITHMS[name](initial, vehicles, ship, **kwargs)
            #trace sa isinya data tiap iterasi (bisa 69rb titik), cuma dipake buat plot
            #jadi dikeluarin dulu biar result.json ga kegedean
            trace = result.pop("trace", None)
            result["experiment"] = {
                "dataset": args.data.name, "dataset_sha256": digest,
                "initial_seed": seed, "search_seed": seed,
                "load_probability": args.load_probability,
                "python_version": sys.version.split()[0],
            }
            result["ship"] = ship
            folder = args.output / name / f"seed_{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "result.json").write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8",
            )
            if not args.no_plots:
                if name == "simulated_annealing":
                    from .Visualization.simulated_annealing import save_plots as save_sa_plots
                    save_sa_plots(result, trace, vehicles, ship, folder)
                else:
                    from .Visualization.hill_climbing import save_plots
                    save_plots(result, vehicles, ship, folder)
            metrics = result["final_metrics"]
            rows.append({
                "algorithm": name, "seed": seed,
                "objective": result["parameters"].get("objective", "fee"),
                "score": metrics["score"],
                #nilai berbobot eta, kosong kalo objective-nya fee biasa
                "weighted_revenue": metrics.get("weighted_revenue", ""),
                "revenue": metrics["revenue"], "is_valid": metrics["is_valid"],
                "iterations": result["iterations"],
                "restarts": result.get("restarts", 0),
                "sideways_moves": result.get("sideways_moves", 0),
                "stuck_count": result.get("stuck_count", 0),
                "accepted_worse_moves": result.get("accepted_worse_moves", 0),
                "duration_seconds": result["duration_seconds"],
                "stop_reason": result["stop_reason"],
                "final_source": result["final_source"],
                "evaluated_neighbors": result.get("evaluated_neighbors", result["iterations"]),
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
