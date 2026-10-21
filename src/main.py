"""CLI eksperimen HC. Jalankan dari repo: python -m src.main --help."""

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
from .Core.data_loader import load_data
from .Core.state import random_state


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = {
    "steepest": steepest_ascent_hill_climbing,
    "stochastic": stochastic_hill_climbing,
    "sideways": hill_climbing_sideways,
    "random_restart": random_restart_hill_climbing,
}


def _nonnegative(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("Harus >= 0")
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
    parser.add_argument("--output", type=Path, default=ROOT / "output/hill_climbing")
    parser.add_argument("--no-plots", action="store_true", help="Hanya simpan JSON dan CSV.")
    args = parser.parse_args(argv)
    if not 0 <= args.load_probability <= 1:
        parser.error("--load-probability harus antara 0 dan 1")
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
            print(f"Mulai {name}, seed={seed}, kendaraan={len(vehicles)}", flush=True)
            result = ALGORITHMS[name](initial, vehicles, ship, **kwargs)
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
                from .Visualization.hill_climbing import save_plots
                save_plots(result, vehicles, ship, folder)
            metrics = result["final_metrics"]
            rows.append({
                "algorithm": name, "seed": seed, "score": metrics["score"],
                "revenue": metrics["revenue"], "is_valid": metrics["is_valid"],
                "iterations": result["iterations"],
                "restarts": result.get("restarts", 0),
                "sideways_moves": result.get("sideways_moves", 0),
                "duration_seconds": result["duration_seconds"],
                "stop_reason": result["stop_reason"],
                "final_source": result["final_source"],
                "evaluated_neighbors": result["evaluated_neighbors"],
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
