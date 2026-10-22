"""CLI eksperimen HC, SA, dan GA. Jalankan: python -m src.main --help."""

import argparse
from pathlib import Path

from .Core.objectives import OBJECTIVES
from .experiment import ALGORITHMS, ExperimentConfig, ROOT, run_experiments


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
    config = ExperimentConfig(**vars(args))
    try:
        run_experiments(config, on_event=lambda message: print(message, flush=True))
    except (OSError, ValueError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
