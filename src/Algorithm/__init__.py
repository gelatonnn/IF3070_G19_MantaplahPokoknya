"""Algoritma pencarian; setiap algoritma memakai kontrak src.Core."""

from .hill_climbing import (
    hill_climbing_sideways,
    random_restart_hill_climbing,
    steepest_ascent_hill_climbing,
    stochastic_hill_climbing,
)
from .simulated_annealing import simulated_annealing


__all__ = [
    "steepest_ascent_hill_climbing",
    "stochastic_hill_climbing",
    "hill_climbing_sideways",
    "random_restart_hill_climbing",
    "simulated_annealing",
    "run_ga",
]


def __getattr__(name):
    # Lazy import supaya python -m src.Algorithm.genetic_algorithm tidak
    # memuat modul GA dua kali melalui package initializer.
    if name == "run_ga":
        from .genetic_algorithm import run_ga
        return run_ga
    raise AttributeError(name)
