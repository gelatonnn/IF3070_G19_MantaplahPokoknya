"""Simulated Annealing utk penempatan kendaraannya"""

import math
import random

from ..Core.evaluation import evaluate
from ..Core.neighbors import random_neighbor
from ..Core.state import copy_state


def simulated_annealing(initial_state, vehicles, ship, *, seed=0,
                        t0=100.0, alpha=0.9999, t_min=0.1):
    rng = random.Random(seed)

    current = copy_state(initial_state)
    current_metrics = evaluate(current, vehicles, ship)

    temperature = t0
    iteration = 0

    while temperature > t_min:
        #ambil 1 neighbor yang random
        neighbor = random_neighbor(current, vehicles, ship, rng)
        if neighbor is None:  # dataset tanpa kendaraan
            break
        neighbor_metrics = evaluate(neighbor, vehicles, ship)

        #hitung delta E = score tetangga - score sekarang
        delta = neighbor_metrics["score"] - current_metrics["score"]

        #lebih baik / sama -> terima, lebih buruk -> terima dengan peluang e^(delta/T)
        if delta >= 0:
            accept = True
        else:
            accept = rng.random() < math.exp(delta / temperature)

        if accept:
            current, current_metrics = neighbor, neighbor_metrics

        #turunin suhu
        temperature = temperature * alpha
        iteration += 1

    return {
        "final_state": current,
        "final_metrics": current_metrics,
        "iterations": iteration,
    }
