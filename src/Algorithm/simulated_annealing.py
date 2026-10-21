"""Simulated Annealing utk penempatan kendaraannya"""

import math
import random
from time import perf_counter

from ..Core.evaluation import evaluate
from ..Core.neighbors import random_neighbor
from ..Core.state import copy_state, empty_state


def simulated_annealing(initial_state, vehicles, ship, *, seed=0,
                        t0=100.0, alpha=0.9999, t_min=0.1):
    rng = random.Random(seed)
    started = perf_counter()

    current = copy_state(initial_state)
    current_metrics = evaluate(current, vehicles, ship)
    initial_metrics = current_metrics

    #simpen state valid terbaik yg pernah dilewatin, soalnya sa bisa aja
    #nerima langkah jelek terus ga balik lagi ke state bagus tadi
    #defaultnya kapal kosong (score 0, pasti valid) biar slalu ada jawaban valid
    best_valid = empty_state(vehicles)
    best_valid_metrics = evaluate(best_valid, vehicles, ship)
    best_valid_source = "empty_baseline"
    if current_metrics["is_valid"]:
        best_valid, best_valid_metrics = copy_state(current), current_metrics
        best_valid_source = "visited"

    temperature = t0
    iteration = 0
    stop_reason = "t_min"

    while temperature > t_min:
        #ambil 1 neighbor yang random
        neighbor = random_neighbor(current, vehicles, ship, rng)
        if neighbor is None:  #dataset tanpa kendaraan
            stop_reason = "no_neighbor"
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

        #kalo state skrg valid dan lebih bagus dari best, update bestnya
        #pake copy biar ga ikut berubah pas current diganti
        if current_metrics["is_valid"] and current_metrics["score"] > best_valid_metrics["score"]:
            best_valid, best_valid_metrics = copy_state(current), current_metrics
            best_valid_source = "visited"

        #turunin suhu
        temperature = temperature * alpha
        iteration += 1

    #format hasil disamain kyk hill climbing biar main.py sama plot bisa dipake bareng
    #final = valid terbaik, terminal = state pas sa berhenti
    return {
        "algorithm": "simulated_annealing",
        "initial_state": copy_state(initial_state),
        "initial_metrics": initial_metrics,
        "final_state": copy_state(best_valid),
        "final_metrics": best_valid_metrics,
        "terminal_state": copy_state(current),
        "terminal_metrics": current_metrics,
        "final_source": best_valid_source,
        "iterations": iteration,
        "final_temperature": temperature,
        "stop_reason": stop_reason,
        "duration_seconds": perf_counter() - started,
        "parameters": {"seed": seed, "t0": t0, "alpha": alpha, "t_min": t_min},
    }
