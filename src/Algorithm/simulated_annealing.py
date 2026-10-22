"""Simulated Annealing untuk penempatan kendaraan pada satu atau banyak kapal."""

import math
import random
from time import perf_counter

from ..Core.evaluation import evaluate
from ..Core.neighbors import random_neighbor
from ..Core.objectives import normalize_objective
from ..Core.state import copy_state, empty_state


def _point(iteration, temperature, metrics):
    return {
        "iteration": iteration,
        "temperature": temperature,
        **{name: metrics[name] for name in (
            "score", "revenue", "objective_value", "overlap", "excess_weight", "is_valid",
        )},
    }


def simulated_annealing(initial_state, vehicles, ship, *, seed=0,
                        t0=100.0, alpha=0.9999, t_min=0.1,
                        stuck_window=1000, history_step=10,
                        max_iterations=None, objective="shipping_fee"):
    """Jalankan SA dan kembalikan state valid terbaik yang pernah dikunjungi."""
    objective = normalize_objective(objective)
    if not isinstance(t0, (int, float)) or isinstance(t0, bool) or not math.isfinite(t0) or t0 <= 0:
        raise ValueError("t0 harus angka > 0")
    if not isinstance(t_min, (int, float)) or isinstance(t_min, bool) or not math.isfinite(t_min) or t_min <= 0:
        raise ValueError("t_min harus angka > 0")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0 < alpha < 1:
        raise ValueError("alpha harus berada di antara 0 dan 1")
    for name, value, minimum in (("stuck_window", stuck_window, 1),
                                 ("history_step", history_step, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} harus integer >= {minimum}")
    if max_iterations is not None and (type(max_iterations) is not int or max_iterations < 0):
        raise ValueError("max_iterations harus integer >= 0 atau None")

    rng = random.Random(seed)
    started = perf_counter()
    current = copy_state(initial_state)
    current_metrics = evaluate(current, vehicles, ship, objective)
    initial_metrics = current_metrics

    best_valid = empty_state(vehicles, ship)
    best_valid_metrics = evaluate(best_valid, vehicles, ship, objective)
    best_valid_source = "empty_baseline"
    if current_metrics["is_valid"]:
        best_valid, best_valid_metrics = copy_state(current), current_metrics
        best_valid_source = "visited"

    best_score = current_metrics["score"]
    since_improve = stuck_count = 0
    accepted_better = accepted_equal = accepted_worse = rejected = 0
    scores = [current_metrics["score"]]
    temperatures = [float(t0)]
    acceptance = []
    history = [_point(0, float(t0), current_metrics)]
    temperature = float(t0)
    iteration = 0
    stop_reason = "t_min"

    while temperature > t_min:
        if max_iterations is not None and iteration >= max_iterations:
            stop_reason = "max_iterations"
            break
        neighbor = random_neighbor(current, vehicles, ship, rng)
        if neighbor is None:
            stop_reason = "no_neighbor"
            break
        neighbor_metrics = evaluate(neighbor, vehicles, ship, objective)
        delta = neighbor_metrics["score"] - current_metrics["score"]
        if delta > 0:
            accept = True
            accepted_better += 1
        elif delta == 0:
            accept = True
            accepted_equal += 1
        else:
            probability = math.exp(delta / temperature)
            acceptance.append((iteration + 1, probability))
            accept = rng.random() < probability
            if accept:
                accepted_worse += 1
            else:
                rejected += 1
        if accept:
            current, current_metrics = neighbor, neighbor_metrics
        if (current_metrics["is_valid"]
                and current_metrics["score"] > best_valid_metrics["score"]):
            best_valid, best_valid_metrics = copy_state(current), current_metrics
            best_valid_source = "visited"
        temperature *= alpha
        iteration += 1
        if current_metrics["score"] > best_score:
            best_score = current_metrics["score"]
            since_improve = 0
        else:
            since_improve += 1
            if since_improve == stuck_window:
                stuck_count += 1
        scores.append(current_metrics["score"])
        temperatures.append(temperature)
        if iteration % history_step == 0:
            history.append(_point(iteration, temperature, current_metrics))
    if history[-1]["iteration"] != iteration:
        history.append(_point(iteration, temperature, current_metrics))

    return {
        "algorithm": "simulated_annealing",
        "initial_state": copy_state(initial_state),
        "initial_metrics": initial_metrics,
        "final_state": copy_state(best_valid),
        "final_metrics": best_valid_metrics,
        "terminal_state": copy_state(current),
        "terminal_metrics": current_metrics,
        "final_source": best_valid_source,
        "history": history,
        "iterations": iteration,
        "evaluated_neighbors": iteration,
        "final_temperature": temperature,
        "accepted_better_moves": accepted_better,
        "accepted_equal_moves": accepted_equal,
        "accepted_worse_moves": accepted_worse,
        "rejected_moves": rejected,
        "stuck_count": stuck_count,
        "stop_reason": stop_reason,
        "duration_seconds": perf_counter() - started,
        "parameters": {"seed": seed, "t0": t0, "alpha": alpha, "t_min": t_min,
                       "stuck_window": stuck_window, "history_step": history_step,
                       "max_iterations": max_iterations, "objective": objective},
        "trace": {"scores": scores, "temperatures": temperatures,
                  "acceptance": acceptance},
    }
