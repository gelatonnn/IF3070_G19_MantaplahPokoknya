import random
from time import perf_counter

from ..Core.evaluation import evaluate
from ..Core.state import copy_state, empty_state, random_state
from ._neighborhood import Neighborhood


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} harus integer >= 0")


def _point(iteration, metrics):
    return {
        "iteration": iteration,
        **{name: metrics[name] for name in (
            "score", "revenue", "overlap", "excess_weight", "is_valid",
        )},
    }


def _search(initial_state, vehicles, ship, *, mode, rng, max_iterations,
            max_sideways=0, neighborhood=None):
    if max_iterations is not None:
        _nonnegative_int(max_iterations, "max_iterations")
    _nonnegative_int(max_sideways, "max_sideways")
    started = perf_counter()
    current = copy_state(initial_state)
    initial_metrics = evaluate(current, vehicles, ship)
    metrics = initial_metrics
    
    best_valid = empty_state(vehicles)
    best_valid_metrics = evaluate(best_valid, vehicles, ship)
    best_valid_source = "empty_baseline"
    if metrics["is_valid"]:
        best_valid, best_valid_metrics = copy_state(current), metrics
        best_valid_source = "visited"
    history = [_point(0, metrics)]
    iterations = scans = evaluated = sideways = improvements = 0
    space = neighborhood if neighborhood is not None else Neighborhood(vehicles, ship)
    while True:
        if max_iterations is not None and iterations >= max_iterations:
            reason = "max_iterations"
            break
        space.bind(current)
        scans += 1
        selected = lateral = None
        selected_score = metrics["score"]
        improving_count = lateral_count = 0
        for change in space.changes():
            evaluated += 1
            score = space.score(change)
            if score > metrics["score"]:
                if mode == "stochastic":
                    improving_count += 1
                    if rng.randrange(improving_count) == 0:
                        selected, selected_score = change, score
                elif score > selected_score:
                    selected, selected_score = change, score
            elif mode == "sideways" and score == metrics["score"]:
                lateral_count += 1
                if rng.randrange(lateral_count) == 0:
                    lateral = change
        if selected is None:
            if mode == "sideways" and lateral is not None and sideways < max_sideways:
                selected = lateral
                selected_score = metrics["score"]
                sideways += 1
            else:
                reason = ("sideways_limit" if mode == "sideways" and lateral is not None
                          else "local_optimum")
                break
        else:
            improvements += 1
        current = space.apply(selected)
        metrics = evaluate(current, vehicles, ship)
        if metrics["score"] != selected_score:
            raise RuntimeError("Score delta tidak sama dengan evaluator core")
        iterations += 1
        history.append(_point(iterations, metrics))
        if metrics["is_valid"] and metrics["score"] >= best_valid_metrics["score"]:
            best_valid, best_valid_metrics = copy_state(current), metrics
            best_valid_source = "visited"
        if mode == "sideways" and max_sideways > 0 and sideways >= max_sideways:
            reason = "sideways_limit"
            break

    final = current if metrics["is_valid"] else best_valid
    final_metrics = evaluate(final, vehicles, ship)
    return {
        "algorithm": mode,
        "initial_state": copy_state(initial_state),
        "initial_metrics": initial_metrics,
        "final_state": copy_state(final),
        "final_metrics": final_metrics,
        "terminal_state": copy_state(current),
        "terminal_metrics": metrics,
        "final_source": "terminal" if metrics["is_valid"] else best_valid_source,
        "history": history,
        "iterations": iterations,
        "neighborhood_scans": scans,
        "evaluated_neighbors": evaluated,
        "improving_moves": improvements,
        "sideways_moves": sideways,
        "stop_reason": reason,
        "duration_seconds": perf_counter() - started,
        "parameters": {"max_iterations": max_iterations, "max_sideways": max_sideways},
    }


def steepest_ascent_hill_climbing(initial_state, vehicles, ship, *,
                                  seed=0, max_iterations=None):
    """Tetangga terbaik; tie terbaik mengikuti urutan enumerasi core."""
    result = _search(initial_state, vehicles, ship, mode="steepest",
                     rng=random.Random(seed), max_iterations=max_iterations)
    result["parameters"]["seed"] = seed
    return result


def stochastic_hill_climbing(initial_state, vehicles, ship, *,
                            seed=0, max_iterations=None):
    """Pilih uniform di antara tetangga dengan score lebih tinggi."""
    result = _search(initial_state, vehicles, ship, mode="stochastic",
                     rng=random.Random(seed), max_iterations=max_iterations)
    result["parameters"]["seed"] = seed
    return result


def hill_climbing_sideways(initial_state, vehicles, ship, *, seed=0,
                          max_iterations=None, max_sideways=20):
    """Steepest + langkah setara; berhenti saat batas TOTAL sideways tercapai.

    max_sideways=0 berarti tidak menerima sideways, tetapi perbaikan boleh.
    Penghitung tidak di-reset setelah langkah membaik.
    """
    result = _search(initial_state, vehicles, ship, mode="sideways",
                     rng=random.Random(seed), max_iterations=max_iterations,
                     max_sideways=max_sideways)
    result["parameters"]["seed"] = seed
    return result


def random_restart_hill_climbing(initial_state, vehicles, ship, *, seed=0,
                                 max_restarts=3, max_iterations=None,
                                 load_probability=0.5):
    """Satu pencarian awal + max_restarts pencarian steepest dari state acak."""
    _nonnegative_int(max_restarts, "max_restarts")
    if max_iterations is not None:
        _nonnegative_int(max_iterations, "max_iterations")
    if not 0 <= load_probability <= 1:
        raise ValueError("load_probability harus antara 0 dan 1")
    started = perf_counter()
    rng = random.Random(seed)
    evaluate(initial_state, vehicles, ship)
    space = Neighborhood(vehicles, ship)
    runs = []
    for index in range(max_restarts + 1):
        initial = (initial_state if index == 0 else
                   random_state(vehicles, ship, rng, load_probability))
        run = _search(initial, vehicles, ship, mode="steepest", rng=rng,
                      max_iterations=max_iterations, neighborhood=space)
        run["restart_index"] = index
        runs.append(run)
    best_index = max(range(len(runs)), key=lambda i: runs[i]["final_metrics"]["score"])
    best = runs[best_index]
    return {
        "algorithm": "random_restart",
        "initial_state": copy_state(initial_state),
        "initial_metrics": runs[0]["initial_metrics"],
        "final_state": copy_state(best["final_state"]),
        "final_metrics": best["final_metrics"],
        "final_source": best["final_source"],
        "best_run_index": best_index,
        "best_run_initial_state": copy_state(best["initial_state"]),
        "runs": runs,
        "iterations": sum(run["iterations"] for run in runs),
        "iterations_per_restart": [run["iterations"] for run in runs],
        "restarts": max_restarts,
        "searches": len(runs),
        "evaluated_neighbors": sum(run["evaluated_neighbors"] for run in runs),
        "stop_reason": "max_restarts",
        "duration_seconds": perf_counter() - started,
        "parameters": {"seed": seed, "max_restarts": max_restarts,
                       "max_iterations": max_iterations,
                       "load_probability": load_probability},
    }
