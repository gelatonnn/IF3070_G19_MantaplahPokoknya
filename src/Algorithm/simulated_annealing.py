"""Simulated Annealing utk penempatan kendaraannya"""

import math
import random
from time import perf_counter

from ..Core.evaluation import evaluate
from ..Core.neighbors import random_neighbor
from ..Core.state import copy_state, empty_state


def _point(iteration, temperature, metrics):
    #1 titik history buat disimpen ke result.json
    return {
        "iteration": iteration,
        "temperature": temperature,
        "score": metrics["score"],
        "revenue": metrics["revenue"],
        "overlap": metrics["overlap"],
        "excess_weight": metrics["excess_weight"],
        "is_valid": metrics["is_valid"],
    }


def simulated_annealing(initial_state, vehicles, ship, *, seed=0,
                        t0=100.0, alpha=0.9999, t_min=0.1,
                        stuck_window=1000, history_step=10):
    #stuck_window: brp iterasi tanpa perbaikan baru dianggep stuck 1x
    #history_step: history di json disimpen tiap n iterasi aja biar filenya ga gede
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

    #buat ngitung stuck, pake score terbaik apa aja (valid / belom)
    best_score = current_metrics["score"]
    since_improve = 0
    stuck_count = 0

    #data lengkap tiap iterasi buat plot
    scores = [current_metrics["score"]]
    temperatures = [t0]
    acceptance = []  #isinya (iterasi, e^(delta/T)) cuma utk langkah yg lebih jelek
    history = [_point(0, t0, current_metrics)]

    #counter langkah buat analisis di laporan
    accepted_better = accepted_equal = accepted_worse = rejected = 0

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
        if delta > 0:
            accept = True
            accepted_better += 1
        elif delta == 0:
            #e^(0/T) = 1 jd langkah yg sama aja pasti diterima
            accept = True
            accepted_equal += 1
        else:
            probability = math.exp(delta / temperature)
            #dicatet buat plot e^(delta/T), delta > 0 ga dicatet krn hasilnya > 1
            acceptance.append((iteration + 1, probability))
            accept = rng.random() < probability
            if accept:
                accepted_worse += 1
            else:
                rejected += 1

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

        #ngitung stuck: kalo udah stuck_window iterasi score terbaik ga naik2,
        #itung 1x stuck. periode stuck yg panjang tetep diitung 1x aja
        if current_metrics["score"] > best_score:
            best_score = current_metrics["score"]
            since_improve = 0
        else:
            since_improve += 1
            if since_improve == stuck_window:
                stuck_count += 1

        #catet data buat plot
        scores.append(current_metrics["score"])
        temperatures.append(temperature)
        if iteration % history_step == 0:
            history.append(_point(iteration, temperature, current_metrics))

    #pastiin titik terakhir masuk history juga
    if history[-1]["iteration"] != iteration:
        history.append(_point(iteration, temperature, current_metrics))

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
        "history": history,
        "iterations": iteration,
        "final_temperature": temperature,
        "accepted_better_moves": accepted_better,
        "accepted_equal_moves": accepted_equal,
        "accepted_worse_moves": accepted_worse,
        "rejected_moves": rejected,
        "stuck_count": stuck_count,
        "stop_reason": stop_reason,
        "duration_seconds": perf_counter() - started,
        "parameters": {"seed": seed, "t0": t0, "alpha": alpha, "t_min": t_min,
                       "stuck_window": stuck_window, "history_step": history_step},
        #data lengkap buat plot, nanti dibuang sama main.py sebelum nulis json
        "trace": {"scores": scores, "temperatures": temperatures,
                  "acceptance": acceptance},
    }
