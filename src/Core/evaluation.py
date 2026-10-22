"""Nilai armada = reward semua kendaraan terangkut - penalti per kapal."""
from .state import footprint, validate_state
from .fleet import is_fleet, ships_by_id, placement_ship_id
from .objectives import normalize_objective, vehicle_values


def evaluate(state, vehicles, ship, objective="shipping_fee"):
    objective = normalize_objective(objective)
    validate_state(state, vehicles, ship)
    values = vehicle_values(vehicles, objective)
    penalty = 1 + max(values.values(), default=0)
    fleet = ships_by_id(ship)
    per_ship = {
        sid: {"grid": [[0] * deck["l"] for _ in range(deck["w"])],
              "revenue": 0, "objective_value": 0, "total_weight": 0,
              "loaded_count": 0, "max_capacity": deck["max_capacity"]}
        for sid, deck in fleet.items()
    }
    for vid, placement in state.items():
        sid = placement_ship_id(placement, ship)
        if sid is None:
            continue
        metrics = per_ship[sid]
        vehicle = vehicles[vid]
        x, y = placement['position']
        dx, dy = footprint(vehicle, placement['orientation'])
        for yy in range(y, y + dy):
            for xx in range(x, x + dx):
                metrics["grid"][yy][xx] += 1
        metrics["revenue"] += vehicle["shipping_fee"]
        metrics["objective_value"] += values[vid]
        metrics["total_weight"] += vehicle["weight"]
        metrics["loaded_count"] += 1
    for metrics in per_ship.values():
        metrics["objective_value"] = round(metrics["objective_value"], 10)
        metrics["overlap"] = sum(max(0, n - 1) for row in metrics["grid"] for n in row)
        metrics["excess_weight"] = max(0, metrics["total_weight"] - metrics["max_capacity"])
        metrics["score"] = round(metrics["objective_value"] - penalty * (
            metrics["overlap"] + metrics["excess_weight"]), 10)
        metrics["is_valid"] = metrics["overlap"] == 0 and metrics["excess_weight"] == 0
    result = {
        key: sum(m[key] for m in per_ship.values())
        for key in ("revenue", "objective_value", "total_weight", "loaded_count",
                    "overlap", "excess_weight", "score")
    }
    result["objective_value"] = round(result["objective_value"], 10)
    result["score"] = round(result["score"], 10)
    result.update(objective=objective, penalty_coefficient=penalty,
                  is_valid=all(m["is_valid"] for m in per_ship.values()),
                  per_ship=per_ship)
    if not is_fleet(ship):
        result["grid"] = per_ship["ship_1"]["grid"]
    return result
