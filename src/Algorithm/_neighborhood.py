"""Evaluasi delta untuk neighborhood yang sama dengan Core.iter_neighbors.

Heuristik implementasi tim: kandidat disimpan sebagai perubahan 1-2 placement,
bukan deepcopy seluruh state. Grid awal tetap berasal dari Core.evaluate.
Tidak menyaring overlap/overweight dan tidak mengurangi ruang tetangga.
Tes membandingkan himpunan tetangga DAN score dengan implementasi core asli.
"""

from itertools import combinations

from ..Core.evaluation import evaluate
from ..Core.state import copy_state, footprint
from ..Core.fleet import ships_by_id, placement_ship_id, make_placement
from ..Core.objectives import normalize_objective, vehicle_values


class Neighborhood:
    def __init__(self, vehicles, ship, objective="shipping_fee"):
        self.vehicles = vehicles
        self.ship = ship
        self.fleet = ships_by_id(ship)
        self.objective = normalize_objective(objective)
        self.values = vehicle_values(vehicles, self.objective)
        offsets = {}
        size = 0
        for sid, deck in self.fleet.items():
            offsets[sid] = size
            size += deck["w"] * deck["l"]
        self.cells = {}
        self.positions = {}
        for vid, vehicle in vehicles.items():
            for orientation in ("H", "V"):
                dx, dy = footprint(vehicle, orientation)
                self.positions[vid, orientation] = [(None, None)]
                self.cells[vid, None, None, orientation] = ()
                for sid, deck in self.fleet.items():
                    for y in range(max(0, deck["w"] - dy + 1)):
                        for x in range(max(0, deck["l"] - dx + 1)):
                            self.positions[vid, orientation].append((sid, (x, y)))
                            self.cells[vid, sid, (x, y), orientation] = tuple(
                                offsets[sid] + yy * deck["l"] + xx
                                for yy in range(y, y + dy)
                                for xx in range(x, x + dx)
                            )

    def bind(self, state):
        """Baca score/grid acuan setiap iterasi; juga memvalidasi state."""
        self.state = state
        self.metrics = evaluate(state, self.vehicles, self.ship, self.objective)
        self.grid = [count for sid in self.fleet
                     for row in self.metrics["per_ship"][sid]["grid"] for count in row]
        self.placements = {
            vid: (placement_ship_id(p, self.ship),
                  tuple(p["position"]) if p["position"] is not None else None,
                  p["orientation"])
            for vid, p in state.items()
        }

    def changes(self):
        for vid in self.vehicles:
            sid, position, orientation = self.placements[vid]
            for target_sid, target in self.positions[vid, orientation]:
                if (target_sid, target) != (sid, position):
                    yield ((vid, target_sid, target, orientation),)
            rotated = "V" if orientation == "H" else "H"
            if (vid, sid, position, rotated) in self.cells:
                yield ((vid, sid, position, rotated),)
        for first, second in combinations(self.vehicles, 2):
            sa, a, ao = self.placements[first]
            sb, b, bo = self.placements[second]
            # Kapal DAN anchor sama = no-op, termasuk kedua kendaraan di luar.
            # Satu di luar tetap swap dua placement: kendaraan masuk DAN keluar.
            if (sa, a) == (sb, b):
                continue
            if (first, sb, b, ao) in self.cells and (second, sa, a, bo) in self.cells:
                yield ((first, sb, b, ao), (second, sa, a, bo))

    def score(self, changes):
        counts = {}
        value_delta = 0
        weight_deltas = {}
        for vid, sid, position, orientation in changes:
            old_sid, old_position, old_orientation = self.placements[vid]
            for cell in self.cells[vid, old_sid, old_position, old_orientation]:
                counts[cell] = counts.get(cell, 0) - 1
            for cell in self.cells[vid, sid, position, orientation]:
                counts[cell] = counts.get(cell, 0) + 1
            loaded_delta = int(position is not None) - int(old_position is not None)
            value_delta += loaded_delta * self.values[vid]
            weight = self.vehicles[vid]["weight"]
            if old_sid is not None:
                weight_deltas[old_sid] = weight_deltas.get(old_sid, 0) - weight
            if sid is not None:
                weight_deltas[sid] = weight_deltas.get(sid, 0) + weight
        overlap_delta = sum(
            max(0, self.grid[cell] + delta - 1) - max(0, self.grid[cell] - 1)
            for cell, delta in counts.items() if delta
        )
        excess_delta = sum(
            max(0, self.metrics["per_ship"][sid]["total_weight"] + delta
                - self.fleet[sid]["max_capacity"])
            - self.metrics["per_ship"][sid]["excess_weight"]
            for sid, delta in weight_deltas.items() if delta
        )
        return round((
            self.metrics["score"] + value_delta
            - self.metrics["penalty_coefficient"] * (overlap_delta + excess_delta)
        ), 10)

    def apply(self, changes):
        result = copy_state(self.state)
        for vid, sid, position, orientation in changes:
            result[vid] = make_placement(position, orientation, self.ship, sid)
        return result
