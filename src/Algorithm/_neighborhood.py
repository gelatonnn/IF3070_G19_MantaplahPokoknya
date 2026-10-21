"""Evaluasi delta untuk neighborhood yang sama dengan Core.iter_neighbors.

Heuristik implementasi tim: kandidat disimpan sebagai perubahan 1-2 placement,
bukan deepcopy seluruh state. Grid awal tetap berasal dari Core.evaluate.
Tidak menyaring overlap/overweight dan tidak mengurangi ruang tetangga.
Tes membandingkan himpunan tetangga DAN score dengan implementasi core asli.
"""

from itertools import combinations

from ..Core.evaluation import evaluate
from ..Core.state import copy_state, footprint


class Neighborhood:
    def __init__(self, vehicles, ship):
        self.vehicles = vehicles
        self.ship = ship
        self.cells = {}
        self.positions = {}
        for vid, vehicle in vehicles.items():
            for orientation in ("H", "V"):
                dx, dy = footprint(vehicle, orientation)
                positions = [
                    (x, y)
                    for y in range(max(0, ship["w"] - dy + 1))
                    for x in range(max(0, ship["l"] - dx + 1))
                ]
                self.positions[vid, orientation] = [None, *positions]
                self.cells[vid, None, orientation] = ()
                for x, y in positions:
                    self.cells[vid, (x, y), orientation] = tuple(
                        yy * ship["l"] + xx
                        for yy in range(y, y + dy)
                        for xx in range(x, x + dx)
                    )

    def bind(self, state):
        """Baca score/grid acuan setiap iterasi; juga memvalidasi state."""
        self.state = state
        self.metrics = evaluate(state, self.vehicles, self.ship)
        self.grid = [count for row in self.metrics["grid"] for count in row]
        self.placements = {
            vid: (tuple(p["position"]) if p["position"] is not None else None,
                  p["orientation"])
            for vid, p in state.items()
        }

    def changes(self):
        for vid in self.vehicles:
            position, orientation = self.placements[vid]
            for target in self.positions[vid, orientation]:
                if target != position:
                    yield ((vid, target, orientation),)
            rotated = "V" if orientation == "H" else "H"
            if (vid, position, rotated) in self.cells:
                yield ((vid, position, rotated),)
        for first, second in combinations(self.vehicles, 2):
            a, ao = self.placements[first]
            b, bo = self.placements[second]
            # Anchor sama = no-op, termasuk kedua kendaraan di luar.
            # Satu di luar tetap swap dua placement: kendaraan masuk DAN keluar.
            if a == b:
                continue
            if (first, b, ao) in self.cells and (second, a, bo) in self.cells:
                yield ((first, b, ao), (second, a, bo))

    def score(self, changes):
        counts = {}
        fee_delta = weight_delta = 0
        for vid, position, orientation in changes:
            old_position, old_orientation = self.placements[vid]
            for cell in self.cells[vid, old_position, old_orientation]:
                counts[cell] = counts.get(cell, 0) - 1
            for cell in self.cells[vid, position, orientation]:
                counts[cell] = counts.get(cell, 0) + 1
            loaded_delta = int(position is not None) - int(old_position is not None)
            fee_delta += loaded_delta * self.vehicles[vid]["shipping_fee"]
            weight_delta += loaded_delta * self.vehicles[vid]["weight"]
        overlap_delta = sum(
            max(0, self.grid[cell] + delta - 1) - max(0, self.grid[cell] - 1)
            for cell, delta in counts.items() if delta
        )
        excess_delta = (
            max(0, self.metrics["total_weight"] + weight_delta - self.ship["max_capacity"])
            - self.metrics["excess_weight"]
        )
        return (
            self.metrics["score"] + fee_delta
            - self.metrics["penalty_coefficient"] * (overlap_delta + excess_delta)
        )

    def apply(self, changes):
        result = copy_state(self.state)
        for vid, position, orientation in changes:
            result[vid] = {"position": position, "orientation": orientation}
        return result
