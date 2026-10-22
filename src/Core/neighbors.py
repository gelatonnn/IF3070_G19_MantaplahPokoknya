"""
FUNGSI UTAMA: bikin semua alternatif dari susunan state skrg
Move, rotate, swap. Each operation returns an independent new state.

Memperbolehkan tumpang tindih dan kelebihan bobot (di handle evaluation). 
Operasi yang tidak valid akan memicu ValueError. Kondisi di luar area dinyatakan dengan position=None.
"""
import random
from itertools import combinations
from .state import copy_state, validate_state, within_bounds, footprint
from .fleet import is_fleet, ships_by_id, placement_ship_id, make_placement


def _place(state, vid, position, orientation, vehicles, ship, ship_id=None):
    if vid not in vehicles:
        raise ValueError(f'ID tidak dikenal: {vid}')
    fleet = ships_by_id(ship)
    if position is None:
        ship_id = None
        deck = next(iter(fleet.values()))
    else:
        if not isinstance(ship_id, str) or ship_id not in fleet:
            raise ValueError("Tujuan kapal tidak dikenal")
        deck = fleet[ship_id]
    if not within_bounds(vehicles[vid], position, orientation, deck):
        raise ValueError(f'Operasi membuat {vid} keluar geladak')
    result = copy_state(state)
    result[vid] = make_placement(position, orientation, ship, ship_id)
    return result


def move(state, vid, position, vehicles, ship, *, ship_id=None):
    """ship_id eksplisit untuk masuk/pindah kapal; None = kapal saat ini."""
    validate_state(state, vehicles, ship)
    if vid not in vehicles:
        raise ValueError(f'ID tidak dikenal: {vid}')
    if ship_id is None:
        ship_id = placement_ship_id(state[vid], ship) if is_fleet(ship) else "ship_1"
    return _place(state, vid, position, state[vid]['orientation'], vehicles, ship, ship_id)


def rotate(state, vid, vehicles, ship):
    validate_state(state, vehicles, ship)
    if vid not in vehicles:
        raise ValueError(f'ID tidak dikenal: {vid}')
    orientation = 'V' if state[vid]['orientation'] == 'H' else 'H'
    return _place(state, vid, state[vid]['position'], orientation, vehicles, ship,
                  placement_ship_id(state[vid], ship))


def swap(state, first, second, vehicles, ship):
    """Tukar kapal + anchor/status luar; orientasi tiap kendaraan tetap."""
    validate_state(state, vehicles, ship)
    if first not in vehicles or second not in vehicles or first == second:
        raise ValueError('Swap memerlukan dua ID berbeda yang dikenal')
    a, b = state[first], state[second]
    result = _place(state, first, b['position'], a['orientation'], vehicles, ship,
                    placement_ship_id(b, ship))
    return _place(result, second, a['position'], b['orientation'], vehicles, ship,
                  placement_ship_id(a, ship))


def iter_neighbors(state, vehicles, ship):
    validate_state(state, vehicles, ship)
    fleet = ships_by_id(ship)
    seen = set()
    def unique(candidate):
        key = tuple((vid, tuple(p['position']) if p['position'] is not None else None,
                     p['orientation'], p.get('ship_id')) for vid, p in sorted(candidate.items()))
        if candidate == state or key in seen:
            return False
        seen.add(key)
        return True
    for vid, vehicle in vehicles.items():
        orientation = state[vid]['orientation']
        dx, dy = footprint(vehicle, orientation)
        positions = (None,)
        for position in positions:
            candidate = move(state, vid, position, vehicles, ship)
            if unique(candidate):
                yield candidate
        for sid, deck in fleet.items():
            for y in range(max(0, deck['w'] - dy + 1)):
                for x in range(max(0, deck['l'] - dx + 1)):
                    candidate = move(state, vid, (x, y), vehicles, ship, ship_id=sid)
                    if unique(candidate):
                        yield candidate
        try:
            candidate = rotate(state, vid, vehicles, ship)
        except ValueError:
            continue
        if unique(candidate):
            yield candidate
    for first, second in combinations(vehicles, 2):
        try:
            candidate = swap(state, first, second, vehicles, ship)
        except ValueError:
            continue
        if unique(candidate):
            yield candidate


def random_neighbor(state, vehicles, ship, rng=None, max_attempts=100):
    validate_state(state, vehicles, ship)
    if type(max_attempts) is not int or max_attempts < 0:
        raise ValueError('max_attempts harus integer >= 0')
    if not vehicles:
        return None
    rng = rng if rng is not None else random.Random()
    fleet = ships_by_id(ship)
    ids = list(vehicles)
    for _ in range(max_attempts):
        vid = rng.choice(ids)
        operation = rng.choice(('move', 'rotate', 'swap'))
        try:
            if operation == 'rotate':
                candidate = rotate(state, vid, vehicles, ship)
            elif operation == 'swap':
                if len(ids) < 2:
                    continue
                first, second = rng.sample(ids, 2)
                candidate = swap(state, first, second, vehicles, ship)
            else:
                dx, dy = footprint(vehicles[vid], state[vid]['orientation'])
                sid = rng.choice(list(fleet)) if is_fleet(ship) else "ship_1"
                deck = fleet[sid]
                if rng.random() < 0.5:
                    position = None
                elif dx <= deck['l'] and dy <= deck['w']:
                    position = (rng.randrange(deck['l'] - dx + 1),
                                rng.randrange(deck['w'] - dy + 1))
                else:
                    continue
                candidate = move(state, vid, position, vehicles, ship, ship_id=sid)
        except ValueError:
            continue
        if candidate != state:
            return candidate
    return next(iter_neighbors(state, vehicles, ship), None)
