"""
FUNGSI UTAMA: bikin semua alternatif dari susunan state skrg
Move, rotate, swap. Each operation returns an independent new state.

Memperbolehkan tumpang tindih dan kelebihan bobot (di handle evaluation). 
Operasi yang tidak valid akan memicu ValueError. Kondisi di luar area dinyatakan dengan position=None.
"""
import random
from itertools import combinations
from .state import copy_state, validate_state, within_bounds, footprint


def _place(state, vid, position, orientation, vehicles, ship):
    if vid not in vehicles:
        raise ValueError(f'ID tidak dikenal: {vid}')
    if not within_bounds(vehicles[vid], position, orientation, ship):
        raise ValueError(f'Operasi membuat {vid} keluar geladak')
    result = copy_state(state)
    result[vid] = {'position': tuple(position) if position is not None else None,
                   'orientation': orientation}
    return result


def move(state, vid, position, vehicles, ship):
    validate_state(state, vehicles, ship)
    if vid not in vehicles:
        raise ValueError(f'ID tidak dikenal: {vid}')
    return _place(state, vid, position, state[vid]['orientation'], vehicles, ship)


def rotate(state, vid, vehicles, ship):
    validate_state(state, vehicles, ship)
    if vid not in vehicles:
        raise ValueError(f'ID tidak dikenal: {vid}')
    orientation = 'V' if state[vid]['orientation'] == 'H' else 'H'
    return _place(state, vid, state[vid]['position'], orientation, vehicles, ship)


def swap(state, first, second, vehicles, ship):
    """Exchange anchors/outside status, retaining each vehicle's own orientation."""
    validate_state(state, vehicles, ship)
    if first not in vehicles or second not in vehicles or first == second:
        raise ValueError('Swap memerlukan dua ID berbeda yang dikenal')
    a, b = state[first], state[second]
    result = _place(state, first, b['position'], a['orientation'], vehicles, ship)
    return _place(result, second, a['position'], b['orientation'], vehicles, ship)


def iter_neighbors(state, vehicles, ship):
    validate_state(state, vehicles, ship)
    seen = set()
    def unique(candidate):
        key = tuple((vid, tuple(p['position']) if p['position'] is not None else None,
                     p['orientation']) for vid, p in sorted(candidate.items()))
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
        for y in range(max(0, ship['w'] - dy + 1)):
            for x in range(max(0, ship['l'] - dx + 1)):
                candidate = move(state, vid, (x, y), vehicles, ship)
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
                if rng.random() < 0.5:
                    position = None
                elif dx <= ship['l'] and dy <= ship['w']:
                    position = (rng.randrange(ship['l'] - dx + 1),
                                rng.randrange(ship['w'] - dy + 1))
                else:
                    continue
                candidate = move(state, vid, position, vehicles, ship)
        except ValueError:
            continue
        if candidate != state:
            return candidate
    return next(iter_neighbors(state, vehicles, ship), None)
