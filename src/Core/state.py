"""universal state dengan ket berikut: x = column; y = row; anchor = top-left cell."""
import random
from copy import deepcopy
from .fleet import is_fleet, ships_by_id, make_placement


def footprint(vehicle, orientation):
    """Return (columns, rows) occupied after orientation."""
    if orientation not in ('H', 'V'):
        raise ValueError('orientation harus H atau V')
    return ((vehicle['l'], vehicle['w']) if orientation == 'H'
            else (vehicle['w'], vehicle['l']))


def within_bounds(vehicle, position, orientation, ship):
    dx, dy = footprint(vehicle, orientation)
    if position is None:
        return True
    if not isinstance(position, (tuple, list)) or len(position) != 2:
        return False
    x, y = position
    return (type(x) is int and type(y) is int
            and 0 <= x and 0 <= y
            and x + dx <= ship['l'] and y + dy <= ship['w'])


def empty_state(vehicles, ship=None):
    """Semua ID di luar. Berikan konfigurasi armada agar ship_id ikut disimpan."""
    return {vid: make_placement(None, 'H', ship) for vid in vehicles}


def copy_state(state):
    return deepcopy(state)


def validate_state(state, vehicles, ship):
    """cek kanan kiri depan belakang ukurannya muat ga"""
    if not isinstance(state, dict) or set(state) != set(vehicles):
        raise ValueError('ID dalam state harus sama persis dengan dataset')
    fleet = ships_by_id(ship)
    for vid, placement in state.items():
        if not isinstance(placement, dict) or not {'position', 'orientation'} <= placement.keys():
            raise ValueError(f'Format placement {vid} tidak lengkap')
        deck = ship
        if is_fleet(ship):
            if "ship_id" not in placement:
                raise ValueError(f"Placement {vid} memerlukan ship_id")
            sid = placement["ship_id"]
            if placement["position"] is None:
                if sid is not None:
                    raise ValueError(f"{vid} di luar harus memiliki ship_id=None")
                deck = next(iter(fleet.values()))
            else:
                if not isinstance(sid, str) or sid not in fleet:
                    raise ValueError(f"ID kapal untuk {vid} tidak dikenal")
                deck = fleet[sid]
        elif placement.get("ship_id") is not None:
            raise ValueError("ship_id hanya untuk state armada")
        if not within_bounds(vehicles[vid], placement['position'], placement['orientation'], deck):
            raise ValueError(f'Posisi {vid} tidak valid atau keluar batas geladak')


def random_state(vehicles, ship, rng=None, load_probability=0.5):
    if not 0 <= load_probability <= 1:
        raise ValueError('load_probability harus antara 0 dan 1')
    rng = rng if rng is not None else random.Random()
    fleet = ships_by_id(ship)
    result = empty_state(vehicles, ship)
    if is_fleet(ship):
        for vid, vehicle in vehicles.items():
            choices = [(sid, orientation) for sid, deck in fleet.items()
                       for orientation in ('H', 'V')
                       if footprint(vehicle, orientation)[0] <= deck['l']
                       and footprint(vehicle, orientation)[1] <= deck['w']]
            if not choices:
                result[vid]['orientation'] = rng.choice(('H', 'V'))
                continue
            sid, orientation = rng.choice(choices)
            result[vid]['orientation'] = orientation
            if rng.random() < load_probability:
                deck = fleet[sid]
                dx, dy = footprint(vehicle, orientation)
                position = (rng.randrange(deck['l'] - dx + 1),
                            rng.randrange(deck['w'] - dy + 1))
                result[vid] = make_placement(position, orientation, ship, sid)
        return result
    for vid, vehicle in vehicles.items():
        orientations = [o for o in ('H', 'V')
                        if footprint(vehicle, o)[0] <= ship['l']
                        and footprint(vehicle, o)[1] <= ship['w']]
        orientation = rng.choice(orientations) if orientations else rng.choice(('H', 'V'))
        result[vid]['orientation'] = orientation
        if orientations and rng.random() < load_probability:
            dx, dy = footprint(vehicle, orientation)
            result[vid]['position'] = (rng.randrange(ship['l'] - dx + 1),
                                       rng.randrange(ship['w'] - dy + 1))
    return result
