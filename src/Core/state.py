"""universal state dengan ket berikut: x = column; y = row; anchor = top-left cell."""
import random
from copy import deepcopy


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


def empty_state(vehicles):
    """semua ID tetap selalu ada, even yang lg ga aktif/gajadi naik"""
    return {vid: {'position': None, 'orientation': 'H'} for vid in vehicles}


def copy_state(state):
    return deepcopy(state)


def validate_state(state, vehicles, ship):
    """cek kanan kiri depan belakang ukurannya muat ga"""
    if not isinstance(state, dict) or set(state) != set(vehicles):
        raise ValueError('ID dalam state harus sama persis dengan dataset')
    for vid, placement in state.items():
        if not isinstance(placement, dict) or not {'position', 'orientation'} <= placement.keys():
            raise ValueError(f'Format placement {vid} tidak lengkap')
        if not within_bounds(vehicles[vid], placement['position'], placement['orientation'], ship):
            raise ValueError(f'Posisi {vid} tidak valid atau keluar batas geladak')


def random_state(vehicles, ship, rng=None, load_probability=0.5):
    if not 0 <= load_probability <= 1:
        raise ValueError('load_probability harus antara 0 dan 1')
    rng = rng if rng is not None else random.Random()
    result = empty_state(vehicles)
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
