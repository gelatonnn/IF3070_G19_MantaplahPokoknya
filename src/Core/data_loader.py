import json
from pathlib import Path
from .fleet import ships_by_id


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} harus integer >= {minimum}')


def load_data(path):
    with Path(path).open(encoding='utf-8-sig') as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError('Root JSON harus berupa object')
    if ("ship" in data) == ("ships" in data):
        raise ValueError("Gunakan tepat satu dari ship atau ships")
    ship = {"ships": data["ships"]} if "ships" in data else data["ship"]
    rows = data.get('vehicles')
    if not isinstance(ship, dict) or not isinstance(rows, list):
        raise ValueError('Diperlukan konfigurasi kapal dan list vehicles')
    ships_by_id(ship)
    vehicles = {}
    for index, vehicle in enumerate(rows):
        if not isinstance(vehicle, dict):
            raise ValueError(f'vehicles[{index}] harus object')
        vid = vehicle.get('id')
        if not isinstance(vid, str) or not vid.strip() or vid in vehicles:
            raise ValueError(f'ID kosong, bukan string, atau duplikat: {vid!r}')
        for field in ('w', 'l', 'weight'):
            _integer(vehicle.get(field), f'{vid}.{field}')
        for field in ('shipping_fee', 'eta'):
            _integer(vehicle.get(field), f'{vid}.{field}', minimum=0)
        vehicles[vid] = dict(vehicle)
    return dict(ship), vehicles


def load_fleet(path):
    """Konfigurasi armada terpisah untuk memakai kembali master kendaraan."""
    with Path(path).open(encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict) or set(data) != {"ships"}:
        raise ValueError("File armada terpisah hanya berisi key ships")
    ships_by_id(data)
    return data
