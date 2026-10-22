"""Kontrak kapal tunggal dan armada; tanpa menambah atribut master kendaraan."""


def is_fleet(ship):
    return isinstance(ship, dict) and "ships" in ship


def ships_by_id(ship):
    """Legacy ship memakai ID internal 'ship_1'; fleet memakai ID dari input."""
    if not isinstance(ship, dict):
        raise ValueError("Konfigurasi kapal harus object")
    rows = ship["ships"] if is_fleet(ship) else [ship]
    if not isinstance(rows, list) or not rows:
        raise ValueError("ships harus list tidak kosong")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Setiap kapal harus object")
        sid = row.get("id") if is_fleet(ship) else "ship_1"
        if not isinstance(sid, str) or not sid.strip() or sid in result:
            raise ValueError("ID kapal harus string nonkosong dan unik")
        for field in ("w", "l", "max_capacity"):
            if type(row.get(field)) is not int or row[field] < 1:
                raise ValueError(f"{sid}.{field} harus integer >= 1")
        result[sid] = row
    return result


def placement_ship_id(placement, ship):
    if placement["position"] is None:
        return None
    return placement["ship_id"] if is_fleet(ship) else "ship_1"


def make_placement(position, orientation, ship, ship_id=None):
    result = {"position": tuple(position) if position is not None else None,
              "orientation": orientation}
    if is_fleet(ship):
        result["ship_id"] = ship_id if position is not None else None
    return result
