"""
FUNGSI UTAMA: eval seberapa bagus neighbornya

Tumpang tindih menghitung kelebihan penumpang, BUKAN jumlah pasangan kendaraan yang tumpang tindih.
F = revenue - P * overlap - P * excess_weight; P = 1 + max(shipping_fee).
Tumpang tindih menghitung kelebihan vehicle, BUKAN jumlah pasangan kendaraan yang tumpang tindih.
"""
try:
    from .state import footprint, validate_state
except ImportError:
    from state import footprint, validate_state


def evaluate(state, vehicles, ship):
    validate_state(state, vehicles, ship)
    penalty = 1 + max((v['shipping_fee'] for v in vehicles.values()), default=0)
    grid = [[0] * ship['l'] for _ in range(ship['w'])]
    revenue = total_weight = loaded_count = 0
    for vid, placement in state.items():
        if placement['position'] is None:
            continue
        vehicle = vehicles[vid]
        x, y = placement['position']
        dx, dy = footprint(vehicle, placement['orientation'])
        for yy in range(y, y + dy):
            for xx in range(x, x + dx):
                grid[yy][xx] += 1
        revenue += vehicle['shipping_fee']
        total_weight += vehicle['weight']
        loaded_count += 1
    overlap = sum(max(0, count - 1) for row in grid for count in row)
    excess_weight = max(0, total_weight - ship['max_capacity'])
    return {'revenue': revenue, 'total_weight': total_weight,
            'overlap': overlap, 'excess_weight': excess_weight,
            'penalty_coefficient': penalty,
            'score': revenue - penalty * (overlap + excess_weight),
            'is_valid': overlap == 0 and excess_weight == 0,
            'loaded_count': loaded_count, 'grid': grid}
