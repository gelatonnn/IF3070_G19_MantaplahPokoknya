"""Satu sumber definisi objective untuk seluruh algoritma pencarian."""

OBJECTIVES = ("shipping_fee", "eta_priority")


def normalize_objective(objective):
    """Validasi nama objective baku; alias tidak diterima."""
    if objective not in OBJECTIVES:
        raise ValueError(f"Objective tidak dikenal: {objective}. Pilihan: {OBJECTIVES}")
    return objective


def vehicle_values(vehicles, objective="shipping_fee"):
    objective = normalize_objective(objective)
    if objective == "shipping_fee":
        return {vid: v["shipping_fee"] for vid, v in vehicles.items()}
    for vid, vehicle in vehicles.items():
        if type(vehicle.get("eta")) is not int or vehicle["eta"] < 0:
            raise ValueError(f"{vid}.eta harus integer >= 0 untuk eta_priority")
    eta_max = max((v["eta"] for v in vehicles.values()), default=0)
    if eta_max == 0:
        return {vid: v["shipping_fee"] for vid, v in vehicles.items()}
    # Rumus bonus branch tim: urgensi berada pada rentang [1, 2].
    return {
        vid: v["shipping_fee"] * (1 + (eta_max - v["eta"]) / eta_max)
        for vid, v in vehicles.items()
    }
