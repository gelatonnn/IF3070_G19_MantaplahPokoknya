"""objective function alternatif (bonus 1): fee dikali bobot urgensi dari eta"""

#idenya: kendaraan yg eta-nya kecil (sisa harinya dikit) lebih mendesak buat dikirim,
#jadi nilainya dinaikin. ga nambah variabel baru, cuma pake shipping_fee sama eta
#
#urgensi = 1 + (eta_max - eta) / eta_max  -> range 1 (paling santai) s/d 2 (paling mendesak)
#score   = sum(fee * urgensi) - penalti * (overlap + excess_weight)
#
#penaltinya sama kyk evaluate() biasa, cuma koefisiennya disesuain sama
#nilai kendaraan terbesar setelah dikali urgensi biar langkah ilegal tetep rugi

from .evaluation import evaluate


def urgency(vehicle, eta_max):
    #kalo semua eta 0 (eta_max = 0) ya semua dianggep sama mendesaknya
    if eta_max == 0:
        return 1.0
    return 1 + (eta_max - vehicle["eta"]) / eta_max


def evaluate_eta(state, vehicles, ship):
    #pake evaluate() biasa dulu buat ngitung grid, overlap, berat, dll
    metrics = evaluate(state, vehicles, ship)

    eta_max = max((v["eta"] for v in vehicles.values()), default=0)

    weighted_revenue = 0.0
    for vid, placement in state.items():
        if placement["position"] is not None:
            vehicle = vehicles[vid]
            weighted_revenue += vehicle["shipping_fee"] * urgency(vehicle, eta_max)

    #koef penalti = 1 + nilai berbobot terbesar, sama logikanya kyk P di evaluate()
    penalty = 1 + max((v["shipping_fee"] * urgency(v, eta_max) for v in vehicles.values()),
                      default=0)

    #score diganti, sisanya (revenue asli, overlap, is_valid, grid) tetep dari evaluate()
    #jadi plot & laporan tetep bisa nampilin fee asli
    metrics["weighted_revenue"] = round(weighted_revenue, 4)
    metrics["penalty_coefficient"] = penalty
    metrics["score"] = round(weighted_revenue - penalty * (metrics["overlap"] + metrics["excess_weight"]), 4)
    return metrics
