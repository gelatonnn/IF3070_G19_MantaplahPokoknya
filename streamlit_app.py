"""Antarmuka eksperimen dan perbandingan optimasi muatan kapal."""

import csv
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import streamlit as st

from src.Core.data_loader import load_data, load_fleet
from src.Core.fleet import ships_by_id
from src.Core.objectives import vehicle_values
from src.experiment import ALGORITHMS, ExperimentConfig, ROOT, run_experiments
from src.ui_support import (comparison_figure, comparison_summary, deck_figure,
                            history_figure, parse_seeds, read_result, run_bundle)


ALGORITHM_LABELS = {
    "steepest": "Hill Climbing · Steepest Ascent",
    "stochastic": "Hill Climbing · Stochastic",
    "sideways": "Hill Climbing · Sideways Move",
    "random_restart": "Hill Climbing · Random Restart",
    "simulated_annealing": "Simulated Annealing",
    "genetic_algorithm": "Genetic Algorithm",
}
RUN_ROOT = ROOT / "output" / "ui_runs"


def apply_style() -> None:
    st.markdown("""
    <style>
      .stApp { background: #F8F5ED; color: #233867; }
      [data-testid="stSidebar"] { background: #E3CE96; }
      [data-testid="stMetric"] { background: #fffdf8; border: 1px solid #dbc99f;
        border-radius: 12px; padding: 14px; }
      [data-testid="stMetricLabel"] { color: #3D607B; }
      div.stButton > button[kind="primary"],
      div.stFormSubmitButton > button[kind="primary"] {
        background: #233867; border-color: #233867; color: #fff;
      }
      div.stButton > button[kind="primary"]:hover,
      div.stFormSubmitButton > button[kind="primary"]:hover {
        background: #3D607B; border-color: #3D607B;
      }
      .hero { padding: 26px 30px; border-radius: 16px;
        background: linear-gradient(120deg, #233867, #3D607B);
        color: #E3CE96; margin-bottom: 20px; }
      .hero h1 { color: #E3CE96; margin: 0; font-size: 2rem; }
      .hero p { color: #fffdf8; margin: 8px 0 0; }
    </style>
    """, unsafe_allow_html=True)


def available_runs() -> list[Path]:
    if not RUN_ROOT.exists():
        return []
    return sorted((path for path in RUN_ROOT.iterdir()
                   if path.is_dir() and (path / "run_manifest.json").is_file()),
                  reverse=True)


def _read_rows(run_dir: Path) -> list[dict]:
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["seed"] = int(row["seed"])
        row["objective_value"] = float(row["objective_value"])
        row["duration_seconds"] = float(row["duration_seconds"])
        row["is_valid"] = row["is_valid"].lower() == "true"
    return rows


def render_result(run_dir: Path, algorithm: str, seed: int) -> None:
    folder = run_dir / algorithm / f"seed_{seed}"
    result = read_result(run_dir, algorithm, seed)
    metrics = result["final_metrics"]
    ship, vehicles = load_data(run_dir / "input_data.json")
    if (run_dir / "input_fleet.json").is_file():
        ship = load_fleet(run_dir / "input_fleet.json")

    st.subheader(f"Detail: {ALGORITHM_LABELS[algorithm]} · seed {seed}")
    st.caption(f"Objective: {result['experiment']['objective']} · Sumber hasil: {result['final_source']}")
    metric_cols = st.columns(4)
    for col, label, value in zip(metric_cols,
                                 ("Nilai objective", "Shipping fee", "Kendaraan dimuat", "Solusi valid"),
                                 (f"{metrics['objective_value']:,.2f}", f"{metrics['revenue']:,}",
                                  metrics["loaded_count"], "Ya" if metrics["is_valid"] else "Tidak")):
        col.metric(label, value)
    info_cols = st.columns(3)
    info_cols[0].metric("Berat total", f"{metrics['total_weight']:,}")
    info_cols[1].metric("Iterasi / generasi", result["iterations"])
    info_cols[2].metric("Durasi", f"{result['duration_seconds']:.2f} detik")
    st.caption(f"Berhenti karena: {result['stop_reason']}")

    if len(metrics["per_ship"]) > 1:
        st.dataframe([{"Kapal": sid, "Kendaraan": info["loaded_count"],
                       "Berat": info["total_weight"], "Kapasitas": info["max_capacity"],
                       "Shipping fee": info["revenue"], "Valid": info["is_valid"]}
                      for sid, info in metrics["per_ship"].items()], hide_index=True)

    layout_tab, history_tab, detail_tab = st.tabs(["Geladak", "Perkembangan", "Data & unduhan"])
    with layout_tab:
        ship_id = st.selectbox("Pilih kapal", list(ships_by_id(ship)),
                               key=f"ship_{run_dir.name}_{algorithm}_{seed}")
        initial_col, final_col = st.columns(2)
        with initial_col:
            st.plotly_chart(deck_figure(result["initial_state"], vehicles, ship, ship_id, "Awal"),
                            width="stretch", key=f"initial_{run_dir.name}_{algorithm}_{seed}_{ship_id}")
        with final_col:
            st.plotly_chart(deck_figure(result["final_state"], vehicles, ship, ship_id, "Akhir"),
                            width="stretch", key=f"final_{run_dir.name}_{algorithm}_{seed}_{ship_id}")
        st.caption("Arah sumbu y dari atas ke bawah. Arahkan kursor ke kendaraan untuk melihat atributnya.")
        with st.expander("Lihat PNG lengkap (termasuk kendaraan di luar kapal)"):
            first, second = st.columns(2)
            first.image(folder / "initial.png", width="stretch")
            second.image(folder / "final.png", width="stretch")
    with history_tab:
        st.plotly_chart(history_figure(result), width="stretch",
                        key=f"history_{run_dir.name}_{algorithm}_{seed}")
        st.caption("Score/fitness dapat mengandung penalti; nilai objective dibaca pada sumbu kanan.")
        with st.expander("Lihat grafik ekspor asli"):
            st.image(folder / "history.png", width="stretch")
            if (folder / "acceptance.png").is_file():
                st.image(folder / "acceptance.png", width="stretch")
    with detail_tab:
        st.json({"parameters": result["parameters"], "stop_reason": result["stop_reason"],
                 "final_source": result["final_source"], "experiment": result["experiment"]})
        for name, filename, mime in (
            ("JSON hasil", "result.json", "application/json"),
            ("PNG susunan awal", "initial.png", "image/png"),
            ("PNG susunan akhir", "final.png", "image/png"),
            ("PNG perkembangan", "history.png", "image/png"),
        ):
            st.download_button(f"Unduh {name}", (folder / filename).read_bytes(),
                               file_name=f"{algorithm}_seed_{seed}_{filename}", mime=mime,
                               key=f"download_{run_dir.name}_{algorithm}_{seed}_{filename}")


def render_run(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    rows = _read_rows(run_dir)
    st.divider()
    st.subheader("Hasil eksperimen")
    st.caption(f"{manifest['created_at']} · {manifest['objective']} · "
               f"{manifest['ship_count']} kapal · {manifest['vehicle_count']} kendaraan · "
               f"{len(rows)} eksekusi")
    if len(rows) > 1:
        st.plotly_chart(comparison_figure(rows, ALGORITHM_LABELS), width="stretch",
                        key=f"comparison_{run_dir.name}")
        st.dataframe(comparison_summary(rows), hide_index=True, width="stretch")
        st.caption("Perbandingan menggunakan dataset, konfigurasi kapal, dan objective yang sama. "
                   "Run tidak valid bernilai 0 pada grafik dan ditandai di tabel detail.")
    st.dataframe([{"Algoritma": ALGORITHM_LABELS[row["algorithm"]], "Seed": row["seed"],
                   "Objective": round(row["objective_value"], 2),
                   "Valid": row["is_valid"],
                   "Durasi (detik)": round(row["duration_seconds"], 3)}
                  for row in rows], hide_index=True, width="stretch")
    st.download_button("Unduh summary.csv", (run_dir / "summary.csv").read_bytes(),
                       file_name=f"{run_dir.name}_summary.csv", mime="text/csv",
                       key=f"summary_{run_dir.name}")
    st.download_button("Unduh seluruh hasil (.zip)", run_bundle(run_dir),
                       file_name=f"{run_dir.name}.zip", mime="application/zip",
                       key=f"bundle_{run_dir.name}")
    choices = [(row["algorithm"], row["seed"]) for row in rows]
    algorithm, seed = st.selectbox("Lihat detail eksekusi", choices,
                                   format_func=lambda item: f"{ALGORITHM_LABELS[item[0]]} · seed {item[1]}",
                                   key=f"detail_choice_{run_dir.name}")
    render_result(run_dir, algorithm, seed)


def main() -> None:
    st.set_page_config(page_title="Optimasi Muatan Kapal", page_icon="⛵", layout="wide")
    apply_style()
    st.markdown('<div class="hero"><h1>Optimasi Muatan Kapal</h1>'
                '<p>Jalankan, bandingkan, dan jelajahi solusi Hill Climbing, '
                'Simulated Annealing, serta Genetic Algorithm.</p></div>',
                unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Riwayat eksperimen")
        runs = available_runs()
        selected_run = st.selectbox("Buka hasil tersimpan", [None, *runs],
                                    format_func=lambda path: "Hasil terbaru" if path is None else path.name)
        st.caption("Setiap eksperimen disimpan di output/ui_runs; hasil lama tidak ditimpa.")

    mode = st.radio("Mode", ["Eksperimen tunggal", "Perbandingan"], horizontal=True)
    st.subheader("1 · Pilih data dan algoritma")
    data_col, algorithm_col = st.columns(2)
    with data_col:
        uploaded_data = st.file_uploader("Dataset kendaraan JSON (opsional)", type="json",
                                         max_upload_size=5,
                                         help="Kosongkan untuk memakai data/data.json.")
        fleet_mode = st.checkbox("Gunakan armada multi-kapal")
        uploaded_fleet = (st.file_uploader("Konfigurasi kapal JSON (opsional)", type="json",
                                           max_upload_size=5,
                                           help="Kosongkan untuk memakai data/fleet.json.")
                          if fleet_mode else None)
    with algorithm_col:
        if mode == "Eksperimen tunggal":
            algorithms = [st.selectbox("Algoritma", list(ALGORITHMS),
                                       format_func=lambda name: ALGORITHM_LABELS[name])]
        else:
            algorithms = st.multiselect("Algoritma yang dibandingkan", list(ALGORITHMS),
                                        default=list(ALGORITHMS),
                                        format_func=lambda name: ALGORITHM_LABELS[name])
        st.caption("Dataset bawaan memuat 90 kendaraan; konfigurasi bawaan armada berisi dua kapal.")

    st.subheader("2 · Atur eksperimen")
    with st.form("experiment_config"):
        objective = st.selectbox("Objective", ["shipping_fee", "eta_priority"])
        load_probability = st.slider("Probabilitas muat state awal", 0.0, 1.0, 0.5, 0.05)
        if mode == "Eksperimen tunggal":
            seeds = [int(st.number_input("Seed", value=1, step=1))]
        else:
            seed_text = st.text_input("Seed (pisahkan dengan koma)", value="1, 2, 3")
            seeds = None

        selected = set(algorithms)
        options = {}
        if selected & {"steepest", "stochastic", "sideways", "random_restart", "simulated_annealing"}:
            natural_stop = st.checkbox("Biarkan HC/SA berhenti alami", value=False)
            iteration_limit = int(st.number_input("Batas iterasi HC/SA", min_value=0,
                                                   value=100, step=10))
            options["max_iterations"] = None if natural_stop else iteration_limit
        if "sideways" in selected:
            options["max_sideways"] = int(st.number_input("Maksimum sideways move", min_value=0, value=20))
        if "random_restart" in selected:
            options["max_restarts"] = int(st.number_input("Restart tambahan", min_value=0, value=3))
        if "simulated_annealing" in selected:
            with st.expander("Parameter Simulated Annealing"):
                options["t0"] = float(st.number_input("Temperatur awal", min_value=0.001, value=100.0))
                options["alpha"] = float(st.number_input("Alpha", min_value=0.0001,
                                                          max_value=0.9999, value=0.9999,
                                                          format="%.4f"))
                options["t_min"] = float(st.number_input("Temperatur minimum", min_value=0.001, value=0.1))
                options["stuck_window"] = int(st.number_input("Stuck window", min_value=1, value=1000))
                options["history_step"] = int(st.number_input("Interval riwayat", min_value=1, value=10))
        if "genetic_algorithm" in selected:
            with st.expander("Parameter Genetic Algorithm"):
                options["population_size"] = int(st.number_input("Ukuran populasi", min_value=2, value=100))
                options["generations"] = int(st.number_input("Jumlah generasi", min_value=1, value=100))
                options["crossover_rate"] = st.slider("Crossover rate", 0.0, 1.0, 0.8, 0.05)
                options["mutation_rate"] = st.slider("Mutation rate", 0.0, 1.0, 0.5, 0.05)
                options["tournament_size"] = int(st.number_input("Tournament size", min_value=1, value=3))
                options["elite_count"] = int(st.number_input("Elite count", min_value=0, value=2))
                options["repair"] = st.checkbox("Perbaiki individu tidak valid")
        submitted = st.form_submit_button("Jalankan optimasi", type="primary")

    if submitted:
        try:
            if mode == "Perbandingan":
                seeds = parse_seeds(seed_text)
            if not algorithms:
                raise ValueError("Pilih sedikitnya satu algoritma.")
            if len(algorithms) * len(seeds) > 30:
                raise ValueError("Maksimum 30 eksekusi sekali jalan; kurangi algoritma atau seed.")
            RUN_ROOT.mkdir(parents=True, exist_ok=True)
            run_dir = RUN_ROOT / (datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + uuid4().hex[:8])
            with TemporaryDirectory(prefix="input_", dir=RUN_ROOT) as input_dir:
                data_path = ROOT / "data/data.json"
                fleet_path = ROOT / "data/fleet.json" if fleet_mode else None
                if uploaded_data is not None:
                    data_path = Path(input_dir) / "data.json"
                    data_path.write_bytes(uploaded_data.getvalue())
                if uploaded_fleet is not None:
                    fleet_path = Path(input_dir) / "fleet.json"
                    fleet_path.write_bytes(uploaded_fleet.getvalue())
                ship, vehicles = load_data(data_path)
                if fleet_path is not None:
                    ship = load_fleet(fleet_path)
                vehicle_values(vehicles, objective)
                config = ExperimentConfig(
                    data=data_path, fleet=fleet_path, objective=objective,
                    algorithm=algorithms[0] if len(algorithms) == 1 else "all",
                    selected_algorithms=algorithms if len(algorithms) > 1 else None,
                    seeds=seeds, load_probability=load_probability,
                    output=run_dir, **options,
                )
                total = len(algorithms) * len(seeds)
                progress = st.progress(0, text=f"0/{total} eksekusi selesai")
                completed = 0
                with st.status("Menjalankan eksperimen", expanded=True) as status:
                    def on_event(message: str) -> None:
                        nonlocal completed
                        status.write(message)
                        if message.startswith("Selesai "):
                            completed += 1
                            progress.progress(completed / total,
                                              text=f"{completed}/{total} eksekusi selesai")

                    run_experiments(config, on_event=on_event)
                    run_dir.joinpath("input_data.json").write_bytes(data_path.read_bytes())
                    if fleet_path is not None:
                        run_dir.joinpath("input_fleet.json").write_bytes(fleet_path.read_bytes())
                    manifest = {
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "mode": mode, "algorithms": algorithms, "seeds": seeds,
                        "objective": objective, "vehicle_count": len(vehicles),
                        "ship_count": len(ships_by_id(ship)),
                        "dataset_name": uploaded_data.name if uploaded_data else "data/data.json",
                    }
                    run_dir.joinpath("run_manifest.json").write_text(
                        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                    status.update(label="Eksperimen selesai", state="complete", expanded=False)
            st.session_state["last_run"] = str(run_dir)
            st.success("Hasil tersimpan. Pilih eksekusi pada bagian detail untuk menjelajahinya.")
        except (OSError, ValueError, ImportError, KeyError, TypeError) as exc:
            st.error(f"Eksperimen tidak dapat dijalankan: {exc}")

    active_run = selected_run or (Path(st.session_state["last_run"])
                                  if "last_run" in st.session_state else None)
    if active_run is not None:
        try:
            render_run(active_run)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.error(f"Hasil eksperimen tidak dapat dibaca: {exc}")


if __name__ == "__main__":
    main()
