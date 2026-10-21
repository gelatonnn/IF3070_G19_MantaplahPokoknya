import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from src.Core.data_loader import load_data
from src.Core.evaluation import evaluate


ROOT = Path(__file__).resolve().parents[1]

SHIP = {"w": 4, "l": 6, "max_capacity": 30}
VEHICLES = [
    {"id": "A", "w": 2, "l": 3, "shipping_fee": 40, "weight": 12, "eta": 0},
    {"id": "B", "w": 2, "l": 2, "shipping_fee": 25, "weight": 8, "eta": 0},
    {"id": "C", "w": 1, "l": 2, "shipping_fee": 10, "weight": 3, "eta": 0},
    {"id": "D", "w": 2, "l": 4, "shipping_fee": 50, "weight": 15, "eta": 0},
]


class ExperimentCLITests(unittest.TestCase):
    def test_all_variants_export_reproducible_metadata_and_valid_solutions(self):
        with TemporaryDirectory() as folder:
            fixture = Path(folder) / "fixture.json"
            fixture.write_text(
                json.dumps({"ship": SHIP, "vehicles": VEHICLES}),
                encoding="utf-8",
            )
            out = Path(folder) / "results"
            command = [
                sys.executable, "-m", "src.main",
                "--data", str(fixture), "--output", str(out),
                "--seeds", "42", "--max-restarts", "1", "--max-sideways", "2", "--no-plots",
            ]
            run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            files = list(out.glob("*/seed_42/result.json"))
            self.assertEqual(len(files), 4)
            ship, vehicles = load_data(fixture)
            digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
            initials = []
            for file in files:
                result = json.loads(file.read_text())
                self.assertEqual(evaluate(result["final_state"], vehicles, ship),
                                 result["final_metrics"])
                self.assertTrue(result["final_metrics"]["is_valid"])
                self.assertEqual(result["experiment"]["dataset_sha256"], digest)
                initials.append(result["initial_state"])
            self.assertTrue(all(initial == initials[0] for initial in initials))
            self.assertTrue((out / "summary.csv").is_file())
            before = {p: p.read_bytes() for p in files}
            repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(before, {p: p.read_bytes() for p in files})

    def test_invalid_options_do_not_start_experiment(self):
        with TemporaryDirectory() as folder:
            out = Path(folder) / "unused"
            for options in (["--max-restarts", "-1"], ["--max-sideways", "-1"],
                            ["--seeds", "1", "1"], ["--load-probability", "2"]):
                with self.subTest(options=options):
                    run = subprocess.run(
                        [sys.executable, "-m", "src.main", "--output", str(out), *options],
                        cwd=ROOT, capture_output=True, text=True,
                    )
                    self.assertNotEqual(run.returncode, 0)
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
