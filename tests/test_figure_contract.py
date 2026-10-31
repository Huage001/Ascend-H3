"""Keep eight-step figures and BF16 Full references consistent across precision."""

import copy
import importlib.util
import unittest
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "long_video_figure", Path(__file__).resolve().parents[1] / "scripts/plot_long_video.py"
)
figure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figure)


def fixture():
    rows = []
    for hardware in figure.HARDWARE:
        for precision in ("bf16", "int8"):
            for method, baseline in figure.METHODS:
                rows.append({
                    "matrix_id": "unified8-v1", "benchmark_seed": 7,
                    "hardware": hardware, "precision": precision, "method": method,
                    "baseline": baseline, "actual_steps": 8, "validated": True,
                    "student_profile": "fasth3-v2" if method == "vsa" else "vdn-official8" if method == "vdn" else "lightx2v8",
                    "nominal_video_sparsity": .8 if method == "vsa" else .9,
                    "dormant_input_vision_file_backed": method in ("vsa", "vdn") and precision == "bf16",
                    "contention_detected": False, "offload": True, "offload_policy": "test",
                    "e2e_seconds": [60.0], "profile": {"actual_forwards": 8,
                        "dit_wall_seconds": 40.0, "attention_seconds": 20.0, "ffn_seconds": 10.0},
                })
    return {"width": 1344, "height": 768, "frames": 345, "devices": 1, "records": rows}


class FigureContracts(unittest.TestCase):
    def test_int8_speedup_uses_bf16_full_not_int8_full(self):
        data = fixture()
        for row in data["records"]:
            if row["method"] == "full":
                row["e2e_seconds"] = [120.0 if row["precision"] == "bf16" else 90.0]
            elif row["method"] == "veda" and row["precision"] == "int8":
                row["e2e_seconds"] = [30.0]
        rows, _ = figure.validate(data)
        for hardware in figure.HARDWARE:
            self.assertEqual(figure.speedup_vs_bf16_full(rows, rows[(hardware, "int8", "veda", False)]), 4.0)
            self.assertAlmostEqual(figure.speedup_vs_bf16_full(rows, rows[(hardware, "int8", "full", False)]), 4 / 3)
            self.assertEqual(figure.speedup_vs_bf16_full(rows, rows[(hardware, "bf16", "full", False)]), 1.0)

    def test_complete_native_student_matrix(self):
        rows, policy = figure.validate(fixture())
        self.assertEqual(len(rows), 20)
        self.assertEqual(policy, "test")

    def test_invalid_comparison_cannot_render(self):
        cases = []
        base = fixture()
        missing = copy.deepcopy(base)
        missing["records"].pop()
        cases.append(missing)
        legacy = copy.deepcopy(base)
        next(row for row in legacy["records"] if row["method"] == "vsa")["student_profile"] = "fasth3-v1"
        cases.append(legacy)
        unmatched_vision = copy.deepcopy(base)
        next(row for row in unmatched_vision["records"] if row["method"] == "vsa" and row["precision"] == "bf16")["dormant_input_vision_file_backed"] = False
        cases.append(unmatched_vision)
        wrong_seed = copy.deepcopy(base)
        wrong_seed["records"][0]["benchmark_seed"] = 123
        cases.append(wrong_seed)
        for field, value in (("actual_steps", 50), ("offload_policy", "other"),
                             ("contention_detected", True), ("devices", 2)):
            bad = copy.deepcopy(base)
            if field == "devices":
                bad[field] = value
            else:
                bad["records"][0][field] = value
            cases.append(bad)
        for bad in cases:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                figure.validate(bad)
