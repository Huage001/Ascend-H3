"""Keep four-step figures from mixing schedules, devices or Offload policies."""

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
                    "hardware": hardware, "precision": precision, "method": method,
                    "baseline": baseline, "actual_steps": 4, "validated": True,
                    "contention_detected": False, "offload": True, "offload_policy": "test",
                    "e2e_seconds": [60.0], "profile": {"actual_forwards": 4,
                        "dit_wall_seconds": 40.0, "attention_seconds": 20.0, "ffn_seconds": 10.0},
                })
    return {"width": 1344, "height": 768, "frames": 345, "devices": 1, "records": rows}


class FigureContracts(unittest.TestCase):
    def test_complete_four_step_matrix(self):
        rows, policy = figure.validate(fixture())
        self.assertEqual(len(rows), 16)
        self.assertEqual(policy, "test")

    def test_invalid_comparison_cannot_render(self):
        cases = []
        base = fixture()
        missing = copy.deepcopy(base)
        missing["records"].pop()
        cases.append(missing)
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
