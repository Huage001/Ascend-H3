"""Startup preparation follows the resolved model/student schedule."""

import argparse
import unittest
from unittest.mock import patch

from h3_ascend.cli import serve_environment
from h3_ascend.integrations.adaln_precompute import resolve_startup_schedule


class Pipeline:
    default_video_shift = 10.0
    default_audio_shift = 3.0

    def __init__(self, student_steps=None):
        self.student_steps = student_steps

    def _resolve_sigma_positions(self, task, sampling):
        assert task == "t2va"
        self.requested = sampling.num_inference_steps
        count = self.student_steps or self.requested or 50
        positions = tuple(1 - i / count for i in range(count + 1)) if self.student_steps else None
        return positions, count


class StartupSchedule(unittest.TestCase):
    def test_model_and_merged_student_choose_actual_steps(self):
        for steps in (4, 8, 50):
            pipeline = Pipeline(steps)
            contract = resolve_startup_schedule(pipeline)
            self.assertIsNone(pipeline.requested)
            self.assertEqual(contract["num_steps"], steps)
            self.assertEqual(len(contract["base_schedule"]), steps + 1)
            self.assertEqual((contract["flow_shift"], contract["audio_flow_shift"]), (10, 3))

    def test_teacher_uses_explicit_launch_steps_or_native_default(self):
        for requested in (None, 4, 8, 32, 50):
            contract = resolve_startup_schedule(Pipeline(), requested)
            self.assertEqual(contract["num_steps"], requested or 50)
            self.assertIsNone(contract["base_schedule"])

    def test_conflicting_student_and_invalid_counts_fail(self):
        with self.assertRaises(ValueError):
            resolve_startup_schedule(Pipeline(8), 4)
        for count in (0, -1, True, 4.5):
            with self.assertRaises(ValueError):
                resolve_startup_schedule(Pipeline(), count)

    def test_launch_environment_carries_explicit_steps(self):
        with patch.dict("os.environ", {}, clear=True):
            args = argparse.Namespace(method="full", offload="text", steps=32, precompute_adaln=None)
            environment = serve_environment(args)
            self.assertEqual(environment["H3_ASCEND_ADALN_STEPS"], "32")
            self.assertEqual(environment["H3_ASCEND_PRECOMPUTE_ADALN"], "1")
            args.steps = None
            self.assertNotIn("H3_ASCEND_ADALN_STEPS", serve_environment(args))
            args.offload = "layer"
            self.assertEqual(serve_environment(args)["H3_ASCEND_PRECOMPUTE_ADALN"], "0")
            args.precompute_adaln = True
            with self.assertRaises(ValueError):
                serve_environment(args)

    def test_long_request_deadline_preserves_explicit_override(self):
        args = argparse.Namespace(method="full", offload="text", steps=8, precompute_adaln=None)
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(serve_environment(args)["VLLM_OMNI_DLO_DP_WAVE_TIMEOUT"], "7200")
        with patch.dict("os.environ", {"VLLM_OMNI_DLO_DP_WAVE_TIMEOUT": "3600"}, clear=True):
            self.assertEqual(serve_environment(args)["VLLM_OMNI_DLO_DP_WAVE_TIMEOUT"], "3600")
