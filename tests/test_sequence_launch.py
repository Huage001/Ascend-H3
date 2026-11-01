"""Public TP/SP choices select one DiT parallel dimension and safe transport defaults."""

import argparse
import tempfile
import unittest
from unittest.mock import patch

from h3_ascend.cli import command, serve_environment


def arguments(**changes):
    values = dict(
        method="full",
        parallel="sp",
        devices=4,
        model="model",
        host="127.0.0.1",
        port=8000,
        precision="int8",
        lora=None,
        offload="none",
        steps=8,
    )
    values.update(changes)
    return argparse.Namespace(**values)


class SequenceLaunch(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        environment = patch.dict("os.environ", {"H3_ASCEND_CACHE": temporary.name}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def test_all_attention_methods_can_select_sp_without_private_flags(self):
        with patch.dict("os.environ", {}):
            for method in ("full", "veda", "spark", "vsa", "vdn"):
                args = arguments(method=method, lora="official" if method == "vdn" else None)
                argv = command(args)
                self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "1")
                self.assertEqual(argv[argv.index("--usp") + 1], "4")
                environment = serve_environment(args)
                self.assertEqual(environment["H3_ASCEND_INPUT_EXCHANGE"], "1")
                if method == "vdn":
                    self.assertEqual(environment["H3_ASCEND_VDN_SP"], "1")

    def test_tp_and_ulysses_alias_remain_supported(self):
        for mode, tp, sp in (("tp", "4", "1"), ("sp", "1", "4"), ("ulysses", "1", "4")):
            argv = command(arguments(parallel=mode))
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], tp)
            self.assertEqual(argv[argv.index("--usp") + 1], sp)

    def test_native_input_transport_requires_qualified_resident_int8_shapes(self):
        with patch.dict("os.environ", {}):
            for world in (1, 2, 4, 8):
                result = serve_environment(arguments(devices=world))
                self.assertEqual(
                    result["H3_ASCEND_INPUT_EXCHANGE"], "1" if world in (2, 4) else "0"
                )
            for changes in (
                dict(precision="bf16"),
                dict(parallel="tp"),
                dict(offload="layer"),
                dict(offload="module"),
            ):
                self.assertEqual(
                    serve_environment(arguments(**changes))["H3_ASCEND_INPUT_EXCHANGE"], "0"
                )

    def test_original_sp_transport_can_be_requested_for_reference(self):
        with patch.dict("os.environ", {"H3_ASCEND_INPUT_EXCHANGE": "0"}):
            environment = serve_environment(arguments())
            self.assertEqual(environment["H3_ASCEND_INPUT_EXCHANGE"], "0")
            self.assertEqual(environment["H3_ASCEND_SEQUENCE_COMM"], "auto")


if __name__ == "__main__":
    unittest.main()
