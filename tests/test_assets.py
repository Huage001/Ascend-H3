"""Catch the gate-only student and dangling-checkpoint failures before loading."""

import json
import tempfile
import unittest
from pathlib import Path

from h3_ascend.assets import validate_assets


class AssetContracts(unittest.TestCase):
    def test_gate_only_file_does_not_count_as_fasth3_student(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_model.safetensors"
            header = {"__metadata__": {"format": "fastvideo-lora-v2",
                                      "finetuned_model": "FastVideo/FastVideo-FastH3-4-step-v1"}}
            for layer in range(50):
                header[f"transformer_blocks.{layer}.attn.to_gate_compress.set_weight"] = {
                    "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
                }
            encoded = json.dumps(header).encode()
            path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(4))
            with self.assertRaisesRegex(ValueError, "complete FastH3 student"):
                validate_assets("vsa", directory)

    def test_vdn_directories_do_not_hide_missing_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model_spec.json").write_text("{}")
            (root / "metadata.json").write_text("{}")
            (root / "adapters/default").mkdir(parents=True)
            (root / "adapters/turbo").mkdir(parents=True)
            (root / "linear_branch").mkdir()
            (root / "adapters/default/adapter_model.safetensors").symlink_to(root / "absent")
            with self.assertRaisesRegex(ValueError, "complete official"):
                validate_assets("vdn", directory)
