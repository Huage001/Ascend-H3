"""Apply the reviewed Omni compatibility fixes to the pinned checkout."""

import argparse
import subprocess
from pathlib import Path

REVISION = "45603043133fda3fe486d8e558c226d3391940f0"
PATCHES = (
    "vllm-omni-cached-request-kwonly.patch",
    "vllm-omni-vsa-layerwise-fusion.patch",
    "vllm-omni-h3-encoder-host-load.patch",
    "vllm-omni-h3-community-lora-schedule.patch",
    "vllm-omni-fasth3-v2-online-int8-gates.patch",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omni", type=Path, required=True)
    args = parser.parse_args()
    root = args.omni.resolve(strict=True)
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != REVISION:
        raise SystemExit(f"Expected vLLM-Omni {REVISION}, got {actual}")
    patches = Path(__file__).resolve().parents[1] / "patches"
    for name in PATCHES:
        patch = str(patches / name)
        command = ["git", "-C", str(root), "apply"]
        if (
            subprocess.run(
                [*command, "--reverse", "--check", patch], capture_output=True
            ).returncode
            == 0
        ):
            print(f"Already applied: {name}")
            continue
        subprocess.run([*command, "--check", patch], check=True)
        subprocess.run([*command, patch], check=True)
        print(f"Applied: {name}")


if __name__ == "__main__":
    main()
