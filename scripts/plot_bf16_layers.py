"""Render resident BF16 latency and mirrored Full-relative speedup panels."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

METHODS = ("full", "vsa", "veda", "vdn", "spark")
NAMES = ("Full", "VSA V2 (80%)", "Veda (90%)", "VDN", "Spark-H3 (90%)")
# PRO 6000 is now above Ascend; both swap their former colors.
HARDWARE = (("cuda", "RTX PRO 6000", -0.18), ("ascend", "Ascend 910B3", 0.18))
COLORS = {"cuda": "#15736d", "ascend": "#dc8c50"}
TEXT_COLORS = {"cuda": "#11635e", "ascend": "#985123"}
BACKGROUND = "#fffdf7"
INK = "#272a2b"
MUTED = "#6c6b65"


def render(source: Path, output: Path) -> None:
    data = json.loads(source.read_text())
    if data.get("matched_core_pairs") != 10 or data["protocol"]["offload"]:
        raise ValueError("Require ten matched, resident BF16 core comparisons")
    rows = {(row["frames"], row["method"]): row for row in data["records"]}
    for frames in (124, 345):
        for method in METHODS:
            for hardware, _, _ in HARDWARE:
                for scope in ("core", "dit_block"):
                    record = rows[frames, method]["hardware"][hardware]
                    baseline = rows[frames, "full"]["hardware"][hardware][scope + "_ms"]
                    latency = record[scope + "_ms"]
                    speedup = record[scope + "_speedup"]
                    if not latency > 0 or not math.isfinite(latency):
                        raise ValueError("Latency must be positive and finite")
                    if not math.isclose(baseline / latency, speedup, rel_tol=1e-9):
                        raise ValueError("Speedup must use this device's same-shape Full baseline")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": INK,
        "svg.fonttype": "none", "svg.hashsalt": "ascend-h3-bf16-layers",
    })
    fig = plt.figure(figsize=(20, 11.8), facecolor=BACKGROUND)
    grid = fig.add_gridspec(2, 2, left=.112, right=.985, top=.82, bottom=.10,
                            hspace=.39, wspace=.24)
    for row_index, frames in enumerate((124, 345)):
        for col_index, (scope, title) in enumerate((
            ("core", "Attention core only"), ("dit_block", "Complete DiT block")
        )):
            pair = grid[row_index, col_index].subgridspec(1, 2, width_ratios=(1.5, 1), wspace=.13)
            latency_axis = fig.add_subplot(pair[0, 0])
            speed_axis = fig.add_subplot(pair[0, 1])
            maximum = max(rows[frames, m]["hardware"][h][scope + "_ms"]
                          for m in METHODS for h, _, _ in HARDWARE)
            latency_axis.set_xlim(0, maximum * 1.26)
            # Separate reversed scale: every speedup bar starts at the right-hand zero.
            speed_axis.set_xlim(8, 0)
            speed_axis.set_xticks((8, 6, 4, 2, 1, 0))
            speed_axis.set_xticklabels(("8×", "6×", "4×", "2×", "1×", "0"))
            speed_axis.axvline(1, color="#9b998f", lw=.8, ls=(0, (3, 3)), zorder=1)
            for axis in (latency_axis, speed_axis):
                axis.set_facecolor(BACKGROUND)
                axis.set_ylim(4.62, -.62)
                axis.grid(axis="x", color="#d8d8d2", alpha=.55, lw=.6)
                axis.set_axisbelow(True)
                axis.spines[["top", "left"]].set_visible(False)
                axis.spines["bottom"].set_color("#cfc8ba")
                axis.tick_params(axis="y", length=0)
                axis.tick_params(axis="x", labelsize=8.5, length=3)
                for y in (1, 3):
                    axis.axhspan(y - .48, y + .48, color="#f3f0e8", alpha=.55, zorder=0)
            latency_axis.spines["right"].set_visible(False)
            speed_axis.spines["right"].set_color("#aaa69a")
            latency_axis.set_yticks(range(5), NAMES)
            latency_axis.tick_params(axis="y", labelsize=10)
            speed_axis.set_yticks([])
            latency_axis.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=(1, 2, 2.5, 5, 10)))
            for hardware, _, offset in HARDWARE:
                for method_index, method in enumerate(METHODS):
                    record = rows[frames, method]["hardware"][hardware]
                    latency = record[scope + "_ms"]
                    speedup = record[scope + "_speedup"]
                    y = method_index + offset
                    latency_axis.barh(y, latency, height=.30, color=COLORS[hardware], zorder=3)
                    speed_axis.barh(y, speedup, height=.30, color=COLORS[hardware], zorder=3)
                    latency_axis.text(latency + maximum * .015, y, f"{latency:.1f}",
                                      ha="left", va="center", fontsize=8.8,
                                      color=TEXT_COLORS[hardware])
                    speed_axis.text(speedup + .12, y, f"{speedup:.2f}×",
                                    ha="right", va="center", fontsize=8.8,
                                    color=TEXT_COLORS[hardware])
            position = latency_axis.get_position()
            fig.text(position.x0, position.y1 + .045, f"{frames} frames · {title}",
                     ha="left", fontsize=13, fontweight="bold")
            latency_axis.text(0, 1.025, "Latency (ms) →", transform=latency_axis.transAxes,
                              fontsize=9.5, color=MUTED, va="bottom")
            speed_axis.text(1, 1.025, "← Speedup vs Full", transform=speed_axis.transAxes,
                            fontsize=9.5, color=MUTED, ha="right", va="bottom")
            latency_axis.set_xlabel("Shorter is faster", fontsize=9, labelpad=8)
            speed_axis.set_xlabel("Longer is faster · dashed line = 1×", fontsize=9, labelpad=8)

    fig.text(.035, .966, "H3 at 1344 × 768: resident BF16 layer measurements",
             fontsize=23, fontweight="bold", va="top")
    fig.text(.035, .923,
             "Layers 0 / 24 / 49 × 3 formal runs · sum / 9 · identical frozen synthetic inputs · real student weights",
             fontsize=11, color=MUTED)
    legend = [Patch(facecolor=COLORS[key], label=label) for key, label, _ in HARDWARE]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(.985, .944),
               frameon=False, ncol=2, handlelength=2.0, columnspacing=2.5, fontsize=11)
    fig.text(.035, .034,
             "Speedup = same-device Full latency / method latency. Both devices use BF16; these are layer measurements, not E2E times.",
             fontsize=10, color=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor=BACKGROUND)
    fig.savefig(output.with_suffix(".svg"), facecolor=BACKGROUND, metadata={"Date": None})
    svg = output.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "benchmarks/bf16-three-layer.json")
    parser.add_argument("--output", type=Path, default=root / "docs/figures/bf16-three-layer")
    args = parser.parse_args()
    render(args.input, args.output)
    print(f"Wrote {args.output.with_suffix('.png')} and SVG")


if __name__ == "__main__":
    main()
