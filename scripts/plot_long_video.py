"""Render measured single-card H3 long-video results as PNG, SVG and PDF.

No example performance values are supplied. Incomplete or mismatched matrices
are rejected; bars use a separate profiled request, headline times use medians
of uninstrumented end-to-end requests.
"""

import argparse
import json
import math
import statistics
from pathlib import Path

HARDWARE = ("Ascend 910B3", "RTX PRO 6000")
METHODS = (("full", False), ("vsa", False), ("veda", False), ("vdn", False), ("spark", False))
LABELS = ("Full + LightX2V8", "VSA · FastH3 V2 · 80%", "Veda + LightX2V8 · 90%", "VDN · official", "Spark-H3 + LightX2V8")


def validate(data):
    if (data["width"], data["height"], data["frames"], data["devices"]) != (1344, 768, 345, 1):
        raise ValueError("Figure requires 1344×768, 345 native frames, one card")
    rows = {}
    policies = set()
    seeds = set()
    for row in data["records"]:
        key = (row["hardware"], row["precision"], row["method"], row["baseline"])
        if key in rows:
            raise ValueError(f"Duplicate record: {key}")
        if not row["validated"] or row["contention_detected"]:
            raise ValueError(f"Invalid or contaminated request: {key}")
        if row.get("matrix_id") != "unified8-v1" or not isinstance(row.get("benchmark_seed"), int):
            raise ValueError(f"A newly measured unified8 request and recorded seed are required: {key}")
        seeds.add(row["benchmark_seed"])
        samples = row["e2e_seconds"]
        if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
            raise ValueError(f"End-to-end measurements required: {key}")
        expected = 8
        if row["method"] == "vsa" and (
            row.get("student_profile") != "fasth3-v2"
            or row.get("nominal_video_sparsity") != .8
        ):
            raise ValueError(f"Figure requires the V2 student and 80% VSA, not legacy v1: {key}")
        if row["actual_steps"] != expected:
            raise ValueError(f"Wrong model/schedule: {key}")
        student = "fasth3-v2" if row["method"] == "vsa" else "vdn-official8" if row["method"] == "vdn" else "lightx2v8"
        if row.get("student_profile") != student:
            raise ValueError(f"Wrong eight-step student: {key}")
        profile = row["profile"]
        spans = (profile["attention_seconds"], profile["ffn_seconds"])
        if any(not math.isfinite(x) or x < 0 for x in spans):
            raise ValueError(f"Invalid profiling spans: {key}")
        if profile["actual_forwards"] != expected:
            raise ValueError(f"The independent profile must retain all {expected} updates: {key}")
        if not math.isfinite(profile["dit_wall_seconds"]) or sum(spans) > profile["dit_wall_seconds"]:
            raise ValueError(f"Overlapping or invalid top-level profiling spans: {key}")
        if not row["offload"]:
            raise ValueError("This matched-Offload figure requires an explicit Offload marker")
        policies.add(row["offload_policy"])
        rows[key] = row
    expected = {(h, p, m, b) for h in HARDWARE for p in ("bf16", "int8") for m, b in METHODS
                if not b}
    if set(rows) != expected:
        raise ValueError(f"Incomplete matrix; missing={sorted(expected - set(rows))}")
    if len(policies) != 1:
        raise ValueError("Both hardware platforms and precisions must use the same Offload policy")
    if len(seeds) != 1:
        raise ValueError("All methods and hardware platforms must use the same reviewed seed")
    return rows, policies.pop()


def render(data, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, policy = validate(data)
    sample_counts = sorted({len(row["e2e_seconds"]) for row in rows.values()})
    sample_note = "N=" + "/".join(map(str, sample_counts)) + " formal requests per cell"
    colors = ("#a63d2a", "#3e8296", "#ee9650")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(15.5, 14.5), facecolor="#fffdf8")
    fig.text(.055, .958, "H3 · 1344 × 768 · 345 frames", fontsize=25, color="#20201e")
    fig.text(.055, .934, "SINGLE CARD  /  BF16 vs W8A8 INT8  /  MATCHED OFFLOAD", fontsize=10,
             color="#126f66", weight="bold")
    grids = fig.add_gridspec(3, 2, left=.055, right=.95, bottom=.10, top=.91,
                            height_ratios=(.38, 1, 1), wspace=.18, hspace=.34)
    for col, hardware in enumerate(HARDWARE):
        baseline = statistics.median(rows[(hardware, "bf16", "full", False)]["e2e_seconds"])
        candidates = [r for k, r in rows.items() if k[0] == hardware]
        best = min(candidates, key=lambda r: statistics.median(r["e2e_seconds"]))
        fastest = statistics.median(best["e2e_seconds"])
        top = fig.add_subplot(grids[0, col])
        top.axis("off")
        top.text(0, 1.03, hardware, fontsize=15, weight="bold")
        top.text(0, .40, f"{baseline / 60:.2f} min", fontsize=29, fontfamily="DejaVu Serif")
        top.text(.56, .40, f"{fastest / 60:.2f} min", fontsize=29, color="#126f66",
                 fontfamily="DejaVu Serif")
        top.text(0, .08, "Full · BF16 · 8 steps", color="#64645e", fontsize=9)
        best_label = LABELS[[m for m, _ in METHODS].index(best["method"])]
        top.text(.56, .08, f"{best_label} · {best['precision'].upper()}", color="#126f66", fontsize=9)
        top.text(.56, -.17, f"{baseline / fastest:.2f}× vs BF16 Full8", color="#126f66", fontsize=9)
        xmax = max(r["profile"]["dit_wall_seconds"] / r["profile"]["actual_forwards"] for r in candidates) * 1000 / 50
        for row_index, precision in enumerate(("bf16", "int8"), 1):
            ax = fig.add_subplot(grids[row_index, col], facecolor="#fffdf8")
            ax.set_title(precision.upper() + "  ·  Offload", loc="left", fontsize=14, pad=24)
            for index, ((method, original), label) in enumerate(zip(METHODS, LABELS)):
                item = rows[(hardware, precision, method, original)]
                reference = statistics.median(rows[(hardware, precision, "full", False)]["e2e_seconds"])
                seconds = statistics.median(item["e2e_seconds"])
                profile = item["profile"]
                stages = (profile["attention_seconds"], profile["ffn_seconds"],
                          profile["dit_wall_seconds"] - profile["attention_seconds"] - profile["ffn_seconds"])
                unit = 1000 / (profile["actual_forwards"] * 50)
                left = 0
                for stage, color in zip(stages, colors):
                    ax.barh(index, stage * unit, left=left, height=.31, color=color)
                    left += stage * unit
                y = index - .30
                ax.text(0, y, f"{label} · {item['actual_steps']} steps", va="bottom", fontsize=9,
                        weight="bold", color="#252523")
                text = f"{seconds / 60:.2f} min  /  {reference / seconds:.2f}×"
                if precision == "int8":
                    unquantized = statistics.median(rows[(hardware, "bf16", method, original)]["e2e_seconds"])
                    text += f"  /  Q {unquantized / seconds:.2f}×"
                ax.text(xmax, y, text,
                        va="bottom", ha="right", fontsize=9, color="#126f66")
            ax.set_ylim(len(METHODS) - .4, -.9)
            ax.set_xlim(0, xmax * 1.01)
            ax.set_yticks([])
            ax.set_xlabel("Separate DiT profile · mean milliseconds per block", fontsize=9)
            ax.tick_params(axis="x", labelsize=8, colors="#68685e")
            for spine in ("top", "left", "right"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_color("#d5c8b7")
    fig.legend(handles=[Patch(color=c, label=label) for c, label in zip(
        colors, ("Attention module", "FFN", "Other DiT work incl. unhidden Offload"))],
        loc="lower left", bbox_to_anchor=(.05, .052), ncol=3, frameon=False, fontsize=9)
    fig.text(.055, .039, "Headline times: median uninstrumented request → completed MP4; "
             f"{sample_note}. Reference: same-device, same-precision Full8. "
             "Q = same-method BF16 / INT8.", fontsize=8, color="#626257")
    fig.text(.055, .028, f"Bars: separate DiT profiling, normalized over actual forwards × 50 blocks; no E2E extrapolation. "
             f"Offload policy: {policy}.", fontsize=8, color="#626257")
    fig.text(.055, .015, "All five methods and Full baseline: eight actual forwards, one common reviewed seed. Each student retains its trained schedule.", fontsize=8, color="#626257")
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output.with_suffix("." + suffix), dpi=190, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.input.read_text()), args.output)
