"""Figure 4: gating-schedule sweep for the temporal residual.

Reads the sweep run (``runs/gating_sweep_039``) and the original formal run
(``runs/formal_experiment_038``) and plots mean background jitter against the
number of gated sampler calls K.  K = 0 is the independent LoRA baseline and
K = 11 is the full-injection ablation.

Usage:
    python scripts/make_figure4_gating_sweep.py \
        --sweep runs/gating_sweep_039 \
        --formal runs/formal_experiment_038 \
        --out manuscript_q4/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASELINE_CONDITION = "independent_lora"
FULL_CONDITION = "ours_full_lora"
LATE_CONDITION = "ours_late_lora"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--column", default="background_jitter")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_style()

    sweep_rows = pd.read_csv(args.sweep / "per_sample_metrics.csv")
    formal_rows = pd.read_csv(args.formal / "per_sample_metrics.csv")

    points: list[dict[str, float]] = []

    def add_point(k: int, values: np.ndarray, label: str, kind: str) -> None:
        points.append(
            {
                "k": k,
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "sem": float(values.std(ddof=1) / np.sqrt(len(values))),
                "n": int(len(values)),
                "label": label,
                "kind": kind,
            }
        )

    add_point(
        0,
        formal_rows.loc[formal_rows.condition == BASELINE_CONDITION, args.column].to_numpy(),
        "no residual",
        "baseline",
    )
    for condition in sorted(sweep_rows.condition.unique()):
        if "shift" in condition:
            continue
        k = int(condition.split("_")[0].replace("last", ""))
        add_point(
            k,
            sweep_rows.loc[sweep_rows.condition == condition, args.column].to_numpy(),
            f"{k} call" + ("s" if k > 1 else ""),
            "sweep",
        )
    add_point(
        11,
        formal_rows.loc[formal_rows.condition == FULL_CONDITION, args.column].to_numpy(),
        "all 11 calls",
        "full",
    )

    points.sort(key=lambda item: item["k"])
    ks = np.asarray([item["k"] for item in points], dtype=float)
    means = np.asarray([item["mean"] for item in points], dtype=float)
    sds = np.asarray([item["sd"] for item in points], dtype=float)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    baseline = means[0]
    ax.axhline(baseline, color="#9a9a9a", lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.text(
        ks[-1] + 0.15,
        baseline,
        "independent LoRA\n(no residual)",
        va="center",
        ha="left",
        fontsize=6.5,
        color="#5f5f5f",
    )

    ax.errorbar(ks, means, yerr=sds, color="#1f4e79", lw=1.2, ms=0, capsize=2.2, zorder=3)
    for point in points:
        if point["k"] == 3:
            ax.plot(point["k"], point["mean"], marker="D", ms=5.4, mfc="white", mec="#8a3b3b", mew=1.2, zorder=5)
            continue
        if point["k"] == 8:
            ax.plot(point["k"], point["mean"], marker="*", ms=10.0, mfc="#8a3b3b", mec="#8a3b3b", mew=0.8, zorder=5)
            continue
        marker = {"baseline": "o", "sweep": "o", "full": "s"}[point["kind"]]
        face = {"baseline": "white", "sweep": "#1f4e79", "full": "white"}[point["kind"]]
        ax.plot(
            point["k"],
            point["mean"],
            marker=marker,
            ms=5.2,
            mfc=face,
            mec="#1f4e79",
            mew=1.1,
            zorder=4,
        )
    full = next(item for item in points if item["k"] == 11)
    ax.annotate(
        "all 11 calls:\nincludes the three\nhighest-noise calls",
        xy=(full["k"], full["mean"]),
        xytext=(full["k"] - 5.2, full["mean"] + 0.0035),
        fontsize=6.2,
        color="#8a3b3b",
        ha="right",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#8a3b3b"},
    )

    ax.set_xticks(ks)
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.set_xlabel("Gated sampler calls $K$ (out of 11)")
    ax.set_ylabel("Background jitter $J$")
    ax.set_xlim(-0.6, ks[-1] + 2.6)
    ax.tick_params(length=2.5, width=0.7)

    fig.savefig(args.out / "fig4_gating_sweep.pdf")
    fig.savefig(args.out / "fig4_gating_sweep.png", dpi=600)
    fig.savefig(args.out / "fig4_gating_sweep.svg")
    plt.close(fig)

    table = pd.DataFrame(points)[["k", "mean", "sd", "n", "label"]]
    table.to_csv(args.out / "fig4_gating_sweep_source.csv", index=False)
    paired = json.loads((args.sweep / "paired_effects.json").read_text(encoding="utf-8"))
    (args.out / "fig4_gating_sweep_paired.json").write_text(
        json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(table.to_string(index=False))
    print("\nbaseline mean:", f"{baseline:.6f}")
    for item in points[1:]:
        print(
            f"K={int(item['k']):2d}  mean={item['mean']:.6f}  "
            f"delta_vs_baseline={baseline - item['mean']:+.6f}  "
            f"relative={100 * (baseline - item['mean']) / baseline:+.2f}%"
        )


if __name__ == "__main__":
    main()
