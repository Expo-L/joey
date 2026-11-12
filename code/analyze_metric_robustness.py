"""Metric-robustness analysis for the formal held-out run.

Revision experiment R2.  The reviewer questions whether the reported temporal
stability effect is an artefact of the single RGB-L1 jitter proxy.  No new
diffusion sampling is required: run 038 stores every generated background, so
the same sequences can be scored with two further measures.

Reported per sequence (background frames only, no composited graphics):

``rgb_l1``          mean absolute RGB difference between adjacent frames,
                    scaled to [0, 1] -- reimplements the manuscript's J so the
                    stored metrics.json values can be cross-checked.
``ssim``            mean structural similarity between adjacent grayscale
                    frames (7x7 uniform window, standard constants).  Higher is
                    more stable.
``highfreq_l1``     mean absolute difference between the high-pass images of
                    adjacent frames.  Subtracting a Gaussian-blurred copy
                    suppresses smooth coherent motion and isolates flicker in
                    texture and edges.  Lower is more stable.
``p95_pair_l1``     95th percentile of the per-pair RGB-L1 values, i.e. the
                    worst frames rather than the average.

Paired effects use the same baseline/treatment unit (asset x seed, n = 30) and
the same resampling seed as the manuscript (20260903).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter

RESAMPLE_SEED = 20260903


def load_frames(path: Path) -> np.ndarray:
    files = sorted(path.glob("f*.png"))
    if not files:
        raise FileNotFoundError(f"no background frames in {path}")
    return np.stack([np.asarray(Image.open(file).convert("RGB"), dtype=np.float32) / 255.0 for file in files])


def rgb_l1(frames: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(frames, axis=0))))


def pair_l1(frames: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2, 3))


def gray(frames: np.ndarray) -> np.ndarray:
    return frames @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def adjacent_ssim(frames: np.ndarray) -> float:
    images = gray(frames)
    window = 7
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    values = []
    for first, second in zip(images[:-1], images[1:]):
        a = first * 255.0
        b = second * 255.0
        mu_a = uniform_filter(a, window)
        mu_b = uniform_filter(b, window)
        var_a = uniform_filter(a * a, window) - mu_a * mu_a
        var_b = uniform_filter(b * b, window) - mu_b * mu_b
        cov = uniform_filter(a * b, window) - mu_a * mu_b
        numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
        values.append(float(np.mean(numerator / denominator)))
    return float(np.mean(values))


def highfreq_l1(frames: np.ndarray) -> float:
    high = frames - np.stack([gaussian_filter(frame, sigma=(1.5, 1.5, 0)) for frame in frames])
    return float(np.mean(np.abs(np.diff(high, axis=0))))


def paired_stats(differences: np.ndarray) -> dict[str, float]:
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(RESAMPLE_SEED)
    bootstrap = np.asarray(
        [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(5000)]
    )
    observed = abs(float(differences.mean()))
    draws = 100_000
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(differences)), replace=True)
    count = int((np.abs((signs * differences[None, :]).mean(axis=1)) >= observed - 1e-12).sum())
    return {
        "n_pairs": int(len(differences)),
        "mean_improvement_baseline_minus_treatment": float(differences.mean()),
        "bootstrap_ci95_low": float(np.percentile(bootstrap, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(bootstrap, 97.5)),
        "sign_flip_p": float((count + 1) / (draws + 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", default="independent_nolora,independent_lora,ours_full_lora,ours_late_lora")
    parser.add_argument("--baseline", default="independent_lora")
    parser.add_argument("--treatment", default="ours_late_lora")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]

    rows: list[dict] = []
    samples_root = args.run / "samples"
    for sequence_dir in sorted(samples_root.iterdir()):
        if not sequence_dir.is_dir():
            continue
        for seed_dir in sorted(sequence_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = int(seed_dir.name.replace("seed_", ""))
            for condition in conditions:
                frame_dir = seed_dir / condition / "backgrounds"
                if not frame_dir.is_dir():
                    continue
                frames = load_frames(frame_dir)
                stored = None
                metrics_path = seed_dir / condition / "metrics.json"
                if metrics_path.exists():
                    stored = json.loads(metrics_path.read_text(encoding="utf-8")).get("background_jitter")
                record = {
                    "sample_id": sequence_dir.name,
                    "seed": seed,
                    "condition": condition,
                    "rgb_l1": rgb_l1(frames),
                    "ssim": adjacent_ssim(frames),
                    "highfreq_l1": highfreq_l1(frames),
                    "p95_pair_l1": float(np.percentile(pair_l1(frames), 95)),
                    "stored_background_jitter": stored,
                }
                record["rgb_l1_minus_stored"] = (
                    float(record["rgb_l1"] - stored) if stored is not None else None
                )
                rows.append(record)
    if not rows:
        raise RuntimeError("no frames found")

    import csv

    fields = list(rows[0].keys())
    with (args.out / "per_sequence_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    max_deviation = max(
        abs(float(row["rgb_l1_minus_stored"])) for row in rows if row["stored_background_jitter"] is not None
    )

    index = {(row["sample_id"], row["seed"], row["condition"]): row for row in rows}
    pairs = sorted({(row["sample_id"], row["seed"]) for row in rows})
    summary: dict[str, object] = {
        "run": str(args.run),
        "n_sequences": len(rows),
        "reimplementation_max_abs_deviation_from_stored_jitter": max_deviation,
        "conditions": {},
        "paired_effects": {},
    }
    lower_is_better = {"rgb_l1": True, "highfreq_l1": True, "p95_pair_l1": True, "ssim": False}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        summary["conditions"][condition] = {
            metric: {
                "mean": float(np.mean([row[metric] for row in subset])),
                "std": float(np.std([row[metric] for row in subset], ddof=1)),
            }
            for metric in lower_is_better
        }
    for metric, lower_better in lower_is_better.items():
        differences = []
        for sample_id, seed in pairs:
            base = index.get((sample_id, seed, args.baseline))
            treatment = index.get((sample_id, seed, args.treatment))
            if base is None or treatment is None:
                continue
            difference = float(base[metric]) - float(treatment[metric])
            differences.append(difference if lower_better else -difference)
        summary["paired_effects"][metric] = {
            "orientation": "positive favours treatment",
            **paired_stats(np.asarray(differences)),
        }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
