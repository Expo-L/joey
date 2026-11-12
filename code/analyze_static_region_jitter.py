"""Separate flicker from intended motion with a static-region jitter measure.

The RGB jitter used in the paper cannot tell an intended change, such as the
procedural foreground moving, from an unintended one, such as a frame-level
flicker. The synthetic sequences come with target frames whose motion plan is
known, so the two can be separated without new diffusion sampling.

For each sequence the target frames are resampled to the rendered resolution and
a per-pixel motion magnitude is computed as the largest absolute change between
consecutive target frames. Pixels below a threshold are, by construction, places
where the plan asks for no change; any variation there in a generated background
is unintended. Jitter restricted to those pixels is reported as static-region
jitter.

The mask is conservative: it can only remove pixels that the plan changes, so a
condition that merely smooths intended motion cannot gain from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

RESAMPLE_SEED = 20260903


def load_gray(path: Path, size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("L").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def static_mask(target_paths: list[Path], size: tuple[int, int], threshold: float) -> np.ndarray:
    frames = np.stack([load_gray(path, size) for path in target_paths])
    motion = np.abs(np.diff(frames, axis=0)).max(axis=0)
    return motion < threshold


def masked_jitter(frames: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    diffs = np.abs(np.diff(frames, axis=0))
    return float(diffs[:, mask].mean())


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
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", default="independent_nolora,independent_lora,ours_full_lora,ours_late_lora")
    parser.add_argument("--baseline", default="independent_lora")
    parser.add_argument("--treatment", default="ours_late_lora")
    parser.add_argument("--threshold", type=float, default=8 / 255)
    parser.add_argument("--eval-start", type=int, default=150)
    parser.add_argument("--eval-count", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = json.loads((args.synthetic / "labels.json").read_text(encoding="utf-8"))
    selected = {label["sample_id"]: label for label in labels[args.eval_start : args.eval_start + args.eval_count]}
    conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]

    rows: list[dict] = []
    for sample_id, label in selected.items():
        sample_dir = args.run / "samples" / sample_id
        if not sample_dir.is_dir():
            continue
        target_paths = [args.synthetic / path for path in label["frame_paths"]]
        size = Image.open(target_paths[0]).size
        for seed_dir in sorted(sample_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = int(seed_dir.name.replace("seed_", ""))
            # Render resolution comes from the stored backgrounds, not the target.
            probe = next((seed_dir / cond / "backgrounds" for cond in conditions if (seed_dir / cond / "backgrounds").is_dir()), None)
            if probe is None:
                continue
            frame_files = sorted(probe.glob("f*.png"))
            if not frame_files:
                continue
            render_size = Image.open(frame_files[0]).size
            mask = static_mask(target_paths, render_size, args.threshold)
            for condition in conditions:
                frame_dir = seed_dir / condition / "backgrounds"
                if not frame_dir.is_dir():
                    continue
                files = sorted(frame_dir.glob("f*.png"))
                frames = np.stack(
                    [np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0 for path in files]
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "seed": seed,
                        "condition": condition,
                        "static_fraction": float(mask.mean()),
                        "static_jitter": masked_jitter(frames, mask),
                        "moving_jitter": masked_jitter(frames, ~mask),
                        "full_jitter": float(np.abs(np.diff(frames, axis=0)).mean()),
                    }
                )

    if not rows:
        raise RuntimeError("no frames found")

    import csv

    with (args.out / "per_sequence_static_jitter.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    index = {(row["sample_id"], row["seed"], row["condition"]): row for row in rows}
    pairs = sorted({(row["sample_id"], row["seed"]) for row in rows})
    summary: dict[str, object] = {
        "run": str(args.run),
        "threshold": args.threshold,
        "mean_static_fraction": float(np.mean([row["static_fraction"] for row in rows])),
        "conditions": {},
        "paired_effects": {},
    }
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        if not subset:
            continue
        summary["conditions"][condition] = {
            key: {
                "mean": float(np.mean([row[key] for row in subset])),
                "std": float(np.std([row[key] for row in subset], ddof=1)),
            }
            for key in ("static_jitter", "moving_jitter", "full_jitter")
        }
    for metric in ("static_jitter", "moving_jitter", "full_jitter"):
        differences = []
        for sample_id, seed in pairs:
            base = index.get((sample_id, seed, args.baseline))
            treatment = index.get((sample_id, seed, args.treatment))
            if base is None or treatment is None:
                continue
            differences.append(float(base[metric]) - float(treatment[metric]))
        if differences:
            summary["paired_effects"][metric] = paired_stats(np.asarray(differences))
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "conditions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
