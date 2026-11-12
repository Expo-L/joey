"""Attribution ablation: isolate the residual without LoRA and the Canny condition.

Two conditions are added to the formal design so that each component of the
pipeline can be attributed separately:

``nolora_late08``   LoRA disabled, late window of eight calls.  Compared with the
                    formal no-LoRA baseline this isolates the residual; compared
                    with the eight-call LoRA condition it isolates LoRA.
``lora_nocanny``    LoRA enabled, ControlNet conditioning scale set to zero and no
                    residual.  Compared with the formal independent LoRA condition
                    this isolates the Canny layout condition.

Outputs follow the layout of ``run_formal_experiment.py`` so the same audit and
plotting code can read them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_formal_experiment as rfe  # noqa: E402
from run_gating_sweep import call_timesteps, gate_threshold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("D:/sd15"))
    parser.add_argument("--controlnet-dir", type=Path, default=Path("D:/sd-controlnet-canny"))
    parser.add_argument("--style-adapter-dir", type=Path, required=True)
    parser.add_argument("--temporal-adapter-path", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--commons", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", default="broadcast television branding, professional graphic packaging, clean high-contrast layout")
    parser.add_argument("--seed-list", default="20260903,20261903,20262903")
    parser.add_argument("--eval-start", type=int, default=150)
    parser.add_argument("--eval-count", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--temporal-scale", type=float, default=0.10)
    parser.add_argument("--gate-calls", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = rfe.parse_seeds(args.seed_list)
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    timesteps = call_timesteps(args.model_dir, args.steps)
    threshold = gate_threshold(timesteps, args.gate_calls, 1000)
    conditions = [
        {"name": f"nolora_late{args.gate_calls:02d}", "kind": "late", "lora": False, "control": 1.0},
        {"name": "lora_nocanny", "kind": "independent", "lora": True, "control": 0.0},
    ]

    labels = json.loads((args.synthetic / "labels.json").read_text(encoding="utf-8"))
    selected = labels[args.eval_start : args.eval_start + args.eval_count]
    if len(selected) != args.eval_count:
        raise ValueError("Evaluation split is smaller than eval-count")
    records = [
        json.loads(line)
        for line in (args.commons / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    config = {
        "experiment": args.out.name,
        "purpose": "component attribution: residual without LoRA, and LoRA without Canny conditioning",
        "conditions": conditions,
        "gate_calls": args.gate_calls,
        "gate_threshold": threshold,
        "denoising_calls": len(timesteps),
        "seeds": seeds,
        "eval_split": [label["sample_id"] for label in selected],
        "steps": args.steps,
        "temporal_scale": args.temporal_scale,
        "resolution": [args.width, args.height],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"threshold": threshold, "timesteps": timesteps}))

    started = time.perf_counter()
    pipe = rfe.load_pipeline(args.model_dir, args.controlnet_dir, device, dtype)
    pipe.load_lora_weights(str(args.style_adapter_dir))
    pipe.disable_lora()
    adapter = rfe.load_temporal_adapter(args.temporal_adapter_path, device)

    rows: list[dict] = []
    total = len(selected) * len(seeds) * len(conditions)
    index = 0
    for target in selected:
        target_frames = [Image.open(args.synthetic / path).convert("RGB") for path in target["frame_paths"]]
        asset_index = int(target["sample_id"].split("_")[1])
        asset_path = args.commons / records[asset_index]["local_media"]
        for seed in seeds:
            for condition in conditions:
                index += 1
                sample_dir = args.out / "samples" / target["sample_id"] / f"seed_{seed}" / condition["name"]
                print(f"sample={index}/{total} sequence={target['sample_id']} seed={seed} condition={condition['name']}", flush=True)
                if (sample_dir / "metrics.json").exists():
                    rows.append(json.loads((sample_dir / "metrics.json").read_text(encoding="utf-8")))
                    print("  resumed", flush=True)
                    continue
                if condition["lora"]:
                    pipe.enable_lora()
                else:
                    pipe.disable_lora()
                if condition["kind"] == "late":
                    backgrounds, condition_images, noise, applied = rfe.generate_ours(
                        pipe, adapter, target_frames, seed, args.prompt, args.width, args.height,
                        args.steps, condition["control"], args.temporal_scale, threshold,
                        args.chunk_size, device, dtype,
                    )
                    residual_summary = {"relative_residual_mean": 0.0}
                else:
                    backgrounds, condition_images = rfe.generate_independent(
                        pipe, target_frames, seed, args.prompt, args.width, args.height,
                        args.steps, condition["control"], args.chunk_size, device,
                    )
                    noise, applied, residual_summary = None, 0, {"relative_residual_mean": 0.0}
                metrics = rfe.save_sample(
                    sample_dir, target, target_frames, backgrounds, condition_images,
                    asset_path, condition["name"], noise, seed, applied,
                )
                metrics["gated_calls"] = applied
                metrics["relative_residual_mean"] = residual_summary["relative_residual_mean"]
                rows.append(metrics)
                print(f"  jitter={metrics['background_jitter']:.6f}", flush=True)

    rfe.write_csv(args.out / "per_sample_metrics.csv", rows)
    summary = {
        **config,
        "status": "completed",
        "num_rows": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "rows": len(rows)}))


if __name__ == "__main__":
    main()
