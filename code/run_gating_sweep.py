"""Sweep the temporal-residual gating schedule on the formal held-out split.

Revision experiment R1.  The original formal run expressed the late gate as a
normalized-timestep cutoff (``max_apply_timestep=0.30``).  Because the PNDM
schedule for ten requested steps exposes eleven denoising calls, that cutoff
happened to select exactly three calls.  A reviewer can reasonably ask whether
three calls were chosen or merely inherited.

This script re-expresses the gate in the units the manuscript will use --
"apply the adapter during the last k of the N observed denoising calls" -- and
sweeps k over the same ten held-out assets and three fixed seeds.  k = N is the
full-injection ablation; k = 0 is the independent LoRA baseline (already in run
038).  It also runs one control condition, ``last3_shift8_lora``, in which the
frame order is cyclically shifted by half a sequence before the adapter is
called and restored afterwards.  The control keeps the parameter count, the
adapter scale and the number of gated calls fixed while destroying temporal
adjacency, which tests whether the cross-frame convolution is what carries the
effect.

Outputs per condition follow the layout of ``run_formal_experiment.py`` so the
existing audit and figure tooling can read them.
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
from diffusers import PNDMScheduler
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import run_formal_experiment as rfe  # noqa: E402
from run_denoising_temporal_adapter import make_condition  # noqa: E402
from temporal_adapter import TimestepTemporalLatentAdapter  # noqa: E402


def call_timesteps(model_dir: Path, steps: int) -> list[int]:
    """Return the timesteps PNDM exposes for ``steps`` requested steps."""
    scheduler = PNDMScheduler.from_pretrained(
        str(model_dir), subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(steps)
    return [int(value) for value in scheduler.timesteps.tolist()]


def gate_threshold(timesteps: list[int], last_k: int, num_train_timesteps: int) -> float:
    """Normalized-timestep cutoff that gates exactly the last ``last_k`` calls."""
    if last_k <= 0 or last_k > len(timesteps):
        raise ValueError(f"last_k must be in [1, {len(timesteps)}], got {last_k}")
    return timesteps[-last_k] / max(num_train_timesteps - 1, 1)


def generate_gated(
    pipe,
    adapter: TimestepTemporalLatentAdapter,
    target_frames: list[Image.Image],
    seed: int,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    control_scale: float,
    temporal_scale: float,
    max_apply_timestep: float,
    chunk_size: int,
    device: str,
    dtype: torch.dtype,
    frame_permutation: torch.Tensor | None = None,
):
    """Same code path as ``run_formal_experiment.generate_ours`` plus a permutation option.

    With ``frame_permutation=None`` this reproduces the formal run exactly.  With a
    permutation, the adapter reads frames in permuted order and the resulting
    residuals are scattered back to their original frames.  Every frame is still
    updated once per gated call with the same scale, but its neighbours come from
    elsewhere in the sequence.  Note that rolling the latent tensor and rolling it
    back would cancel exactly; scattering only the residual does not.
    """
    conditions = [make_condition(frame, (width, height)) for frame in target_frames]
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt, device, 1, True, negative_prompt="")
    frame_count = len(target_frames)
    prompt_positive = prompt_embeds.repeat(frame_count, 1, 1)
    prompt_negative = negative_prompt_embeds.repeat(frame_count, 1, 1)
    control_tensor = pipe.prepare_image(
        conditions, width, height, frame_count, 1, device, dtype,
        do_classifier_free_guidance=False, guess_mode=False,
    )
    latent_height, latent_width = height // 8, width // 8
    noise = []
    for frame_index in range(frame_count):
        generator = torch.Generator(device=device).manual_seed(seed + frame_index)
        noise.append(
            torch.randn(
                (1, pipe.unet.config.in_channels, latent_height, latent_width),
                generator=generator, device=device, dtype=dtype,
            )
        )
    initial_noise = torch.cat(noise, dim=0)
    latents = initial_noise * pipe.scheduler.init_noise_sigma
    pipe.scheduler.set_timesteps(steps, device=device)
    applied_steps = 0
    residual_norms: list[float] = []
    with torch.no_grad():
        for timestep in pipe.scheduler.timesteps:
            normalized_timestep = timestep.float().reshape(1) / max(
                int(pipe.scheduler.config.num_train_timesteps) - 1, 1
            )
            apply_adapter = float(normalized_timestep.item()) <= max_apply_timestep
            if apply_adapter:
                if frame_permutation is None:
                    refined = adapter(latents.float().unsqueeze(0), normalized_timestep).squeeze(0)
                    residual = refined - latents.float()
                else:
                    source = latents.float()[frame_permutation]
                    refined = adapter(source.unsqueeze(0), normalized_timestep).squeeze(0)
                    scattered = torch.zeros_like(latents, dtype=torch.float32)
                    scattered[frame_permutation] = refined - source
                    residual = scattered
                updated = latents.float() + temporal_scale * residual
                adapted_latents = updated.to(dtype)
                applied_steps += 1
                denominator = float(latents.float().norm())
                if denominator > 0:
                    residual_norms.append(float((temporal_scale * residual).norm()) / denominator)
            else:
                adapted_latents = latents
            noise_chunks = []
            for start in range(0, frame_count, chunk_size):
                end = min(start + chunk_size, frame_count)
                model_latents = torch.cat([adapted_latents[start:end], adapted_latents[start:end]], dim=0)
                model_input = pipe.scheduler.scale_model_input(model_latents, timestep)
                timestep_batch = timestep.expand(2 * (end - start))
                control_batch = torch.cat([control_tensor[start:end], control_tensor[start:end]], dim=0)
                text_batch = torch.cat([prompt_negative[start:end], prompt_positive[start:end]], dim=0)
                down_samples, mid_sample = pipe.controlnet(
                    model_input, timestep_batch, encoder_hidden_states=text_batch,
                    controlnet_cond=control_batch, conditioning_scale=control_scale,
                    guess_mode=False, return_dict=False,
                )
                predicted = pipe.unet(
                    model_input, timestep_batch, encoder_hidden_states=text_batch,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                ).sample
                uncond, text = predicted.chunk(2)
                noise_chunks.append(uncond + 7.5 * (text - uncond))
            noise_pred = torch.cat(noise_chunks, dim=0)
            latents = pipe.scheduler.step(noise_pred, timestep, latents).prev_sample
        decoded = []
        for start in range(0, len(latents), chunk_size):
            images = pipe.vae.decode(
                latents[start : start + chunk_size] / pipe.vae.config.scaling_factor
            ).sample
            decoded.append(
                ((images.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            )
    backgrounds = [Image.fromarray(array, mode="RGB") for chunk in decoded for array in chunk]
    residual_summary = {
        "relative_residual_mean": float(np.mean(residual_norms)) if residual_norms else 0.0,
        "relative_residual_max": float(np.max(residual_norms)) if residual_norms else 0.0,
    }
    return backgrounds, conditions, initial_noise.detach().cpu(), applied_steps, residual_summary


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
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--temporal-scale", type=float, default=0.10)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--gate-call-list", default="1,2,3,4,6,11")
    parser.add_argument("--run-permute-control", action="store_true")
    parser.add_argument("--control-k", type=int, default=3)
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
    num_train_timesteps = 1000
    gate_calls = [int(value) for value in args.gate_call_list.split(",") if value.strip()]
    conditions: list[dict[str, object]] = []
    for k in gate_calls:
        conditions.append(
            {
                "name": f"last{k:02d}_lora",
                "last_k": k,
                "threshold": gate_threshold(timesteps, k, num_train_timesteps),
                "permute": False,
            }
        )
    if args.run_permute_control:
        conditions.append(
            {
                "name": f"last{args.control_k:02d}_perm_lora",
                "last_k": args.control_k,
                "threshold": gate_threshold(timesteps, args.control_k, num_train_timesteps),
                "permute": True,
            }
        )

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
        "purpose": "gating-schedule sensitivity for the timestep-conditioned temporal residual",
        "denoising_calls": len(timesteps),
        "call_timesteps": timesteps,
        "normalized_timesteps": [t / max(num_train_timesteps - 1, 1) for t in timesteps],
        "conditions": conditions,
        "seeds": seeds,
        "eval_split": [label["sample_id"] for label in selected],
        "steps": args.steps,
        "temporal_scale": args.temporal_scale,
        "controlnet_scale": args.controlnet_scale,
        "resolution": [args.width, args.height],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"timesteps": timesteps, "thresholds": {c["name"]: c["threshold"] for c in conditions}}, indent=2))

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
                sample_dir = args.out / "samples" / target["sample_id"] / f"seed_{seed}" / str(condition["name"])
                print(
                    f"sample={index}/{total} sequence={target['sample_id']} seed={seed} "
                    f"condition={condition['name']} gate={condition['last_k']}",
                    flush=True,
                )
                if (sample_dir / "metrics.json").exists():
                    rows.append(json.loads((sample_dir / "metrics.json").read_text(encoding="utf-8")))
                    print("  resumed", flush=True)
                    continue
                pipe.enable_lora()
                permutation = None
                if condition["permute"]:
                    generator = torch.Generator().manual_seed(seed + 7919)
                    permutation = torch.randperm(len(target_frames), generator=generator).to(device)
                backgrounds, condition_images, noise, applied, residual_summary = generate_gated(
                    pipe,
                    adapter,
                    target_frames,
                    seed,
                    args.prompt,
                    args.width,
                    args.height,
                    args.steps,
                    args.controlnet_scale,
                    args.temporal_scale,
                    float(condition["threshold"]),
                    args.chunk_size,
                    device,
                    dtype,
                    frame_permutation=permutation,
                )
                if applied != int(condition["last_k"]):
                    raise RuntimeError(
                        f"{condition['name']} gated {applied} calls, expected {condition['last_k']}"
                    )
                metrics = rfe.save_sample(
                    sample_dir,
                    target,
                    target_frames,
                    backgrounds,
                    condition_images,
                    asset_path,
                    str(condition["name"]),
                    noise,
                    seed,
                    applied,
                )
                metrics["gated_calls"] = applied
                metrics["relative_residual_mean"] = residual_summary["relative_residual_mean"]
                metrics["relative_residual_max"] = residual_summary["relative_residual_max"]
                (sample_dir / "gate_info.json").write_text(
                    json.dumps(
                        {
                            "gated_calls": applied,
                            "permuted": bool(condition["permute"]),
                            "threshold": float(condition["threshold"]),
                            **residual_summary,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                rows.append(metrics)
                print(
                    f"  jitter={metrics['background_jitter']:.6f} "
                    f"rel_residual={metrics['relative_residual_mean']:.4f}",
                    flush=True,
                )

    rfe.write_csv(args.out / "per_sample_metrics.csv", rows)

    formal = Path(args.out).parent / "formal_experiment_038" / "per_sample_metrics.csv"
    merged = list(rows)
    if formal.exists():
        import csv as _csv

        # Run 038 already contains the endpoints of the schedule: the
        # full-injection condition (all 11 calls) and the original late gate
        # (three calls).  Re-used here under the k-call naming, with sweep rows
        # taking precedence when the same condition was recomputed.
        aliases = {"ours_full_lora": "last11_lora", "ours_late_lora": "last03_lora"}
        seen = {(str(row["sample_id"]), int(row["seed"]), str(row["condition"])) for row in merged}
        with formal.open(encoding="utf-8-sig") as handle:
            for row in _csv.DictReader(handle):
                condition = aliases.get(row["condition"], row["condition"])
                if condition not in {"independent_lora", "last11_lora", "last03_lora"}:
                    continue
                key = (str(row["sample_id"]), int(row["seed"]), condition)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(
                    {
                        **{
                            field: (float(value) if field in rfe.METRIC_KEYS else value)
                            for field, value in row.items()
                        },
                        "condition": condition,
                    }
                )

    condition_names = [str(condition["name"]) for condition in conditions]
    aggregate_rows = []
    for name in condition_names:
        subset = [row for row in rows if row["condition"] == name]
        record = {"condition": name, "n": len(subset), "gated_calls": int(subset[0].get("gated_calls", -1)) if subset else 0}
        for key in rfe.METRIC_KEYS:
            values = np.asarray([float(row[key]) for row in subset], dtype=np.float64)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(record)
    rfe.write_csv(args.out / "aggregate_metrics.csv", aggregate_rows)

    effects = {}
    if any(row["condition"] == "independent_lora" for row in merged):
        for name in condition_names:
            effects[f"{name}_vs_independent_lora"] = rfe.paired_effects(merged, "independent_lora", name)["background_jitter"]
    prespecified_pairs = [
        ("last01_lora", "last03_lora"),
        ("last02_lora", "last03_lora"),
        ("last04_lora", "last03_lora"),
        ("last06_lora", "last03_lora"),
        ("last11_lora", "last03_lora"),
        (f"last{args.control_k:02d}_perm_lora", f"last{args.control_k:02d}_lora"),
    ]
    available = {str(row["condition"]) for row in merged}
    for baseline, treatment in prespecified_pairs:
        if baseline in available and treatment in available:
            effects[f"{baseline}_vs_{treatment}"] = rfe.paired_effects(merged, baseline, treatment)["background_jitter"]
    (args.out / "paired_effects.json").write_text(
        json.dumps(effects, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        **config,
        "status": "completed",
        "num_rows": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "aggregate_metrics": str(args.out / "aggregate_metrics.csv"),
        "paired_effects": str(args.out / "paired_effects.json"),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "rows": len(rows), "elapsed_seconds": summary["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
