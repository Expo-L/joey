"""Run a reproducible multi-sequence, multi-seed formal evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image, ImageFilter, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_controlnet_inference import control_edge_recall  # noqa: E402
from run_denoising_temporal_adapter import compose_frame, make_condition, write_gif  # noqa: E402
from run_pilot_experiment import evaluate_sequence, prepare_asset  # noqa: E402
from run_temporal_controlnet_eval import adjacent_jitter  # noqa: E402
from temporal_adapter import TimestepTemporalLatentAdapter  # noqa: E402


CONDITIONS = [
    "independent_nolora",
    "independent_lora",
    "locked_lora",
    "ours_full_lora",
    "ours_late_lora",
]
METRIC_KEYS = [
    "background_jitter",
    "frame_jitter",
    "control_edge_recall",
    "logo_fidelity",
    "frame_mae",
    "bbox_iou",
    "safe_area_violation",
    "center_error",
    "trajectory_residual",
]


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def frame_jitter(images: list[Image.Image]) -> float:
    return adjacent_jitter(images)


def load_pipeline(model_dir: Path, controlnet_dir: Path, device: str, dtype: torch.dtype) -> StableDiffusionControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(
        str(controlnet_dir), torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    load_kwargs = {
        "controlnet": controlnet,
        "torch_dtype": dtype,
        "safety_checker": None,
        "local_files_only": True,
    }
    if (model_dir / "unet/diffusion_pytorch_model.fp16.safetensors").exists():
        load_kwargs["variant"] = "fp16"
    pipe = StableDiffusionControlNetPipeline.from_pretrained(str(model_dir), **load_kwargs).to(device)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.vae.eval()
    return pipe


def generate_independent(
    pipe: StableDiffusionControlNetPipeline,
    target_frames: list[Image.Image],
    seed: int,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    control_scale: float,
    batch_size: int,
    device: str,
) -> tuple[list[Image.Image], list[Image.Image]]:
    backgrounds: list[Image.Image] = []
    conditions = [make_condition(frame, (width, height)) for frame in target_frames]
    for start in range(0, len(conditions), batch_size):
        end = min(start + batch_size, len(conditions))
        generators = [torch.Generator(device=device).manual_seed(seed + index) for index in range(start, end)]
        result = pipe(
            prompt=[prompt] * (end - start),
            image=conditions[start:end],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=7.5,
            controlnet_conditioning_scale=control_scale,
            generator=generators,
        )
        backgrounds.extend([image.convert("RGB") for image in result.images])
    return backgrounds, conditions


def generate_locked(
    pipe: StableDiffusionControlNetPipeline,
    target_frames: list[Image.Image],
    seed: int,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    control_scale: float,
    device: str,
) -> tuple[list[Image.Image], list[Image.Image]]:
    conditions = [make_condition(frame, (width, height)) for frame in target_frames]
    generator = torch.Generator(device=device).manual_seed(seed)
    background = pipe(
        prompt=prompt,
        image=conditions[len(conditions) // 2],
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=7.5,
        controlnet_conditioning_scale=control_scale,
        generator=generator,
    ).images[0].convert("RGB")
    return [background.copy() for _ in target_frames], conditions


def load_temporal_adapter(path: Path, device: str) -> TimestepTemporalLatentAdapter:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("class_name") != "TimestepTemporalLatentAdapter":
        raise ValueError("Formal evaluation requires a TimestepTemporalLatentAdapter checkpoint")
    adapter = TimestepTemporalLatentAdapter(
        channels=int(checkpoint["channels"]),
        hidden=int(checkpoint["hidden"]),
        time_hidden=int(checkpoint["time_hidden"]),
    ).to(device).float().eval()
    adapter.load_state_dict(checkpoint["state_dict"])
    return adapter


def generate_ours(
    pipe: StableDiffusionControlNetPipeline,
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
) -> tuple[list[Image.Image], list[Image.Image], torch.Tensor, int]:
    conditions = [make_condition(frame, (width, height)) for frame in target_frames]
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt, device, 1, True, negative_prompt="")
    frame_count = len(target_frames)
    prompt_positive = prompt_embeds.repeat(frame_count, 1, 1)
    prompt_negative = negative_prompt_embeds.repeat(frame_count, 1, 1)
    control_tensor = pipe.prepare_image(
        conditions,
        width,
        height,
        frame_count,
        1,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    )
    latent_height, latent_width = height // 8, width // 8
    noise = []
    for frame_index in range(frame_count):
        generator = torch.Generator(device=device).manual_seed(seed + frame_index)
        noise.append(
            torch.randn(
                (1, pipe.unet.config.in_channels, latent_height, latent_width),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    initial_noise = torch.cat(noise, dim=0)
    latents = initial_noise * pipe.scheduler.init_noise_sigma
    pipe.scheduler.set_timesteps(steps, device=device)
    applied_steps = 0
    with torch.no_grad():
        for timestep in pipe.scheduler.timesteps:
            normalized_timestep = timestep.float().reshape(1) / max(
                int(pipe.scheduler.config.num_train_timesteps) - 1, 1
            )
            apply_adapter = float(normalized_timestep.item()) <= max_apply_timestep
            if apply_adapter:
                refined = adapter(latents.float().unsqueeze(0), normalized_timestep).squeeze(0)
                residual = refined - latents.float()
                adapted_latents = (latents.float() + temporal_scale * residual).to(dtype)
                applied_steps += 1
            else:
                adapted_latents = latents
            noise_chunks = []
            for start in range(0, frame_count, chunk_size):
                end = min(start + chunk_size, frame_count)
                model_latents = torch.cat([adapted_latents[start:end], adapted_latents[start:end]], dim=0)
                model_input = pipe.scheduler.scale_model_input(model_latents, timestep)
                timestep_batch = timestep.expand(2 * (end - start))
                control_batch = torch.cat([control_tensor[start:end], control_tensor[start:end]], dim=0)
                text_batch = torch.cat(
                    [prompt_negative[start:end], prompt_positive[start:end]], dim=0
                )
                down_samples, mid_sample = pipe.controlnet(
                    model_input,
                    timestep_batch,
                    encoder_hidden_states=text_batch,
                    controlnet_cond=control_batch,
                    conditioning_scale=control_scale,
                    guess_mode=False,
                    return_dict=False,
                )
                predicted = pipe.unet(
                    model_input,
                    timestep_batch,
                    encoder_hidden_states=text_batch,
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
                ((images.float().clamp(-1, 1) + 1) * 127.5)
                .round()
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
    backgrounds = [Image.fromarray(array, mode="RGB") for chunk in decoded for array in chunk]
    return backgrounds, conditions, initial_noise.detach().cpu(), applied_steps


def save_sample(
    sample_dir: Path,
    target: dict,
    target_frames: list[Image.Image],
    backgrounds: list[Image.Image],
    conditions: list[Image.Image],
    asset_path: Path,
    condition_name: str,
    initial_noise: torch.Tensor | None,
    base_seed: int,
    applied_steps: int | None,
) -> dict[str, float | int | str]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    background_dir = sample_dir / "backgrounds"
    frame_dir = sample_dir / "frames"
    background_dir.mkdir(exist_ok=True)
    frame_dir.mkdir(exist_ok=True)
    for index, image in enumerate(backgrounds):
        image.save(background_dir / f"f{index:02d}.png")
    frame_paths: list[Path] = []
    frame_images: list[Image.Image] = []
    for item, background in zip(target["trajectory"], backgrounds):
        path = frame_dir / f"f{int(item['frame']):02d}.png"
        compose_frame(background, asset_path, target, item, path)
        frame_paths.append(path)
        frame_images.append(Image.open(path).convert("RGB"))
    write_gif(frame_paths, sample_dir / "preview.gif")
    if initial_noise is not None:
        torch.save(
            {
                "base_seed": base_seed,
                "per_frame_seeds": [base_seed + index for index in range(len(backgrounds))],
                "tensor": initial_noise,
            },
            sample_dir / "initial_noise.pt",
        )
    metrics = evaluate_sequence(
        target,
        target_frames,
        frame_images,
        target["trajectory"],
        condition_name,
    )
    metrics.update(
        {
            "background_jitter": adjacent_jitter(backgrounds),
            "frame_jitter": frame_jitter(frame_images),
            "control_edge_recall": float(
                np.mean([control_edge_recall(condition, background) for condition, background in zip(conditions, backgrounds)])
            ),
            "seed": base_seed,
            "condition": condition_name,
            "sample_id": target["sample_id"],
        }
    )
    if applied_steps is not None:
        metrics["adapter_applied_steps"] = applied_steps
    (sample_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def aggregate(rows: list[dict[str, float | int | str]]) -> list[dict[str, float | int | str]]:
    output = []
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        record: dict[str, float | int | str] = {"condition": condition, "n": len(subset)}
        for key in METRIC_KEYS:
            values = np.asarray([float(row[key]) for row in subset], dtype=np.float64)
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(record)
    return output


def sign_flip_pvalue(differences: np.ndarray) -> tuple[float, str]:
    """Return a two-sided sign-flip p-value and the method used.

    Exact enumeration is feasible for n <= 20. For larger paired samples we use
    a deterministic Monte Carlo approximation and report it as such.
    """
    differences = np.asarray(differences, dtype=np.float64)
    observed = abs(float(differences.mean()))
    n = len(differences)
    if n == 0:
        return float("nan"), "undefined (no pairs)"
    if n <= 20:
        total = 1 << n
        count = 0
        for mask in range(total):
            signs = np.asarray([1.0 if mask & (1 << index) else -1.0 for index in range(n)])
            if abs(float((differences * signs).mean())) >= observed - 1e-12:
                count += 1
        return float((count + 1) / (total + 1)), "exact sign-flip enumeration"
    rng = np.random.default_rng(20260903)
    samples = 100_000
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(samples, n), replace=True)
    null_means = np.abs((signs * differences[None, :]).mean(axis=1))
    count = int((null_means >= observed - 1e-12).sum())
    return float((count + 1) / (samples + 1)), f"Monte Carlo sign-flip ({samples:,} draws; seed=20260903)"


def paired_effects(rows: list[dict[str, float | int | str]], baseline: str, treatment: str) -> dict[str, dict[str, float | int | str]]:
    index = {}
    for row in rows:
        index[(str(row["sample_id"]), int(row["seed"]), str(row["condition"]))] = row
    differences: dict[str, list[float]] = {key: [] for key in METRIC_KEYS}
    sample_seeds = sorted({(str(row["sample_id"]), int(row["seed"])) for row in rows})
    for sample_id, seed in sample_seeds:
        base = index.get((sample_id, seed, baseline))
        treatment_row = index.get((sample_id, seed, treatment))
        if base is None:
            continue
        if treatment_row is None:
            continue
        for metric in METRIC_KEYS:
            differences[metric].append(float(base[metric]) - float(treatment_row[metric]))
    result = {}
    for metric, values in differences.items():
        array = np.asarray(values, dtype=np.float64)
        if len(array) == 0:
            continue
        rng = np.random.default_rng(20260903)
        bootstrap = np.asarray(
            [rng.choice(array, size=len(array), replace=True).mean() for _ in range(5000)], dtype=np.float64
        )
        p_value, p_method = sign_flip_pvalue(array)
        result[metric] = {
            "baseline": baseline,
            "treatment": treatment,
            "n_pairs": int(len(array)),
            "mean_improvement_baseline_minus_treatment": float(array.mean()),
            "std_difference": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "bootstrap_ci95_low": float(np.percentile(bootstrap, 2.5)),
            "bootstrap_ci95_high": float(np.percentile(bootstrap, 97.5)),
            "sign_flip_p": p_value,
            "p_method": p_method,
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    parser.add_argument("--eval-start", type=int, default=0)
    parser.add_argument("--eval-count", type=int, default=4)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=150)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--controlnet-scale", type=float, default=1.0)
    parser.add_argument("--temporal-scale", type=float, default=0.10)
    parser.add_argument("--late-max-timestep", type=float, default=0.30)
    parser.add_argument("--chunk-size", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seed_list)
    random.seed(seeds[0])
    np.random.seed(seeds[0])
    torch.manual_seed(seeds[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
        "status": "running",
        "model_id": "runwayml/stable-diffusion-v1-5",
        "model_dir": str(args.model_dir),
        "controlnet_dir": str(args.controlnet_dir),
        "style_adapter_dir": str(args.style_adapter_dir),
        "temporal_adapter_path": str(args.temporal_adapter_path),
        "train_split": {
            "synthetic_label_index_start": args.train_start,
            "synthetic_label_index_end_exclusive": args.train_start + args.train_count,
            "note": "Training indices are supplied from the adapter-training runs; this field records the declared split for auditability.",
        },
        "eval_split": [label["sample_id"] for label in selected],
        "conditions": CONDITIONS,
        "seeds": seeds,
        "steps": args.steps,
        "resolution": [args.width, args.height],
        "controlnet_scale": args.controlnet_scale,
        "temporal_scale": args.temporal_scale,
        "late_max_timestep": args.late_max_timestep,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "diffusers": __import__("diffusers").__version__,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.perf_counter()
    pipe = load_pipeline(args.model_dir, args.controlnet_dir, device, dtype)
    pipe.load_lora_weights(str(args.style_adapter_dir))
    pipe.disable_lora()
    adapter = load_temporal_adapter(args.temporal_adapter_path, device)
    rows: list[dict[str, float | int | str]] = []
    total = len(selected) * len(seeds) * len(CONDITIONS)
    sample_index = 0

    def run_condition(
        condition_name: str,
        target: dict,
        target_frames: list[Image.Image],
        asset_path: Path,
        seed: int,
    ) -> None:
        nonlocal sample_index
        sample_index += 1
        print(f"sample={sample_index}/{total} sequence={target['sample_id']} seed={seed} condition={condition_name}", flush=True)
        condition_root = args.out / "samples" / target["sample_id"] / f"seed_{seed}" / condition_name
        if (condition_root / "metrics.json").exists():
            rows.append(json.loads((condition_root / "metrics.json").read_text(encoding="utf-8")))
            print("  resumed", flush=True)
            return
        if condition_name == "independent_nolora":
            pipe.disable_lora()
            backgrounds, conditions = generate_independent(
                pipe, target_frames, seed, args.prompt, args.width, args.height, args.steps, args.controlnet_scale, args.chunk_size, device
            )
            noise = None
            applied = None
        elif condition_name == "independent_lora":
            pipe.enable_lora()
            backgrounds, conditions = generate_independent(
                pipe, target_frames, seed, args.prompt, args.width, args.height, args.steps, args.controlnet_scale, args.chunk_size, device
            )
            noise = None
            applied = None
        elif condition_name == "locked_lora":
            pipe.enable_lora()
            backgrounds, conditions = generate_locked(
                pipe, target_frames, seed, args.prompt, args.width, args.height, args.steps, args.controlnet_scale, device
            )
            noise = None
            applied = None
        else:
            pipe.enable_lora()
            cutoff = args.late_max_timestep if condition_name == "ours_late_lora" else 1.0
            backgrounds, conditions, noise, applied = generate_ours(
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
                cutoff,
                args.chunk_size,
                device,
                dtype,
            )
        metrics = save_sample(
            condition_root,
            target,
            target_frames,
            backgrounds,
            conditions,
            asset_path,
            condition_name,
            noise,
            seed,
            applied,
        )
        rows.append(metrics)
        print(f"  jitter={metrics['background_jitter']:.6f} edge_recall={metrics['control_edge_recall']:.6f}", flush=True)

    for target in selected:
        target_frames = [Image.open(args.synthetic / path).convert("RGB") for path in target["frame_paths"]]
        asset_index = int(target["sample_id"].split("_")[1])
        asset_path = args.commons / records[asset_index]["local_media"]
        for seed in seeds:
            for condition in CONDITIONS:
                run_condition(condition, target, target_frames, asset_path, seed)

    aggregate_rows = aggregate(rows)
    write_csv(args.out / "per_sample_metrics.csv", rows)
    write_csv(args.out / "aggregate_metrics.csv", aggregate_rows)
    effects_lora = paired_effects(rows, "independent_lora", "ours_late_lora")
    effects_nolora = paired_effects(rows, "independent_nolora", "independent_lora")
    effects_gate = paired_effects(rows, "ours_full_lora", "ours_late_lora")
    (args.out / "paired_effects.json").write_text(
        json.dumps(
            {
                "ours_late_vs_independent_lora": effects_lora,
                "independent_lora_vs_independent_nolora": effects_nolora,
                "late_gate_vs_full_injection": effects_gate,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = {
        **config,
        "status": "completed",
        "not_a_paper_result": True,
        "num_sequences": len(selected),
        "num_seeds": len(seeds),
        "num_rows": len(rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "per_sample_metrics": str(args.out / "per_sample_metrics.csv"),
        "aggregate_metrics": str(args.out / "aggregate_metrics.csv"),
        "paired_effects": str(args.out / "paired_effects.json"),
        "interpretation": "Formal multi-sequence and multi-seed prototype evaluation on a held-out synthetic split. It is not yet a complete journal benchmark because the dataset is small and external human/semantic evaluation is not included.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    config["status"] = "completed"
    (args.out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"experiment": args.out.name, "status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"], "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
