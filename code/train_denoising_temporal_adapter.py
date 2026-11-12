"""Train a timestep-conditioned temporal adapter for diffusion denoising."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from temporal_adapter import TimestepTemporalLatentAdapter  # noqa: E402


def image_tensor(path: Path, resolution: int) -> torch.Tensor:
    image = ImageOps.pad(
        Image.open(path).convert("RGB"),
        (resolution, resolution),
        method=Image.Resampling.LANCZOS,
        color=(0, 0, 0),
    )
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def encode_sequence(
    vae: AutoencoderKL,
    paths: list[Path],
    resolution: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    pixels = torch.stack([image_tensor(path, resolution) for path in paths])
    chunks = []
    with torch.no_grad():
        for chunk in pixels.split(4):
            latents = vae.encode(chunk.to(device, dtype=dtype)).latent_dist.mode()
            chunks.append((latents * vae.config.scaling_factor).float())
    return torch.cat(chunks, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("D:/sd15"))
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--time-hidden", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-correlation", type=float, default=0.80)
    parser.add_argument("--train-start", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    rng = torch.Generator(device=device).manual_seed(args.seed)
    if not 0.0 <= args.target_correlation < 1.0:
        raise ValueError("target-correlation must be in [0, 1)")

    labels = json.loads((args.synthetic / "labels.json").read_text(encoding="utf-8"))
    selected = labels[args.train_start : args.train_start + args.train_count]
    if not selected:
        raise ValueError("No training sequences selected")
    started = time.perf_counter()
    vae = AutoencoderKL.from_pretrained(
        str(args.model_dir), subfolder="vae", variant="fp16", torch_dtype=dtype, local_files_only=True
    ).to(device).eval()
    vae.requires_grad_(False)
    sequence_latents = []
    for label in selected:
        paths = [args.synthetic / path for path in label["frame_paths"]]
        sequence_latents.append(encode_sequence(vae, paths, args.resolution, device, dtype))
    clean = torch.stack(sequence_latents).to(device)
    del vae
    if device == "cuda":
        torch.cuda.empty_cache()

    scheduler = DDIMScheduler.from_pretrained(str(args.model_dir), subfolder="scheduler", local_files_only=True)
    num_train_timesteps = int(scheduler.config.num_train_timesteps)
    adapter = TimestepTemporalLatentAdapter(
        channels=clean.shape[2], hidden=args.hidden, time_hidden=args.time_hidden
    ).to(device).float()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    rows: list[dict[str, float | int]] = []
    rho = float(args.target_correlation)
    independent_weight = float(np.sqrt(max(1.0 - rho * rho, 1e-8)))
    for step in range(1, args.max_steps + 1):
        sequence_index = int(torch.randint(clean.shape[0], (1,), generator=rng, device=device).item())
        clean_batch = clean[sequence_index : sequence_index + 1]
        timestep = torch.randint(num_train_timesteps, (1,), generator=rng, device=device, dtype=torch.long)
        independent_noise = torch.randn(clean_batch.shape, generator=rng, device=device, dtype=torch.float32)
        shared_noise = torch.randn(
            (clean_batch.shape[0], 1, clean_batch.shape[2], clean_batch.shape[3], clean_batch.shape[4]),
            generator=rng,
            device=device,
            dtype=torch.float32,
        )
        target_noise = rho * shared_noise + independent_weight * independent_noise
        noisy = scheduler.add_noise(clean_batch, independent_noise, timestep.expand(clean_batch.shape[0]))
        target = scheduler.add_noise(clean_batch, target_noise, timestep.expand(clean_batch.shape[0]))
        normalized_timestep = timestep.float() / max(num_train_timesteps - 1, 1)

        optimizer.zero_grad(set_to_none=True)
        prediction = adapter(noisy, normalized_timestep)
        reconstruction_loss = F.smooth_l1_loss(prediction, target)
        temporal_loss = F.smooth_l1_loss(
            prediction[:, 1:] - prediction[:, :-1], target[:, 1:] - target[:, :-1]
        )
        loss = reconstruction_loss + 0.2 * temporal_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        rows.append(
            {
                "step": step,
                "loss": value,
                "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                "temporal_loss": float(temporal_loss.detach().cpu()),
                "timestep": int(timestep.item()),
            }
        )
        if step == 1 or step % 100 == 0 or step == args.max_steps:
            print(f"step={step}/{args.max_steps} loss={value:.6f} timestep={int(timestep.item())}", flush=True)

    adapter_path = args.out / "timestep_temporal_adapter.pt"
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "channels": int(clean.shape[2]),
            "hidden": args.hidden,
            "time_hidden": args.time_hidden,
            "class_name": "TimestepTemporalLatentAdapter",
            "num_train_timesteps": num_train_timesteps,
        },
        adapter_path,
    )
    log_path = args.out / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["step", "loss", "reconstruction_loss", "temporal_loss", "timestep"]
        )
        writer.writeheader()
        writer.writerows(rows)
    losses = [row["loss"] for row in rows]
    summary = {
        "experiment": args.out.name,
        "status": "completed",
        "not_a_paper_result": True,
        "model_id": "runwayml/stable-diffusion-v1-5",
        "model_dir": str(args.model_dir),
        "synthetic_dir": str(args.synthetic),
        "train_sequence_ids": [label["sample_id"] for label in selected],
        "num_train_sequences": len(selected),
        "frames_per_sequence": len(selected[0]["frame_paths"]),
        "resolution": args.resolution,
        "hidden_channels": args.hidden,
        "time_hidden": args.time_hidden,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "target_correlation": rho,
        "num_train_timesteps": num_train_timesteps,
        "seed": args.seed,
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "diffusers": __import__("diffusers").__version__,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_loss_last_20": float(np.mean(losses[-20:])),
        "adapter_path": str(adapter_path),
        "train_log": str(log_path),
        "interpretation": "This trains a timestep-conditioned temporal residual on synthetic noisy latent pairs. It is a reproducibility and integration prototype, not a final paper benchmark.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "experiment": summary["experiment"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "initial_loss": summary["initial_loss"],
                "final_loss": summary["final_loss"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
