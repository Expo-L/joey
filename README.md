# Brand-constrained diffusion for broadcast packaging

Material for the paper *Brand-constrained diffusion for broadcast packaging with a late-window
temporal residual*. The paper is a camera-ready conference submission. This repository holds the
extended tables, the per-sequence metrics and the scripts behind every reported number.

## What is here

| Path | Contents |
|---|---|
| `docs/supplementary.pdf` | Extended tables S1 to S4 and the reproduction notes |
| `docs/supplementary.docx` | Editable version of the same document |
| `tables/` | The four extended tables as CSV, one file per table |
| `data/formal_experiment_038/` | 150 sequences: 10 held-out assets, 3 seeds, 5 conditions |
| `data/gating_sweep_039/` | 180 sequences: the schedule sweep over six values of K |
| `data/attribution_ablation_041/` | 60 sequences: the two attribution conditions |
| `data/metric_robustness_040/` | SSIM, high-pass jitter and worst-frame measures, computed offline |
| `data/static_region_042/` | Jitter on pixels the motion plan leaves unchanged and on the rest |
| `figures/` | Figures 1 and 2 as PDF and PNG, plus the figure source CSVs |
| `code/` | Scripts for the experiments, the metrics and the manuscript build |

## Numbers and how they were produced

All metrics come from the stored frames; no value in the paper was computed by hand.

- Background jitter is the mean absolute RGB difference between adjacent background frames.
- The five pre-specified conditions share the same three seeds, initial noise, Canny condition and prompt.
- The unit of analysis is the asset and seed pair, so n = 30.
- Bootstrap intervals use 5,000 resamples, sign-flip tests use 100,000 draws, and the resampling seed is 20260903.
- The sweep reproduces the single sample value for K = 3 exactly, 0.2649148503939311.

## Reproducing the runs

The runs need a local Stable Diffusion v1.5 checkpoint, a Canny ControlNet and the two adapters
described in the paper. With those in place:

```
python code/run_gating_sweep.py --synthetic <sequences> --commons <manifest>   --style-adapter-dir <lora> --temporal-adapter-path <residual>   --out runs/gating_sweep_039 --gate-call-list 1,2,3,4,6,8
python code/run_attribution_ablation.py --synthetic <sequences> --commons <manifest>   --style-adapter-dir <lora> --temporal-adapter-path <residual>   --out runs/attribution_ablation_041 --gate-calls 8
python code/analyze_metric_robustness.py --run runs/formal_experiment_038 --out runs/metric_robustness_040
python code/analyze_static_region_jitter.py --run runs/formal_experiment_038 --synthetic <sequences>   --out runs/static_region_042
```

## Data and licence notes

The third-party source images that the synthetic sequences were built from are not redistributed
here. The metrics, the scripts and the figures are released for research use. If you reuse the
figures, cite the paper.
