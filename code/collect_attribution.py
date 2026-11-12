"""Collect the numbers used by the second-revision manuscript.

Reads the formal run, the schedule sweep, the attribution ablation and the
static-region analysis, and writes ``numbers_v2.json`` with the placeholder
values referenced by ``build_manuscript_v2.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent / "numbers_v2.json"


def load(run: str) -> pd.DataFrame:
    frame = pd.read_csv(RUNS / run / "per_sample_metrics.csv")
    return frame.set_index(["sample_id", "seed"])


def main() -> None:
    formal = load("formal_experiment_038")
    sweep = load("gating_sweep_039")
    attrib = load("attribution_ablation_041")
    static = json.loads((RUNS / "static_region_042" / "summary.json").read_text(encoding="utf-8"))

    def j(frame: pd.DataFrame, condition: str) -> pd.Series:
        return frame[frame["condition"] == condition]["background_jitter"]

    def fj(frame: pd.DataFrame, condition: str) -> pd.Series:
        return frame[frame["condition"] == condition]["frame_jitter"]

    def paired(base: pd.Series, treat: pd.Series, frame_base: pd.Series | None = None,
               frame_treat: pd.Series | None = None) -> tuple[float, float]:
        index = base.index.intersection(treat.index)
        delta = float((treat.reindex(index) - base.reindex(index)).mean())
        if frame_base is None:
            return delta, float("nan")
        findex = index.intersection(frame_base.index).intersection(frame_treat.index)
        fdelta = float((frame_treat.reindex(findex) - frame_base.reindex(findex)).mean())
        return delta, fdelta

    values: dict[str, str] = {}

    def put(key: str, value: float, digits: int = 4, signed: bool = True) -> None:
        text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
        values[key] = text

    d, fd = paired(j(formal, "independent_nolora"), j(formal, "independent_lora"),
                   fj(formal, "independent_nolora"), fj(formal, "independent_lora"))
    put("ATTRIB_LORA", d)
    put("ATTRIB_LORA_FRAME", fd)

    d, fd = paired(j(formal, "independent_lora"), j(sweep, "last08_lora"),
                   fj(formal, "independent_lora"), fj(sweep, "last08_lora"))
    put("ATTRIB_GATE", d)
    put("ATTRIB_GATE_FRAME", fd)

    d, fd = paired(j(formal, "independent_nolora"), j(attrib, "nolora_late08"),
                   fj(formal, "independent_nolora"), fj(attrib, "nolora_late08"))
    put("ATTRIB_NO_LORA_GAIN", d)
    put("ATTRIB_NO_LORA_GAIN_FRAME", fd)

    d, fd = paired(j(sweep, "last08_lora"), j(formal, "ours_full_lora"),
                   fj(sweep, "last08_lora"), fj(formal, "ours_full_lora"))
    put("ATTRIB_FULL_VS_LATE", d)
    put("ATTRIB_FULL_VS_LATE_FRAME", fd)

    d, fd = paired(j(formal, "independent_lora"), j(attrib, "lora_nocanny"),
                   fj(formal, "independent_lora"), fj(attrib, "lora_nocanny"))
    put("ATTRIB_CANNY_DELTA", d)
    put("ATTRIB_CANNY_DELTA_FRAME", fd)

    composite = []
    for condition in ["independent_nolora", "independent_lora", "locked_lora", "ours_full_lora", "ours_late_lora"]:
        composite.append(float((fj(formal, condition) - j(formal, condition)).mean()))
    put("ATTRIB_COMPOSITE", sum(composite) / len(composite))

    recall = attrib[attrib["condition"] == "lora_nocanny"]["control_edge_recall"].mean()
    put("ATTRIB_NOCANNY_RECALL", float(recall), digits=4, signed=False)

    conditions = static["conditions"]
    put("ATTRIB_STATIC_BASE", conditions["independent_lora"]["static_jitter"]["mean"], signed=False)
    put("ATTRIB_STATIC_GATE", conditions["ours_late_lora"]["static_jitter"]["mean"], signed=False)
    put("ATTRIB_MOVING_BASE", conditions["independent_lora"]["moving_jitter"]["mean"], signed=False)
    put("ATTRIB_MOVING_GATE", conditions["ours_late_lora"]["moving_jitter"]["mean"], signed=False)

    OUT.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(values, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
