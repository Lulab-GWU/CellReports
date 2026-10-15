#!/usr/bin/env python3
"""Calculate neuron-level calcium transient rate (ER) and amplitude (AM).

Input is a canonical per-subject ``data/processed/<assay>/<mouse>/`` layout.
Peaks are detected once on each complete trace and assigned to behavioral
states by peak time, so gaps between behavioral epochs cannot create
artificial peaks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utilities import (
    TransientSettings,
    epm_arm_categories,
    genotype_from_mouse,
    status_mask,
    transient_metrics_in_mask,
)


ASSAY_LABELS = {"oft": "OFT", "epm": "EPM", "mtc": "SOCIAL"}
BASE_STATES = {
    "oft": ("center", "corner"),
    "epm": ("open", "closed"),
    "mtc": ("social", "object", "left", "middle", "right"),
}


def estimate_fps(time_s: pd.Series) -> float:
    differences = np.diff(pd.to_numeric(time_s, errors="coerce").to_numpy(dtype=float))
    differences = differences[np.isfinite(differences) & (differences > 0)]
    if not len(differences):
        raise ValueError("Cannot estimate frame rate from time_s.")
    return float(1 / np.median(differences))


def state_masks(assay: str, behavior: pd.DataFrame) -> dict[str, np.ndarray]:
    if "frame_label" not in behavior.columns:
        raise ValueError("behavior.parquet has no frame_label column.")
    labels = behavior["frame_label"].to_numpy()
    masks = {
        state: status_mask(labels, state, ASSAY_LABELS[assay])
        for state in BASE_STATES[assay]
    }
    if assay == "epm" and {"x", "y"}.issubset(behavior.columns):
        arms = epm_arm_categories(labels, behavior["x"], behavior["y"])
        masks.update({arm: arms.eq(arm).to_numpy() for arm in ("Open-P", "Open-NP", "Closed-P", "Closed-NP")})
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-data", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/neuron_er_am.csv"))
    parser.add_argument("--assays", nargs="+", choices=tuple(ASSAY_LABELS), default=tuple(ASSAY_LABELS))
    parser.add_argument("--prominence", type=float, default=5.0)
    parser.add_argument("--width-frames", type=float, default=2.0)
    args = parser.parse_args()

    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for assay in args.assays:
        assay_root = args.processed_data / assay
        for subject_dir in sorted(path for path in assay_root.glob("NC*_*" ) if path.is_dir()):
            try:
                activity = pd.read_parquet(subject_dir / "activity.parquet")
                behavior = pd.read_parquet(subject_dir / "behavior.parquet")
                n = min(len(activity), len(behavior))
                activity, behavior = activity.iloc[:n], behavior.iloc[:n]
                fps = estimate_fps(activity["time_s"])
                traces = activity.drop(columns="time_s")
                settings = TransientSettings(
                    fps=fps,
                    prominence=args.prominence,
                    width_frames=args.width_frames,
                )
                for state, mask in state_masks(assay, behavior).items():
                    if not mask.any():
                        continue
                    metrics = transient_metrics_in_mask(traces, mask, settings).reset_index()
                    metrics.insert(0, "state", state)
                    metrics.insert(0, "assay", ASSAY_LABELS[assay])
                    metrics.insert(0, "genotype", genotype_from_mouse(subject_dir.name))
                    metrics.insert(0, "mouse_id", subject_dir.name)
                    metrics["fps"] = fps
                    metrics["peak_prominence"] = args.prominence
                    metrics["minimum_peak_width_frames"] = args.width_frames
                    rows.append(metrics)
            except (FileNotFoundError, KeyError, ValueError) as error:
                skipped.append({"mouse_id": subject_dir.name, "assay": assay, "reason": str(error)})

    if not rows:
        raise SystemExit("No ER/AM rows were produced; inspect the processed-data path and frame labels.")
    result = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    skipped_path = args.output.with_name(f"{args.output.stem}_skipped.csv")
    pd.DataFrame(skipped, columns=["mouse_id", "assay", "reason"]).to_csv(skipped_path, index=False)
    print(f"Wrote {len(result):,} neuron-state rows to {args.output}")
    print(f"Wrote {len(skipped):,} skipped-subject records to {skipped_path}")


if __name__ == "__main__":
    main()
