#!/usr/bin/env python3
"""Generate seeded ON/OFF classifications from canonical processed traces.

The deposited ensemble tables remain the record used for the accepted paper.
This script exposes a deterministic reanalysis with the manuscript's 5th and
95th percentile rule; it never overwrites deposited classifications.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utilities import DEFAULT_RANDOM_SEED, classify_on_off, genotype_from_mouse

from calculate_er_am import ASSAY_LABELS, state_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-data", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/tables/on_off_reanalysis.csv")
    )
    parser.add_argument("--assays", nargs="+", choices=tuple(ASSAY_LABELS), default=tuple(ASSAY_LABELS))
    parser.add_argument("--shuffles", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
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
                traces = activity.iloc[:n].drop(columns="time_s")
                behavior = behavior.iloc[:n]
                for state, mask in state_masks(assay, behavior).items():
                    if not mask.any() or mask.all():
                        continue
                    classified = classify_on_off(
                        traces,
                        mask,
                        n_shuffles=args.shuffles,
                        alpha_percent=5.0,
                        random_seed=args.seed,
                    ).reset_index()
                    classified.insert(0, "state", state)
                    classified.insert(0, "assay", ASSAY_LABELS[assay])
                    classified.insert(0, "genotype", genotype_from_mouse(subject_dir.name))
                    classified.insert(0, "mouse_id", subject_dir.name)
                    rows.append(classified)
            except (FileNotFoundError, KeyError, ValueError) as error:
                skipped.append({"mouse_id": subject_dir.name, "assay": assay, "reason": str(error)})

    if not rows:
        raise SystemExit("No classifications were produced; inspect processed data and frame labels.")
    result = pd.concat(rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    skipped_path = args.output.with_name(f"{args.output.stem}_skipped.csv")
    pd.DataFrame(skipped, columns=["mouse_id", "assay", "reason"]).to_csv(skipped_path, index=False)
    print(f"Wrote {len(result):,} neuron-state rows to {args.output}")
    print(f"Wrote {len(skipped):,} skipped-subject records to {skipped_path}")


if __name__ == "__main__":
    main()
