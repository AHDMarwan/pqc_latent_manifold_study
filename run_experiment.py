from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pqc_latent import ExperimentConfig
from pqc_latent.config import DEFAULT_FAMILIES
from pqc_latent.dataset import build_descriptor_dataset
from pqc_latent.pipeline import analyze_descriptor_dataset, run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the corrected PQC latent-manifold benchmark."
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="Use publication-scale Monte Carlo sample counts.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a very small configuration for CI validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pqc_latent_results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-umap",
        action="store_true",
        help="Skip UMAP while retaining PCA and statistical analyses.",
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        help="Restrict simulation to specific qubit counts (useful for CI shards).",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=DEFAULT_FAMILIES,
        help="Restrict simulation to specific circuit families.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate only pqc_descriptor_dataset.csv; skip latent-space analysis.",
    )
    parser.add_argument(
        "--analyze-csv",
        type=Path,
        nargs="+",
        help="Analyze and merge one or more precomputed raw descriptor CSV files.",
    )
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> ExperimentConfig:
    if args.publication:
        config = ExperimentConfig.publication(
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        config = ExperimentConfig(seed=args.seed, output_dir=args.output_dir)

    if args.smoke:
        config = config.with_overrides(
            qubit_counts=(4,),
            depths=(1, 2),
            n_instances=2,
            n_state_samples=8,
            n_pair_samples=28,
            n_grad_samples=4,
            output_dir=args.output_dir,
        )

    overrides = {}
    if args.qubits:
        overrides["qubit_counts"] = tuple(args.qubits)
    if args.families:
        overrides["families"] = tuple(args.families)
    if overrides:
        config = config.with_overrides(**overrides)
    return config


def main() -> None:
    args = parse_args()
    config = make_config(args)

    if args.generate_only and args.analyze_csv:
        raise SystemExit("--generate-only and --analyze-csv are mutually exclusive")

    if args.analyze_csv:
        frames = [pd.read_csv(path) for path in args.analyze_csv]
        raw_df = pd.concat(frames, ignore_index=True)
        identity = ["family", "n_qubits", "depth_setting", "instance", "seed"]
        duplicated = raw_df.duplicated(identity, keep=False)
        if duplicated.any():
            duplicate_rows = raw_df.loc[duplicated, identity].head(10).to_dict("records")
            raise ValueError(f"Duplicate simulation rows detected: {duplicate_rows}")
        summary = analyze_descriptor_dataset(
            raw_df,
            config,
            use_umap=not args.no_umap,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.generate_only:
        df = build_descriptor_dataset(config, save=True, show_progress=True)
        print(
            json.dumps(
                {
                    "n_rows": int(len(df)),
                    "qubit_counts": list(config.qubit_counts),
                    "families": list(config.families),
                    "output": str(config.output_dir / "pqc_descriptor_dataset.csv"),
                },
                indent=2,
            )
        )
        return

    summary = run_full_pipeline(
        config,
        use_umap=not args.no_umap,
        show_progress=True,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
