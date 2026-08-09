from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import DEFAULT_FAMILIES, ExperimentConfig
from .descriptors import quantum_descriptors, structural_descriptors


def deterministic_instance_seed(
    base_seed: int,
    family_index: int,
    n_qubits: int,
    depth: int,
    instance: int,
) -> int:
    return base_seed + 100_000 * family_index + 1_000 * n_qubits + 100 * depth + instance


def build_descriptor_dataset(
    config: ExperimentConfig,
    save: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | int | str]] = []
    total = (
        len(config.families)
        * len(config.qubit_counts)
        * len(config.depths)
        * config.n_instances
    )
    progress = tqdm(total=total, disable=not show_progress, desc="PQC designs")
    start = time.time()

    for family in config.families:
        family_index = DEFAULT_FAMILIES.index(family)
        for n_qubits in config.qubit_counts:
            for depth in config.depths:
                structural = structural_descriptors(family, n_qubits, depth)

                for instance in range(config.n_instances):
                    local_seed = deterministic_instance_seed(
                        config.seed,
                        family_index,
                        n_qubits,
                        depth,
                        instance,
                    )
                    rng = np.random.default_rng(local_seed)
                    quantum = quantum_descriptors(
                        family=family,
                        n_qubits=n_qubits,
                        depth=depth,
                        n_state_samples=config.n_state_samples,
                        n_pair_samples=config.n_pair_samples,
                        n_bins=config.n_bins,
                        n_grad_samples=config.n_grad_samples,
                        trainability_observables=config.trainability_observables,
                        rng=rng,
                    )
                    records.append(
                        {
                            "family": family,
                            "n_qubits": n_qubits,
                            "depth_setting": depth,
                            "instance": instance,
                            "seed": local_seed,
                            **quantum,
                            **structural,
                        }
                    )
                    progress.update(1)

    progress.close()
    df = pd.DataFrame(records)
    df.attrs["elapsed_seconds"] = time.time() - start

    if save:
        df.to_csv(output_dir / "pqc_descriptor_dataset.csv", index=False)
    return df


def aggregate_design_points(df: pd.DataFrame, save_path: Path | None = None) -> pd.DataFrame:
    group_cols = ["family", "n_qubits", "depth_setting"]
    numeric_cols = [
        column
        for column in df.select_dtypes(include=np.number).columns
        if column not in {"instance", "seed"}
    ]
    design_df = df.groupby(group_cols, as_index=False)[numeric_cols].mean()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        design_df.to_csv(save_path, index=False)
    return design_df
