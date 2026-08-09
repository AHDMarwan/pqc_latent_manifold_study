from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .analysis import (
    DEFAULT_CONFOUNDERS,
    DEFAULT_STRUCTURAL_FEATURES,
    clustering_quality,
    confound_regression,
    controlled_family_comparisons,
    pareto_analysis,
    participation_ratio_dimension,
    pca_stability_across_qubits,
    quantum_features,
    residualize_quantum_features,
    run_embedding,
    save_embedding,
    spearman_correlation,
)
from .config import ExperimentConfig
from .dataset import aggregate_design_points, build_descriptor_dataset
from .plotting import plot_correlation, plot_embedding


def analyze_descriptor_dataset(
    raw_df: pd.DataFrame,
    config: ExperimentConfig,
    *,
    use_umap: bool = True,
) -> dict:
    """Run all statistical/latent-space analyses on an existing raw dataset.

    Keeping analysis separate from quantum simulation makes publication-scale
    GitHub Actions runs easy to shard across qubit counts and circuit families.
    """
    config.validate()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_identity = {"family", "n_qubits", "depth_setting", "instance", "seed"}
    missing = sorted(required_identity.difference(raw_df.columns))
    if missing:
        raise ValueError(f"Raw descriptor dataset is missing columns: {missing}")

    sort_cols = ["family", "n_qubits", "depth_setting", "instance"]
    raw_df = raw_df.sort_values(sort_cols).reset_index(drop=True)
    raw_df.to_csv(output_dir / "pqc_descriptor_dataset.csv", index=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    design_df = aggregate_design_points(
        raw_df,
        save_path=output_dir / "pqc_design_level_dataset.csv",
    )

    q_features = quantum_features(config.primary_trainability_feature)
    full_features = q_features + [
        feature
        for feature in DEFAULT_STRUCTURAL_FEATURES
        if feature not in q_features and feature in design_df.columns
    ]

    correlation = spearman_correlation(design_df, full_features)
    correlation.to_csv(output_dir / "spearman_correlation_matrix.csv")
    plot_correlation(correlation, output_dir / "spearman_correlation_matrix.png")

    q_embedding = run_embedding(
        design_df,
        q_features,
        seed=config.seed,
        use_umap=use_umap,
    )
    full_embedding = run_embedding(
        design_df,
        full_features,
        seed=config.seed,
        use_umap=use_umap,
    )
    save_embedding(q_embedding, output_dir, "quantum_only")
    save_embedding(full_embedding, output_dir, "full")

    plot_embedding(
        q_embedding.clean,
        q_embedding.pca_coordinates,
        title="Quantum-only PCA by family",
        xlabel=f"PC1 ({q_embedding.pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({q_embedding.pca.explained_variance_ratio_[1]:.1%})",
        output_path=output_dir / "quantum_only_pca_by_family.png",
    )
    if q_embedding.umap_coordinates is not None:
        plot_embedding(
            q_embedding.clean,
            q_embedding.umap_coordinates,
            title="Quantum-only UMAP by family",
            xlabel="UMAP1",
            ylabel="UMAP2",
            output_path=output_dir / "quantum_only_umap_by_family.png",
        )

    confounds = confound_regression(
        q_embedding.clean,
        q_embedding.pca_coordinates,
        confounders=DEFAULT_CONFOUNDERS,
    )
    confounds.to_csv(output_dir / "latent_axis_confound_regression.csv", index=False)

    controlled = controlled_family_comparisons(design_df, q_features)
    controlled.to_csv(output_dir / "controlled_family_comparisons.csv", index=False)

    residual_df, residual_features, residual_diagnostics = residualize_quantum_features(
        design_df,
        q_features,
        confounders=DEFAULT_CONFOUNDERS,
    )
    residual_diagnostics.to_csv(
        output_dir / "residualization_diagnostics.csv",
        index=False,
    )
    residual_embedding = run_embedding(
        residual_df,
        residual_features,
        seed=config.seed,
        use_umap=use_umap,
    )
    save_embedding(residual_embedding, output_dir, "residualized")
    plot_embedding(
        residual_embedding.clean,
        residual_embedding.pca_coordinates,
        title="Residualized PCA by family",
        xlabel=f"PC1 ({residual_embedding.pca.explained_variance_ratio_[0]:.1%})",
        ylabel=f"PC2 ({residual_embedding.pca.explained_variance_ratio_[1]:.1%})",
        output_path=output_dir / "residualized_pca_by_family.png",
    )
    if residual_embedding.umap_coordinates is not None:
        plot_embedding(
            residual_embedding.clean,
            residual_embedding.umap_coordinates,
            title="Residualized UMAP by family",
            xlabel="UMAP1",
            ylabel="UMAP2",
            output_path=output_dir / "residualized_umap_by_family.png",
        )

    cluster_df = clustering_quality(
        {
            "quantum_only": q_embedding.standardized,
            "residualized": residual_embedding.standardized,
        },
        seed=config.seed,
    )
    cluster_df.to_csv(output_dir / "cluster_quality.csv", index=False)

    global_pareto = pareto_analysis(
        design_df,
        config.primary_trainability_feature,
        group_cols=None,
    )
    global_pareto.to_csv(output_dir / "pareto_designs_global.csv", index=False)

    budget_pareto = pareto_analysis(
        design_df,
        config.primary_trainability_feature,
        group_cols=("n_qubits", "depth_setting"),
    )
    budget_pareto.to_csv(output_dir / "pareto_designs_fixed_size_depth.csv", index=False)

    stability = pca_stability_across_qubits(
        design_df,
        q_features,
        seed=config.seed,
    )
    stability.to_csv(output_dir / "pca_stability_across_qubit_counts.csv", index=False)

    if not cluster_df.empty:
        best = cluster_df.sort_values("silhouette", ascending=False).iloc[0]
        best_cluster = {
            "space": str(best["space"]),
            "k": int(best["k"]),
            "silhouette": float(best["silhouette"]),
        }
    else:
        best_cluster = None

    summary = {
        "n_raw_rows": int(len(raw_df)),
        "n_design_points": int(len(design_df)),
        "quantum_features": q_features,
        "quantum_pca_two_component_variance": float(
            q_embedding.pca.explained_variance_ratio_[:2].sum()
        ),
        "quantum_participation_ratio_dimension": participation_ratio_dimension(
            q_embedding.standardized
        ),
        "pc1_R2_from_structural_scale": float(
            confounds.loc[
                confounds["latent_axis"] == "PC1",
                "R2_from_structural_scale",
            ].iloc[0]
        ),
        "pc2_R2_from_structural_scale": float(
            confounds.loc[
                confounds["latent_axis"] == "PC2",
                "R2_from_structural_scale",
            ].iloc[0]
        ),
        "best_exploratory_cluster_result": best_cluster,
        "global_pareto_design_count": int(global_pareto["pareto_quantum"].sum()),
        "fixed_size_depth_pareto_design_count": int(
            budget_pareto["pareto_quantum"].sum()
        ),
    }

    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    archive = shutil.make_archive(str(output_dir), "zip", output_dir)
    summary["archive_path"] = archive
    return summary


def run_full_pipeline(
    config: ExperimentConfig,
    *,
    use_umap: bool = True,
    show_progress: bool = True,
) -> dict:
    config.validate()
    raw_df = build_descriptor_dataset(config, save=False, show_progress=show_progress)
    return analyze_descriptor_dataset(raw_df, config, use_umap=use_umap)
