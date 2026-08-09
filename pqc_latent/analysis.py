from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler


DEFAULT_STRUCTURAL_FEATURES = [
    "parameter_count",
    "logical_depth",
    "total_gate_count",
    "two_qubit_gate_count",
    "connectivity_density",
    "mean_degree",
    "n_components",
    "largest_component_fraction",
    "global_efficiency",
    "avg_shortest_path_lcc",
    "diameter_lcc",
    "mean_edges_per_layer",
    "edge_coverage",
    "edge_reuse_fraction",
]

DEFAULT_CONFOUNDERS = [
    "n_qubits",
    "depth_setting",
    "parameter_count",
    "two_qubit_gate_count",
]


@dataclass
class EmbeddingResult:
    clean: pd.DataFrame
    standardized: np.ndarray
    pca: PCA
    pca_coordinates: np.ndarray
    umap_coordinates: np.ndarray | None
    loadings: pd.DataFrame


def quantum_features(primary_trainability_feature: str) -> list[str]:
    return [
        "expressibility_kl",
        "meyer_wallach_mean",
        primary_trainability_feature,
    ]


def available_features(data: pd.DataFrame, requested: Iterable[str]) -> list[str]:
    return [feature for feature in requested if feature in data.columns]


def spearman_correlation(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    clean_features = available_features(data, features)
    return data[clean_features].corr(method="spearman")


def run_embedding(
    data: pd.DataFrame,
    features: list[str],
    *,
    seed: int = 42,
    use_umap: bool = True,
) -> EmbeddingResult:
    clean = data.dropna(subset=features).copy()
    if len(clean) < 3:
        raise ValueError("At least three complete rows are required for an embedding")

    standardized = StandardScaler().fit_transform(clean[features])
    n_components = min(len(features), 6, len(clean))
    pca = PCA(n_components=n_components, random_state=seed)
    pca_coordinates = pca.fit_transform(standardized)

    umap_coordinates = None
    if use_umap:
        try:
            import umap.umap_ as umap
        except ImportError as exc:
            raise ImportError("UMAP requested; install 'umap-learn'.") from exc
        n_neighbors = min(15, max(2, len(clean) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="euclidean",
            random_state=seed,
        )
        umap_coordinates = reducer.fit_transform(standardized)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    return EmbeddingResult(
        clean=clean,
        standardized=standardized,
        pca=pca,
        pca_coordinates=pca_coordinates,
        umap_coordinates=umap_coordinates,
        loadings=loadings,
    )


def embedding_scores(result: EmbeddingResult) -> pd.DataFrame:
    columns = [column for column in ["family", "n_qubits", "depth_setting"] if column in result.clean]
    scores = result.clean[columns].copy()
    for i in range(result.pca_coordinates.shape[1]):
        scores[f"PC{i + 1}"] = result.pca_coordinates[:, i]
    if result.umap_coordinates is not None:
        scores["UMAP1"] = result.umap_coordinates[:, 0]
        scores["UMAP2"] = result.umap_coordinates[:, 1]
    return scores


def confound_regression(
    latent_data: pd.DataFrame,
    coordinates: np.ndarray,
    confounders: list[str] = DEFAULT_CONFOUNDERS,
    n_axes: int = 2,
) -> pd.DataFrame:
    clean = latent_data.dropna(subset=confounders).copy()
    if len(clean) != len(latent_data):
        raise ValueError("Confounder columns contain missing values")

    X = clean[confounders].to_numpy(dtype=float)
    rows = []
    for axis in range(min(n_axes, coordinates.shape[1])):
        y = coordinates[:, axis]
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        row: dict[str, float | str] = {
            "latent_axis": f"PC{axis + 1}",
            "R2_from_structural_scale": float(r2_score(y, pred)),
            "intercept": float(model.intercept_),
        }
        for name, coefficient in zip(confounders, model.coef_):
            row[f"coef_{name}"] = float(coefficient)
        rows.append(row)
    return pd.DataFrame(rows)


def residualize_quantum_features(
    data: pd.DataFrame,
    features: list[str],
    confounders: list[str] = DEFAULT_CONFOUNDERS,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    clean = data.dropna(subset=features + confounders).copy()
    X = clean[confounders].to_numpy(dtype=float)
    diagnostics = []
    residual_features = []

    for feature in features:
        y = clean[feature].to_numpy(dtype=float)
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        residual_name = f"resid_{feature}"
        clean[residual_name] = y - pred
        residual_features.append(residual_name)
        diagnostics.append(
            {
                "descriptor": feature,
                "R2_removed_by_confounders": float(r2_score(y, pred)),
            }
        )

    return clean, residual_features, pd.DataFrame(diagnostics)


def controlled_family_comparisons(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for (n_qubits, depth), block in data.groupby(["n_qubits", "depth_setting"]):
        for feature in features:
            values = block.set_index("family")[feature].dropna()
            if len(values) < 2:
                continue
            rows.append(
                {
                    "n_qubits": int(n_qubits),
                    "depth": int(depth),
                    "descriptor": feature,
                    "between_family_std": float(values.std(ddof=1)),
                    "between_family_range": float(values.max() - values.min()),
                    "minimum_value_family": str(values.idxmin()),
                    "maximum_value_family": str(values.idxmax()),
                }
            )
    return pd.DataFrame(rows)


def clustering_quality(
    spaces: dict[str, np.ndarray],
    *,
    seed: int = 42,
    max_k: int = 8,
) -> pd.DataFrame:
    rows = []
    for space_name, matrix in spaces.items():
        upper = min(max_k, len(matrix) - 1)
        for k in range(2, upper + 1):
            labels = KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(matrix)
            rows.append(
                {
                    "space": space_name,
                    "k": k,
                    "silhouette": float(silhouette_score(matrix, labels)),
                }
            )
    return pd.DataFrame(rows)


def pareto_mask(costs: np.ndarray) -> np.ndarray:
    costs = np.asarray(costs, dtype=float)
    efficient = np.ones(len(costs), dtype=bool)
    for i in range(len(costs)):
        for j in range(len(costs)):
            if i == j:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                efficient[i] = False
                break
    return efficient


def pareto_analysis(
    data: pd.DataFrame,
    trainability_feature: str,
    *,
    group_cols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    result = data.copy()
    result["pareto_quantum"] = False

    def mark(block: pd.DataFrame) -> None:
        costs = np.column_stack(
            [
                block["expressibility_kl"].to_numpy(),
                -block["meyer_wallach_mean"].to_numpy(),
                -block[trainability_feature].to_numpy(),
            ]
        )
        result.loc[block.index, "pareto_quantum"] = pareto_mask(costs)

    if group_cols:
        grouper = group_cols[0] if len(group_cols) == 1 else list(group_cols)
        for _, block in result.groupby(grouper):
            mark(block)
    else:
        mark(result)
    return result


def participation_ratio_dimension(standardized: np.ndarray) -> float:
    covariance = np.cov(standardized, rowvar=False)
    eigvals = np.linalg.eigvalsh(covariance)
    eigvals = np.clip(eigvals, 0.0, None)
    denom = float(np.sum(eigvals**2))
    if denom == 0.0:
        return 0.0
    return float(np.sum(eigvals) ** 2 / denom)


def pca_stability_across_qubits(
    data: pd.DataFrame,
    features: list[str],
    *,
    seed: int = 42,
    n_subspace_components: int = 2,
) -> pd.DataFrame:
    """Compare PCA subspace orientation, not only explained variance."""
    clean = data.dropna(subset=features).copy()
    global_scaler = StandardScaler().fit(clean[features])
    models: dict[int, PCA] = {}
    rows = []

    for n_qubits, block in clean.groupby("n_qubits"):
        matrix = global_scaler.transform(block[features])
        n_components = min(len(features), len(block))
        model = PCA(n_components=n_components, random_state=seed).fit(matrix)
        models[int(n_qubits)] = model
        rows.append(
            {
                "n_qubits": int(n_qubits),
                "pc1_variance": float(model.explained_variance_ratio_[0]),
                "pc2_variance": float(
                    model.explained_variance_ratio_[1]
                    if len(model.explained_variance_ratio_) > 1
                    else np.nan
                ),
                "pc1_plus_pc2": float(model.explained_variance_ratio_[:2].sum()),
            }
        )

    if not rows:
        return pd.DataFrame()

    reference_n = min(models)
    reference = models[reference_n]
    reference_basis = reference.components_[:n_subspace_components].T
    reference_pc1 = reference.components_[0]

    for row in rows:
        model = models[int(row["n_qubits"])]
        basis = model.components_[:n_subspace_components].T
        angles = np.degrees(subspace_angles(reference_basis, basis))
        row["reference_n_qubits"] = reference_n
        row["max_subspace_angle_deg"] = float(np.max(angles))
        row["mean_subspace_angle_deg"] = float(np.mean(angles))
        row["abs_pc1_loading_cosine"] = float(
            abs(np.dot(reference_pc1, model.components_[0]))
        )

    return pd.DataFrame(rows)


def save_embedding(result: EmbeddingResult, output_dir: Path, prefix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_scores(result).to_csv(output_dir / f"{prefix}_embedding_scores.csv", index=False)
    result.loadings.to_csv(output_dir / f"{prefix}_pca_loadings.csv")
