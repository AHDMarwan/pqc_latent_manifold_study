from IPython.display import display




import warnings
warnings.filterwarnings("ignore")

from itertools import combinations
from pathlib import Path
import json
import math
import time
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm

import pennylane as qml
from pennylane import numpy as pnp

from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import umap.umap_ as umap

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PennyLane:", qml.__version__)
print("Output directory:", OUTPUT_DIR)


QUBIT_COUNTS = [4, 6]
DEPTHS = [1, 2, 3]

FAMILIES = [
    "hea_line",
    "hea_ring",
    "brickwall",
    "ttn",
    "symmetry_xy",
]

N_INSTANCES = 1
N_STATE_SAMPLES = 12
N_PAIR_SAMPLES = 60
N_GRAD_SAMPLES = 6
N_BINS = 20

CONFIG = {
    "qubit_counts": QUBIT_COUNTS,
    "depths": DEPTHS,
    "families": FAMILIES,
    "n_instances": N_INSTANCES,
    "n_state_samples": N_STATE_SAMPLES,
    "n_pair_samples": N_PAIR_SAMPLES,
    "n_grad_samples": N_GRAD_SAMPLES,
    "n_bins": N_BINS,
    "seed": SEED,
}

print(json.dumps(CONFIG, indent=2))


def edges_for_family(family, n_qubits, layer):
    if family == "hea_line":
        return [(i, i + 1) for i in range(n_qubits - 1)]

    if family == "hea_ring":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
        if n_qubits > 2:
            edges.append((n_qubits - 1, 0))
        return edges

    if family == "brickwall":
        start = layer % 2
        return [(i, i + 1) for i in range(start, n_qubits - 1, 2)]

    if family == "ttn":
        level = layer % max(1, math.ceil(math.log2(n_qubits)))
        stride = 2 ** level
        edges = []
        block = 2 * stride
        for base in range(0, n_qubits, block):
            a = base
            b = base + stride
            if b < n_qubits:
                edges.append((a, b))
        return edges

    if family == "symmetry_xy":
        start = layer % 2
        return [(i, i + 1) for i in range(start, n_qubits - 1, 2)]

    raise ValueError(f"Unknown family: {family}")


def parameter_count(family, n_qubits, depth):
    if family in {"hea_line", "hea_ring", "brickwall", "ttn"}:
        return 3 * n_qubits * depth

    if family == "symmetry_xy":
        return sum(
            n_qubits + len(edges_for_family(family, n_qubits, layer))
            for layer in range(depth)
        )

    raise ValueError(f"Unknown family: {family}")


def apply_ansatz(theta, family, n_qubits, depth):
    idx = 0

    for layer in range(depth):
        if family in {"hea_line", "hea_ring", "brickwall", "ttn"}:
            for q in range(n_qubits):
                qml.Rot(theta[idx], theta[idx + 1], theta[idx + 2], wires=q)
                idx += 3

            for a, b in edges_for_family(family, n_qubits, layer):
                qml.CNOT(wires=[a, b])

        elif family == "symmetry_xy":
            for q in range(n_qubits):
                qml.RZ(theta[idx], wires=q)
                idx += 1

            for a, b in edges_for_family(family, n_qubits, layer):
                angle = theta[idx]
                qml.IsingXX(angle, wires=[a, b])
                qml.IsingYY(angle, wires=[a, b])
                idx += 1

        else:
            raise ValueError(f"Unknown family: {family}")

    assert idx == len(theta)


def random_parameters(family, n_qubits, depth, local_rng):
    size = parameter_count(family, n_qubits, depth)
    values = local_rng.uniform(0, 2 * np.pi, size=size)
    return pnp.array(values, requires_grad=True)


def interaction_graph(family, n_qubits, depth):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_qubits))

    for layer in range(depth):
        graph.add_edges_from(edges_for_family(family, n_qubits, layer))

    return graph


def graph_descriptors(graph):
    n = graph.number_of_nodes()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    connected = nx.is_connected(graph) if n > 0 else False

    return {
        "connectivity_edges": graph.number_of_edges(),
        "connectivity_density": nx.density(graph) if n > 1 else 0.0,
        "mean_degree": degrees.mean() if len(degrees) else 0.0,
        "max_degree": degrees.max() if len(degrees) else 0.0,
        "degree_std": degrees.std() if len(degrees) else 0.0,
        "n_components": nx.number_connected_components(graph),
        "avg_shortest_path": (
            nx.average_shortest_path_length(graph)
            if connected and n > 1 else np.nan
        ),
        "diameter": (
            nx.diameter(graph)
            if connected and n > 1 else np.nan
        ),
        "clustering_coefficient": (
            nx.average_clustering(graph)
            if n > 1 else 0.0
        ),
    }


def resource_descriptors(family, n_qubits, depth):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(theta):
        apply_ansatz(theta, family, n_qubits, depth)
        return qml.expval(qml.PauliZ(0))

    theta = pnp.zeros(
        parameter_count(family, n_qubits, depth),
        requires_grad=True,
    )

    specs = qml.specs(circuit)(theta)
    resources = specs["resources"]

    two_qubit_gate_count = sum(
        count
        for gate_name, count in resources.gate_types.items()
        if gate_name in {"CNOT", "IsingXX", "IsingYY"}
    )

    return {
        "parameter_count": parameter_count(family, n_qubits, depth),
        "compiled_depth": resources.depth,
        "total_gate_count": resources.num_gates,
        "two_qubit_gate_count": two_qubit_gate_count,
    }


def make_state_qnode(family, n_qubits, depth):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state_circuit(theta):
        apply_ansatz(theta, family, n_qubits, depth)
        return qml.state()

    return state_circuit


def sample_states(family, n_qubits, depth, n_samples, local_rng):
    qnode = make_state_qnode(family, n_qubits, depth)
    states = []

    for _ in range(n_samples):
        theta = random_parameters(family, n_qubits, depth, local_rng)
        states.append(np.asarray(qnode(theta), dtype=complex))

    return np.stack(states)


def kl_expressibility_loss(
    states,
    n_qubits,
    n_pairs=300,
    n_bins=40,
    local_rng=None,
):
    if local_rng is None:
        local_rng = np.random.default_rng()

    n_states = len(states)
    first = local_rng.integers(0, n_states, size=n_pairs)
    second = local_rng.integers(0, n_states, size=n_pairs)

    same = first == second
    while np.any(same):
        second[same] = local_rng.integers(
            0, n_states, size=np.sum(same)
        )
        same = first == second

    fidelities = np.abs(
        np.sum(np.conj(states[first]) * states[second], axis=1)
    ) ** 2

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    empirical, _ = np.histogram(fidelities, bins=bins)
    empirical = empirical.astype(float)
    empirical /= empirical.sum()

    hilbert_dim = 2 ** n_qubits
    left = bins[:-1]
    right = bins[1:]

    haar = (
        (1 - left) ** (hilbert_dim - 1)
        - (1 - right) ** (hilbert_dim - 1)
    )
    haar /= haar.sum()

    eps = 1e-12
    return float(entropy(empirical + eps, haar + eps))


def reduced_density_one_qubit(state, qubit, n_qubits):
    tensor = state.reshape([2] * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    return moved @ moved.conj().T


def meyer_wallach(state, n_qubits):
    purities = []

    for qubit in range(n_qubits):
        rho = reduced_density_one_qubit(state, qubit, n_qubits)
        purities.append(np.real(np.trace(rho @ rho)))

    return float(2.0 * (1.0 - np.mean(purities)))


def entangling_capability(states, n_qubits):
    values = [
        meyer_wallach(state, n_qubits)
        for state in states
    ]
    return float(np.mean(values)), float(np.std(values))


def trainability_proxy(
    family,
    n_qubits,
    depth,
    n_samples,
    local_rng,
):
    dev = qml.device("default.qubit", wires=n_qubits)

    coeffs = [1.0 / n_qubits] * n_qubits
    observables = [qml.PauliZ(i) for i in range(n_qubits)]
    hamiltonian = qml.Hamiltonian(coeffs, observables)

    @qml.qnode(dev, interface="autograd")
    def cost(theta):
        apply_ansatz(theta, family, n_qubits, depth)
        return qml.expval(hamiltonian)

    gradient_function = qml.grad(cost)
    gradients = []

    for _ in range(n_samples):
        theta = random_parameters(
            family, n_qubits, depth, local_rng
        )
        gradient = np.asarray(
            gradient_function(theta),
            dtype=float,
        )
        gradients.append(gradient)

    flat = np.concatenate(gradients)
    variance = float(np.var(flat))

    return {
        "gradient_variance": variance,
        "log10_gradient_variance": float(
            np.log10(variance + 1e-20)
        ),
        "mean_abs_gradient": float(np.mean(np.abs(flat))),
        "near_zero_gradient_fraction": float(
            np.mean(np.abs(flat) < 1e-10)
        ),
    }


test_family = "brickwall"
test_n = 4
test_depth = 2
test_rng = np.random.default_rng(123)

test_states = sample_states(
    test_family,
    test_n,
    test_depth,
    8,
    test_rng,
)

print("State array shape:", test_states.shape)
print(
    "KL expressibility:",
    kl_expressibility_loss(
        test_states,
        test_n,
        n_pairs=50,
        n_bins=15,
        local_rng=test_rng,
    ),
)
print(
    "Meyer–Wallach:",
    entangling_capability(test_states, test_n),
)
print(
    "Trainability:",
    trainability_proxy(
        test_family,
        test_n,
        test_depth,
        n_samples=4,
        local_rng=test_rng,
    ),
)
print(
    "Resources:",
    resource_descriptors(
        test_family,
        test_n,
        test_depth,
    ),
)


records = []
start_time = time.time()

total = (
    len(FAMILIES)
    * len(QUBIT_COUNTS)
    * len(DEPTHS)
    * N_INSTANCES
)

progress = tqdm(total=total)

for family in FAMILIES:
    for n_qubits in QUBIT_COUNTS:
        for depth in DEPTHS:
            structural = {}
            structural.update(
                resource_descriptors(
                    family,
                    n_qubits,
                    depth,
                )
            )
            structural.update(
                graph_descriptors(
                    interaction_graph(
                        family,
                        n_qubits,
                        depth,
                    )
                )
            )

            for instance in range(N_INSTANCES):
                local_seed = (
                    SEED
                    + 100000 * FAMILIES.index(family)
                    + 1000 * n_qubits
                    + 100 * depth
                    + instance
                )
                local_rng = np.random.default_rng(local_seed)

                states = sample_states(
                    family,
                    n_qubits,
                    depth,
                    N_STATE_SAMPLES,
                    local_rng,
                )

                expressibility = kl_expressibility_loss(
                    states,
                    n_qubits,
                    n_pairs=N_PAIR_SAMPLES,
                    n_bins=N_BINS,
                    local_rng=local_rng,
                )

                ent_mean, ent_std = entangling_capability(
                    states,
                    n_qubits,
                )

                trainability = trainability_proxy(
                    family,
                    n_qubits,
                    depth,
                    N_GRAD_SAMPLES,
                    local_rng,
                )

                records.append({
                    "family": family,
                    "n_qubits": n_qubits,
                    "depth_setting": depth,
                    "instance": instance,
                    "seed": local_seed,
                    "expressibility_kl": expressibility,
                    "meyer_wallach_mean": ent_mean,
                    "meyer_wallach_std": ent_std,
                    **trainability,
                    **structural,
                })

                progress.update(1)

progress.close()

df = pd.DataFrame(records)
elapsed = time.time() - start_time

raw_path = OUTPUT_DIR / "pqc_descriptor_dataset.csv"
df.to_csv(raw_path, index=False)

print(f"Created {len(df)} rows in {elapsed / 60:.1f} minutes")
print("Saved:", raw_path)
display(df.head())


print("Dataset shape:", df.shape)

print("\nMissing values:")
display(
    df.isna()
      .sum()
      .sort_values(ascending=False)
      .head(12)
)

descriptor_cols = [
    "expressibility_kl",
    "meyer_wallach_mean",
    "log10_gradient_variance",
    "parameter_count",
    "compiled_depth",
    "two_qubit_gate_count",
    "connectivity_density",
    "mean_degree",
    "avg_shortest_path",
]

print("\nDescriptor ranges:")
display(df[descriptor_cols].describe().T)


group_cols = [
    "family",
    "n_qubits",
    "depth_setting",
]

numeric_cols = (
    df.select_dtypes(include=np.number)
      .columns
      .tolist()
)

numeric_cols = [
    column
    for column in numeric_cols
    if column not in {"instance", "seed"}
]

design_df = (
    df.groupby(group_cols, as_index=False)[numeric_cols]
      .mean()
)

design_path = OUTPUT_DIR / "pqc_design_level_dataset.csv"
design_df.to_csv(design_path, index=False)

print("Design-level rows:", len(design_df))
display(design_df.head())


analysis_features = [
    "expressibility_kl",
    "meyer_wallach_mean",
    "log10_gradient_variance",
    "parameter_count",
    "compiled_depth",
    "two_qubit_gate_count",
    "connectivity_density",
    "mean_degree",
    "avg_shortest_path",
]

correlation = design_df[analysis_features].corr(method="spearman")
correlation.to_csv(
    OUTPUT_DIR / "spearman_correlation_matrix.csv"
)

plt.figure(figsize=(10, 8))
plt.imshow(correlation, aspect="auto")
plt.colorbar(label="Spearman correlation")
plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90,
)
plt.yticks(
    range(len(correlation.index)),
    correlation.index,
)
plt.title("Descriptor correlation matrix")
plt.tight_layout()
plt.show()

display(correlation.round(2))


quantum_features = [
    "expressibility_kl",
    "meyer_wallach_mean",
    "log10_gradient_variance",
]

full_features = analysis_features


def run_embedding(data, features, prefix):
    clean = data.dropna(subset=features).copy()
    standardized = StandardScaler().fit_transform(
        clean[features]
    )

    pca = PCA(
        n_components=min(len(features), 6),
        random_state=SEED,
    )
    pca_coordinates = pca.fit_transform(standardized)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, max(3, len(clean) - 1)),
        min_dist=0.15,
        metric="euclidean",
        random_state=SEED,
    )
    umap_coordinates = reducer.fit_transform(standardized)

    scores = clean[
        ["family", "n_qubits", "depth_setting"]
    ].copy()

    scores["PC1"] = pca_coordinates[:, 0]
    scores["PC2"] = pca_coordinates[:, 1]
    scores["UMAP1"] = umap_coordinates[:, 0]
    scores["UMAP2"] = umap_coordinates[:, 1]

    scores.to_csv(
        OUTPUT_DIR / f"{prefix}_embedding_scores.csv",
        index=False,
    )

    loadings = pd.DataFrame(
        pca.components_.T,
        index=features,
        columns=[
            f"PC{i + 1}"
            for i in range(pca.n_components_)
        ],
    )

    loadings.to_csv(
        OUTPUT_DIR / f"{prefix}_pca_loadings.csv"
    )

    return (
        clean,
        standardized,
        pca,
        pca_coordinates,
        umap_coordinates,
        loadings,
    )


(
    q_clean,
    q_standardized,
    q_pca,
    q_pca_coordinates,
    q_umap_coordinates,
    q_loadings,
) = run_embedding(
    design_df,
    quantum_features,
    "quantum_only",
)

(
    full_clean,
    full_standardized,
    full_pca,
    full_pca_coordinates,
    full_umap_coordinates,
    full_loadings,
) = run_embedding(
    design_df,
    full_features,
    "full",
)

print(
    "Quantum-only explained variance:",
    q_pca.explained_variance_ratio_,
)
print(
    "Full-space explained variance:",
    full_pca.explained_variance_ratio_,
)

display(q_loadings.round(3))


def plot_embedding(
    clean,
    coordinates,
    title,
    xlabel,
    ylabel,
    color_column="family",
):
    plt.figure(figsize=(8, 6))

    categories = clean[color_column].astype(str).unique()

    for category in categories:
        mask = (
            clean[color_column].astype(str).values
            == category
        )
        plt.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=55,
            alpha=0.85,
            label=category,
        )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.show()


plot_embedding(
    q_clean,
    q_pca_coordinates,
    "Quantum-only PCA by family",
    f"PC1 ({q_pca.explained_variance_ratio_[0]:.1%})",
    f"PC2 ({q_pca.explained_variance_ratio_[1]:.1%})",
)

plot_embedding(
    q_clean,
    q_umap_coordinates,
    "Quantum-only UMAP by family",
    "UMAP1",
    "UMAP2",
)

plot_embedding(
    q_clean,
    q_pca_coordinates,
    "Quantum-only PCA by depth",
    f"PC1 ({q_pca.explained_variance_ratio_[0]:.1%})",
    f"PC2 ({q_pca.explained_variance_ratio_[1]:.1%})",
    color_column="depth_setting",
)


confounders = [
    "n_qubits",
    "depth_setting",
    "parameter_count",
    "two_qubit_gate_count",
]

latent_df = q_clean.copy()
latent_df["PC1"] = q_pca_coordinates[:, 0]
latent_df["PC2"] = q_pca_coordinates[:, 1]

results = []
X_confounders = latent_df[confounders].values

for target in ["PC1", "PC2"]:
    model = LinearRegression().fit(
        X_confounders,
        latent_df[target].values,
    )
    predictions = model.predict(X_confounders)

    row = {
        "latent_axis": target,
        "R2_from_structural_scale": r2_score(
            latent_df[target],
            predictions,
        ),
    }

    for name, coefficient in zip(
        confounders,
        model.coef_,
    ):
        row[f"coef_{name}"] = coefficient

    results.append(row)

confound_results = pd.DataFrame(results)
confound_results.to_csv(
    OUTPUT_DIR / "latent_axis_confound_regression.csv",
    index=False,
)

display(confound_results)


controlled_rows = []

for (n_qubits, depth), block in design_df.groupby(
    ["n_qubits", "depth_setting"]
):
    for feature in quantum_features:
        values = block.set_index("family")[feature]

        controlled_rows.append({
            "n_qubits": n_qubits,
            "depth": depth,
            "descriptor": feature,
            "between_family_std": values.std(),
            "between_family_range": (
                values.max() - values.min()
            ),
            "minimum_value_family": values.idxmin(),
            "maximum_value_family": values.idxmax(),
        })

controlled_df = pd.DataFrame(controlled_rows)
controlled_df.to_csv(
    OUTPUT_DIR / "controlled_family_comparisons.csv",
    index=False,
)

display(controlled_df.head(15))


residual_df = design_df.dropna(
    subset=quantum_features + confounders
).copy()

for feature in quantum_features:
    model = LinearRegression().fit(
        residual_df[confounders],
        residual_df[feature],
    )

    residual_df[f"resid_{feature}"] = (
        residual_df[feature]
        - model.predict(residual_df[confounders])
    )

residual_features = [
    f"resid_{feature}"
    for feature in quantum_features
]

X_residual = StandardScaler().fit_transform(
    residual_df[residual_features]
)

residual_pca = PCA(
    n_components=3,
    random_state=SEED,
)

residual_pca_coordinates = residual_pca.fit_transform(
    X_residual
)

residual_umap_coordinates = umap.UMAP(
    n_components=2,
    n_neighbors=min(15, max(3, len(residual_df) - 1)),
    min_dist=0.15,
    random_state=SEED,
).fit_transform(X_residual)

print(
    "Residual PCA explained variance:",
    residual_pca.explained_variance_ratio_,
)

plot_embedding(
    residual_df,
    residual_pca_coordinates,
    "Residualized PCA by family",
    f"PC1 ({residual_pca.explained_variance_ratio_[0]:.1%})",
    f"PC2 ({residual_pca.explained_variance_ratio_[1]:.1%})",
)

plot_embedding(
    residual_df,
    residual_umap_coordinates,
    "Residualized UMAP by family",
    "UMAP1",
    "UMAP2",
)


cluster_rows = []

spaces = {
    "quantum_only": q_standardized,
    "residualized": X_residual,
}

for space_name, matrix in spaces.items():
    max_k = min(8, len(matrix) - 1)

    for k in range(2, max_k + 1):
        labels = KMeans(
            n_clusters=k,
            n_init=50,
            random_state=SEED,
        ).fit_predict(matrix)

        cluster_rows.append({
            "space": space_name,
            "k": k,
            "silhouette": silhouette_score(
                matrix,
                labels,
            ),
        })

cluster_quality = pd.DataFrame(cluster_rows)
cluster_quality.to_csv(
    OUTPUT_DIR / "cluster_quality.csv",
    index=False,
)

display(cluster_quality)


def pareto_mask(costs):
    costs = np.asarray(costs, dtype=float)
    efficient = np.ones(len(costs), dtype=bool)

    for i in range(len(costs)):
        for j in range(len(costs)):
            if i == j:
                continue

            dominates = (
                np.all(costs[j] <= costs[i])
                and np.any(costs[j] < costs[i])
            )

            if dominates:
                efficient[i] = False
                break

    return efficient


pareto_df = design_df.copy()

cost_matrix = np.column_stack([
    pareto_df["expressibility_kl"].values,
    -pareto_df["meyer_wallach_mean"].values,
    -pareto_df["log10_gradient_variance"].values,
])

pareto_df["pareto_quantum"] = pareto_mask(
    cost_matrix
)

pareto_df.to_csv(
    OUTPUT_DIR / "pareto_designs.csv",
    index=False,
)

print(
    "Pareto-optimal designs:",
    pareto_df["pareto_quantum"].sum(),
)

display(
    pareto_df.loc[
        pareto_df["pareto_quantum"],
        [
            "family",
            "n_qubits",
            "depth_setting",
            "expressibility_kl",
            "meyer_wallach_mean",
            "log10_gradient_variance",
            "parameter_count",
            "two_qubit_gate_count",
        ],
    ].sort_values(
        ["n_qubits", "depth_setting"]
    )
)


stability_rows = []

for n_qubits, block in design_df.groupby("n_qubits"):
    clean = block.dropna(subset=quantum_features)

    matrix = StandardScaler().fit_transform(
        clean[quantum_features]
    )

    model = PCA(
        n_components=3,
        random_state=SEED,
    ).fit(matrix)

    stability_rows.append({
        "n_qubits": n_qubits,
        "pc1_variance": model.explained_variance_ratio_[0],
        "pc2_variance": model.explained_variance_ratio_[1],
        "pc1_plus_pc2": model.explained_variance_ratio_[:2].sum(),
    })

stability_df = pd.DataFrame(stability_rows)
stability_df.to_csv(
    OUTPUT_DIR / "pca_stability_across_qubit_counts.csv",
    index=False,
)

display(stability_df)


two_component_variance = float(
    q_pca.explained_variance_ratio_[:2].sum()
)

pc1_r2 = float(
    confound_results.loc[
        confound_results["latent_axis"] == "PC1",
        "R2_from_structural_scale",
    ].iloc[0]
)

pc2_r2 = float(
    confound_results.loc[
        confound_results["latent_axis"] == "PC2",
        "R2_from_structural_scale",
    ].iloc[0]
)

best_cluster = (
    cluster_quality
    .sort_values("silhouette", ascending=False)
    .iloc[0]
    .to_dict()
)

summary = {
    "n_raw_rows": int(len(df)),
    "n_design_points": int(len(design_df)),
    "quantum_pca_two_component_variance": two_component_variance,
    "pc1_R2_from_structural_scale": pc1_r2,
    "pc2_R2_from_structural_scale": pc2_r2,
    "best_exploratory_cluster_result": best_cluster,
    "pareto_design_count": int(
        pareto_df["pareto_quantum"].sum()
    ),
}

with open(
    OUTPUT_DIR / "analysis_summary.json",
    "w",
) as file:
    json.dump(summary, file, indent=2)

print(json.dumps(summary, indent=2))

print("\nInterpretation:")

if two_component_variance >= 0.85:
    print(
        "- The quantum descriptors are strongly compressible "
        "into two dimensions."
    )
else:
    print(
        "- More than two latent dimensions may be necessary."
    )

if pc1_r2 >= 0.70:
    print(
        "- PC1 is largely explained by structural scale."
    )
else:
    print(
        "- PC1 retains substantial variation beyond scale."
    )

if pc2_r2 >= 0.70:
    print(
        "- PC2 is largely explained by structural scale."
    )
else:
    print(
        "- PC2 may contain architecture-specific information."
    )

print(
    "- Persistent family separation in the residualized plots "
    "is the strongest evidence for an architecture manifold."
)


archive_path = shutil.make_archive(
    "pqc_latent_results",
    "zip",
    OUTPUT_DIR,
)

print("Archive created:", archive_path)

try:
    from google.colab import files
    print("Colab download skipped in GitHub Actions")
except Exception:
    print(
        "Not running in Colab. Retrieve:",
        archive_path,
    )