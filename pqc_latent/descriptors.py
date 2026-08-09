from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np
from scipy.stats import entropy

from .circuits import (
    apply_ansatz,
    apply_circuit,
    edges_for_family,
    effective_hilbert_dimension,
    parameter_count,
    random_parameters,
)


def interaction_graph(family: str, n_qubits: int, depth: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n_qubits))
    for layer in range(depth):
        graph.add_edges_from(edges_for_family(family, n_qubits, layer))
    return graph


def graph_descriptors(graph: nx.Graph) -> dict[str, float]:
    n = graph.number_of_nodes()
    degrees = np.array([degree for _, degree in graph.degree()], dtype=float)
    connected = nx.is_connected(graph) if n > 0 else False

    components = list(nx.connected_components(graph)) if n else []
    largest_nodes = max(components, key=len) if components else set()
    largest = graph.subgraph(largest_nodes).copy() if largest_nodes else nx.Graph()
    lcc_n = largest.number_of_nodes()

    return {
        "connectivity_edges": float(graph.number_of_edges()),
        "connectivity_density": float(nx.density(graph) if n > 1 else 0.0),
        "mean_degree": float(degrees.mean() if len(degrees) else 0.0),
        "max_degree": float(degrees.max() if len(degrees) else 0.0),
        "degree_std": float(degrees.std() if len(degrees) else 0.0),
        "n_components": float(nx.number_connected_components(graph) if n else 0),
        "largest_component_fraction": float(lcc_n / n if n else 0.0),
        "global_efficiency": float(nx.global_efficiency(graph) if n > 1 else 0.0),
        "avg_shortest_path": float(
            nx.average_shortest_path_length(graph) if connected and n > 1 else np.nan
        ),
        "diameter": float(nx.diameter(graph) if connected and n > 1 else np.nan),
        "avg_shortest_path_lcc": float(
            nx.average_shortest_path_length(largest) if lcc_n > 1 else 0.0
        ),
        "diameter_lcc": float(nx.diameter(largest) if lcc_n > 1 else 0.0),
        "clustering_coefficient": float(nx.average_clustering(graph) if n > 1 else 0.0),
    }


def temporal_topology_descriptors(
    family: str,
    n_qubits: int,
    depth: int,
) -> dict[str, float]:
    """Layer-aware descriptors that a collapsed interaction graph loses."""
    layers = [edges_for_family(family, n_qubits, layer) for layer in range(depth)]
    counts = np.asarray([len(edges) for edges in layers], dtype=float)
    all_edges = [tuple(sorted(edge)) for layer in layers for edge in layer]
    unique_edges = set(all_edges)
    total_possible = n_qubits * (n_qubits - 1) / 2

    if not all_edges:
        reuse_fraction = 0.0
    else:
        reuse_fraction = 1.0 - len(unique_edges) / len(all_edges)

    return {
        "mean_edges_per_layer": float(counts.mean() if len(counts) else 0.0),
        "std_edges_per_layer": float(counts.std() if len(counts) else 0.0),
        "edge_coverage": float(len(unique_edges) / total_possible if total_possible else 0.0),
        "edge_reuse_fraction": float(reuse_fraction),
    }


def resource_descriptors(family: str, n_qubits: int, depth: int) -> dict[str, float]:
    """Logical tape resources reported by PennyLane.

    The notebook called this quantity ``compiled_depth``.  No hardware target
    or transpilation basis was specified, so ``logical_depth`` is the more
    accurate name.  A backward-compatible ``compiled_depth`` alias is retained.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(theta):
        apply_ansatz(theta, family, n_qubits, depth)
        return qml.expval(qml.PauliZ(0))

    theta = pnp.zeros(parameter_count(family, n_qubits, depth), requires_grad=True)
    specs = qml.specs(circuit)(theta)
    resources = specs["resources"]
    two_qubit_gate_count = sum(
        count
        for gate_name, count in resources.gate_types.items()
        if gate_name in {"CNOT", "IsingXX", "IsingYY"}
    )

    logical_depth = float(resources.depth)
    return {
        "parameter_count": float(parameter_count(family, n_qubits, depth)),
        "logical_depth": logical_depth,
        "compiled_depth": logical_depth,
        "total_gate_count": float(resources.num_gates),
        "two_qubit_gate_count": float(two_qubit_gate_count),
    }


def structural_descriptors(family: str, n_qubits: int, depth: int) -> dict[str, float]:
    result: dict[str, float] = {}
    result.update(resource_descriptors(family, n_qubits, depth))
    result.update(graph_descriptors(interaction_graph(family, n_qubits, depth)))
    result.update(temporal_topology_descriptors(family, n_qubits, depth))
    return result


def make_state_qnode(family: str, n_qubits: int, depth: int):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def state_circuit(theta):
        apply_circuit(theta, family, n_qubits, depth)
        return qml.state()

    return state_circuit


def sample_states(
    family: str,
    n_qubits: int,
    depth: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    qnode = make_state_qnode(family, n_qubits, depth)
    states = []
    for _ in range(n_samples):
        theta = random_parameters(family, n_qubits, depth, rng)
        states.append(np.asarray(qnode(theta), dtype=complex))
    return np.stack(states)


def _sample_unique_pairs(
    n_states: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample unique unordered state pairs without pretending reuse is independent."""
    pairs = np.asarray(list(combinations(range(n_states), 2)), dtype=int)
    if len(pairs) == 0:
        raise ValueError("At least two states are required")
    if n_pairs >= len(pairs):
        return pairs
    selected = rng.choice(len(pairs), size=n_pairs, replace=False)
    return pairs[selected]


def fidelity_samples(
    states: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pairs = _sample_unique_pairs(len(states), n_pairs, rng)
    first = states[pairs[:, 0]]
    second = states[pairs[:, 1]]
    return np.abs(np.sum(np.conj(first) * second, axis=1)) ** 2


def haar_fidelity_bin_probabilities(hilbert_dim: int, bins: np.ndarray) -> np.ndarray:
    if hilbert_dim < 2:
        raise ValueError("hilbert_dim must be >= 2")
    left = bins[:-1]
    right = bins[1:]
    probabilities = (1.0 - left) ** (hilbert_dim - 1) - (1.0 - right) ** (
        hilbert_dim - 1
    )
    probabilities = np.clip(probabilities, 0.0, None)
    return probabilities / probabilities.sum()


def kl_expressibility_loss(
    states: np.ndarray,
    hilbert_dim: int,
    n_pairs: int = 300,
    n_bins: int = 40,
    rng: np.random.Generator | None = None,
) -> float:
    """Histogram KL(P_PQC || P_Haar) using the appropriate accessible dimension."""
    if rng is None:
        rng = np.random.default_rng()
    fidelities = fidelity_samples(states, n_pairs=n_pairs, rng=rng)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    empirical, _ = np.histogram(fidelities, bins=bins)
    empirical = empirical.astype(float)
    if empirical.sum() == 0:
        raise RuntimeError("No fidelity samples fell inside [0, 1]")
    empirical /= empirical.sum()
    haar = haar_fidelity_bin_probabilities(hilbert_dim, bins)
    eps = 1e-12
    return float(entropy(empirical + eps, haar + eps))


def reduced_density_one_qubit(state: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    tensor = state.reshape([2] * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    return moved @ moved.conj().T


def meyer_wallach(state: np.ndarray, n_qubits: int) -> float:
    purities = []
    for qubit in range(n_qubits):
        rho = reduced_density_one_qubit(state, qubit, n_qubits)
        purities.append(np.real(np.trace(rho @ rho)))
    return float(2.0 * (1.0 - np.mean(purities)))


def entangling_capability(states: np.ndarray, n_qubits: int) -> tuple[float, float]:
    values = np.asarray([meyer_wallach(state, n_qubits) for state in states], dtype=float)
    return float(values.mean()), float(values.std(ddof=1) if len(values) > 1 else 0.0)


def make_trainability_observable(kind: str, n_qubits: int):
    """Build a non-trivial observable, including inside a fixed-excitation sector."""
    import pennylane as qml

    if kind == "local_z":
        return qml.PauliZ(0)
    if kind == "staggered_z":
        coeffs = [((-1.0) ** i) / n_qubits for i in range(n_qubits)]
        return qml.Hamiltonian(coeffs, [qml.PauliZ(i) for i in range(n_qubits)])
    raise ValueError(f"Unknown trainability observable: {kind!r}")


def trainability_proxy(
    family: str,
    n_qubits: int,
    depth: int,
    n_samples: int,
    rng: np.random.Generator,
    observable_kind: str,
) -> dict[str, float]:
    """Estimate gradient statistics over random initializations.

    Variance is first computed per parameter across initializations and then
    aggregated.  This matches the usual barren-plateau question more closely
    than flattening parameters and random seeds into one vector.
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)
    observable = make_trainability_observable(observable_kind, n_qubits)

    @qml.qnode(dev, interface="autograd")
    def cost(theta):
        apply_circuit(theta, family, n_qubits, depth)
        return qml.expval(observable)

    gradient_function = qml.grad(cost)
    gradients = []
    for _ in range(n_samples):
        theta = random_parameters(family, n_qubits, depth, rng)
        gradients.append(np.asarray(gradient_function(theta), dtype=float))

    matrix = np.stack(gradients, axis=0)
    per_parameter_variance = np.var(matrix, axis=0, ddof=1)
    variance_mean = float(np.mean(per_parameter_variance))
    variance_median = float(np.median(per_parameter_variance))
    flat = matrix.reshape(-1)

    suffix = observable_kind
    return {
        f"gradient_variance_mean_{suffix}": variance_mean,
        f"gradient_variance_median_{suffix}": variance_median,
        f"log10_gradient_variance_mean_{suffix}": float(
            np.log10(variance_mean + 1e-20)
        ),
        f"mean_abs_gradient_{suffix}": float(np.mean(np.abs(flat))),
        f"near_zero_gradient_fraction_{suffix}": float(np.mean(np.abs(flat) < 1e-10)),
    }


def quantum_descriptors(
    family: str,
    n_qubits: int,
    depth: int,
    n_state_samples: int,
    n_pair_samples: int,
    n_bins: int,
    n_grad_samples: int,
    trainability_observables: tuple[str, ...],
    rng: np.random.Generator,
) -> dict[str, float]:
    states = sample_states(family, n_qubits, depth, n_state_samples, rng)
    hilbert_dim = effective_hilbert_dimension(family, n_qubits)
    ent_mean, ent_std = entangling_capability(states, n_qubits)

    result = {
        "effective_hilbert_dim": float(hilbert_dim),
        "expressibility_kl": kl_expressibility_loss(
            states,
            hilbert_dim=hilbert_dim,
            n_pairs=n_pair_samples,
            n_bins=n_bins,
            rng=rng,
        ),
        "meyer_wallach_mean": ent_mean,
        "meyer_wallach_std": ent_std,
    }
    for observable in trainability_observables:
        result.update(
            trainability_proxy(
                family,
                n_qubits,
                depth,
                n_samples=n_grad_samples,
                rng=rng,
                observable_kind=observable,
            )
        )
    return result
