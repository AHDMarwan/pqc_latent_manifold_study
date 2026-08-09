from __future__ import annotations

import math
from math import comb

import numpy as np
from .config import DEFAULT_FAMILIES


def validate_family(family: str) -> None:
    if family not in DEFAULT_FAMILIES:
        raise ValueError(f"Unknown family: {family!r}. Expected one of {DEFAULT_FAMILIES}.")


def edges_for_family(family: str, n_qubits: int, layer: int) -> list[tuple[int, int]]:
    """Return the two-qubit interaction edges used at a given layer."""
    validate_family(family)

    if family == "hea_line":
        return [(i, i + 1) for i in range(n_qubits - 1)]

    if family == "hea_ring":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
        if n_qubits > 2:
            edges.append((n_qubits - 1, 0))
        return edges

    if family in {"brickwall", "symmetry_xy"}:
        start = layer % 2
        return [(i, i + 1) for i in range(start, n_qubits - 1, 2)]

    if family == "ttn":
        # This follows the notebook's repeated level schedule.  It is a
        # TTN-inspired connectivity pattern rather than a strict one-pass TTN.
        level = layer % max(1, math.ceil(math.log2(n_qubits)))
        stride = 2**level
        block = 2 * stride
        edges: list[tuple[int, int]] = []
        for base in range(0, n_qubits, block):
            a = base
            b = base + stride
            if b < n_qubits:
                edges.append((a, b))
        return edges

    raise AssertionError("unreachable")


def symmetry_excitation_count(n_qubits: int) -> int:
    """Half-filled sector used by the symmetry-preserving ansatz."""
    return n_qubits // 2


def initial_bitstring(family: str, n_qubits: int) -> np.ndarray:
    """Initial computational-basis state for a circuit family.

    The original notebook started every circuit from |0...0>.  That makes the
    number-conserving XY ansatz trivial.  Here the symmetry-preserving family
    starts in a non-zero, approximately half-filled excitation sector.
    """
    validate_family(family)
    bits = np.zeros(n_qubits, dtype=int)
    if family == "symmetry_xy":
        k = symmetry_excitation_count(n_qubits)
        bits[::2][:k] = 1
        # For odd n, the slicing above can be shorter than k only in unusual
        # custom states; fall back to the first k positions if necessary.
        if int(bits.sum()) != k:
            bits[:] = 0
            bits[:k] = 1
    return bits


def prepare_initial_state(family: str, n_qubits: int) -> None:
    import pennylane as qml

    bits = initial_bitstring(family, n_qubits)
    if np.any(bits):
        qml.BasisState(bits, wires=range(n_qubits))


def effective_hilbert_dimension(family: str, n_qubits: int) -> int:
    """Dimension of the state space that the ansatz is intended to explore.

    Full-Hilbert-space Haar is appropriate for unconstrained families.  The
    symmetry-preserving XY circuit is compared with Haar random states inside
    its fixed-excitation sector, whose dimension is C(n, k).
    """
    validate_family(family)
    if family == "symmetry_xy":
        return comb(n_qubits, symmetry_excitation_count(n_qubits))
    return 2**n_qubits


def parameter_count(family: str, n_qubits: int, depth: int) -> int:
    validate_family(family)
    if family in {"hea_line", "hea_ring", "brickwall", "ttn"}:
        return 3 * n_qubits * depth
    if family == "symmetry_xy":
        return sum(
            n_qubits + len(edges_for_family(family, n_qubits, layer))
            for layer in range(depth)
        )
    raise AssertionError("unreachable")


def apply_ansatz(theta, family: str, n_qubits: int, depth: int) -> None:
    """Apply only the trainable ansatz gates; state preparation is separate."""
    import pennylane as qml

    validate_family(family)
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
                # exp[-i angle/2 (XX + YY)] because XX and YY commute here.
                qml.IsingXX(angle, wires=[a, b])
                qml.IsingYY(angle, wires=[a, b])
                idx += 1

    if idx != len(theta):
        raise RuntimeError(f"Consumed {idx} parameters but received {len(theta)}")


def apply_circuit(theta, family: str, n_qubits: int, depth: int) -> None:
    prepare_initial_state(family, n_qubits)
    apply_ansatz(theta, family, n_qubits, depth)


def random_parameters(
    family: str,
    n_qubits: int,
    depth: int,
    rng: np.random.Generator,
):
    from pennylane import numpy as pnp

    size = parameter_count(family, n_qubits, depth)
    values = rng.uniform(0.0, 2.0 * np.pi, size=size)
    return pnp.array(values, requires_grad=True)
