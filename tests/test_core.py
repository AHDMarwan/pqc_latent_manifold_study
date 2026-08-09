from math import comb

import numpy as np

from pqc_latent.analysis import pareto_mask, participation_ratio_dimension
from pqc_latent.circuits import (
    effective_hilbert_dimension,
    initial_bitstring,
    parameter_count,
    symmetry_excitation_count,
)
from pqc_latent.descriptors import haar_fidelity_bin_probabilities


def test_symmetry_initial_state_is_nonzero_and_half_filled():
    bits = initial_bitstring("symmetry_xy", 6)
    assert bits.sum() == symmetry_excitation_count(6) == 3


def test_symmetry_haar_dimension_uses_sector_dimension():
    assert effective_hilbert_dimension("symmetry_xy", 6) == comb(6, 3)
    assert effective_hilbert_dimension("hea_line", 6) == 2**6


def test_parameter_counts_match_notebook_conventions():
    assert parameter_count("hea_line", 4, 2) == 24
    assert parameter_count("brickwall", 4, 2) == 24
    assert parameter_count("symmetry_xy", 4, 2) == 4 + 2 + 4 + 1


def test_haar_histogram_probabilities_normalize():
    bins = np.linspace(0, 1, 21)
    probs = haar_fidelity_bin_probabilities(16, bins)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)


def test_pareto_mask():
    costs = np.array([[1, 1], [2, 2], [0.5, 3]])
    mask = pareto_mask(costs)
    assert mask.tolist() == [True, False, True]


def test_participation_ratio_for_isotropic_three_dimensional_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10000, 3))
    dimension = participation_ratio_dimension(x)
    assert 2.9 < dimension <= 3.0
