# PQC Latent-Manifold Study — Python package

This directory is a modular Python conversion of the original
`pqc_latent_manifold_study.ipynb` notebook.

## Structure

- `pqc_latent/config.py`: experiment settings.
- `pqc_latent/circuits.py`: circuit families and state preparation.
- `pqc_latent/descriptors.py`: structural, expressibility, entanglement and trainability descriptors.
- `pqc_latent/dataset.py`: Monte Carlo dataset generation and aggregation.
- `pqc_latent/analysis.py`: PCA/UMAP, confound analysis, residualization, clustering, Pareto and stability analyses.
- `pqc_latent/plotting.py`: non-interactive figure generation.
- `pqc_latent/pipeline.py`: end-to-end experiment orchestration.
- `run_experiment.py`: command-line entry point.
- `tests/test_core.py`: basic scientific and numerical sanity checks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows PowerShell use `.venv\\Scripts\\Activate.ps1` instead.

## Run

Fast notebook-equivalent defaults:

```bash
python run_experiment.py
```

Small smoke test:

```bash
python run_experiment.py --smoke --no-umap
```

Publication-scale Monte Carlo settings:

```bash
python run_experiment.py --publication
```

## Scientific corrections relative to the notebook

1. **Non-trivial symmetry-sector initialization.** `symmetry_xy` starts from a
   half-filled computational basis state instead of `|0...0>`, because a
   number-conserving XY circuit leaves the vacuum trivial.
2. **Symmetry-aware expressibility target.** Haar fidelity statistics for
   `symmetry_xy` use the fixed-excitation sector dimension `C(n, k)` rather
   than the full Hilbert-space dimension `2^n`.
3. **Non-constant trainability observables.** The original average
   magnetization is constant in a fixed-excitation sector. The package reports
   local `Z_0` and staggered magnetization gradient statistics instead.
4. **Per-parameter gradient variance.** Gradient variance is estimated across
   random initializations for each parameter before aggregation, rather than
   flattening all parameters and seeds together.
5. **Unique fidelity pairs.** Pair sampling does not count repeated state pairs
   as independent Monte Carlo evidence.
6. **Logical depth naming.** PennyLane `qml.specs` resources are reported as
   `logical_depth`; `compiled_depth` remains only as a backward-compatible
   alias because no hardware compilation target is specified.
7. **Layer-aware topology descriptors.** Additional descriptors retain some
   information lost when all layer interactions are collapsed into one graph.
8. **Stronger PCA stability check.** Stability across qubit counts includes
   principal-subspace angles and loading cosine similarity, not only explained
   variance.
9. **Budget-conditioned Pareto front.** A fixed `(n_qubits, depth)` Pareto
   result is exported in addition to the global Pareto front.
10. **Intrinsic-dimension diagnostic.** The summary reports the covariance
    participation-ratio dimension of the standardized quantum descriptor space.

## Important interpretation caveat

PCA and especially UMAP visual separation are exploratory evidence, not proof
of an intrinsic manifold. A publication should also report uncertainty over
Monte Carlo replicates, embedding stability across random seeds and UMAP
hyperparameters, cluster-label agreement metrics, and ideally additional
independent quantum descriptors or explicit intrinsic-dimension estimators.
