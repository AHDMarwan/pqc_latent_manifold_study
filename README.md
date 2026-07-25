# README.md

## Latent Geometry of Parameterized Quantum Circuits

### Overview

This experiment investigates whether **parameterized quantum circuits (PQCs)** exhibit an intrinsic low-dimensional organization based on their structural and quantum descriptors.

Instead of evaluating PQCs only through downstream tasks (e.g., classification accuracy or VQE performance), this study asks whether circuits can first be represented in a **descriptor space**, and whether that space possesses a stable latent geometry shared across different ansatz families.

The workflow builds a benchmark of multiple PQC architectures, computes intrinsic descriptors, and analyzes the resulting descriptor space using dimensionality reduction and statistical analysis.

---

# Research Objective

The main research question is:

> **Do different parameterized quantum circuit architectures organize into a stable low-dimensional latent space after controlling for circuit size and complexity?**

If such a latent space exists, PQCs could be characterized by a small number of latent coordinates instead of many independent descriptors.

---

# Circuit Families

The notebook generates several commonly used PQC architectures:

* Hardware-Efficient (Line)
* Hardware-Efficient (Ring)
* Brickwall
* Tree Tensor Network (TTN)
* Symmetry-Preserving XY Ansatz

Each family is evaluated for multiple:

* qubit counts
* circuit depths
* Monte Carlo repetitions

---

# Computed Descriptors

## Structural descriptors

* Number of parameters
* Compiled circuit depth
* Total gate count
* Two-qubit gate count
* Connectivity density
* Mean node degree
* Maximum node degree
* Degree standard deviation
* Number of connected components
* Average shortest path
* Graph diameter
* Clustering coefficient

---

## Quantum descriptors

### Expressibility

Measured using the **KL-divergence** between the circuit fidelity distribution and the Haar-random fidelity distribution.

Lower values indicate higher expressibility.

---

### Entangling capability

Measured using the **Meyer–Wallach global entanglement** averaged over randomly sampled parameters.

Higher values indicate stronger entanglement.

---

### Trainability proxy

Measured using the variance of parameter gradients over random initializations.

The notebook reports

* Gradient variance
* Log-gradient variance
* Mean absolute gradient
* Fraction of near-zero gradients

This serves as a proxy for barren plateau behavior.

---

# Experimental Pipeline

The notebook performs the following steps.

## 1. Generate PQCs

Generate every circuit family for multiple

* qubit counts
* depths
* random seeds

---

## 2. Compute descriptors

For every circuit compute

* structural descriptors
* expressibility
* entangling capability
* trainability proxy

---

## 3. Build dataset

All descriptor values are saved into

```
pqc_descriptor_dataset.csv
```

Monte Carlo repetitions are then averaged to obtain architecture-level descriptors

```
pqc_design_level_dataset.csv
```

---

## 4. Correlation analysis

Compute the Spearman correlation matrix between all descriptors.

Output

```
spearman_correlation_matrix.csv
```

---

## 5. Dimensionality reduction

Two latent spaces are constructed.

### Quantum-only latent space

Uses

* Expressibility
* Entanglement
* Trainability

### Full latent space

Uses

* Structural descriptors
* Quantum descriptors

Both are analyzed using

* PCA
* UMAP

Outputs include

* PCA coordinates
* PCA loadings
* UMAP coordinates

---

## 6. Confounding analysis

A key question is whether the latent space is simply a consequence of

* circuit depth
* parameter count
* gate count

Each principal component is regressed against these structural variables.

If the latent coordinates are almost completely explained by circuit size, then the apparent geometry is not architecture-specific.

---

## 7. Residualized latent space

To isolate architectural effects,

the influence of

* depth
* qubit count
* parameter count
* two-qubit gate count

is removed from each quantum descriptor.

PCA and UMAP are then recomputed on the residuals.

If family clustering remains, this provides stronger evidence for an intrinsic architectural manifold.

---

## 8. Controlled comparisons

For every

* qubit count
* circuit depth

the notebook compares descriptor values across circuit families.

This separates architectural effects from scaling effects.

---

## 9. Clustering

Exploratory clustering is performed using

* K-Means
* Silhouette score

to estimate whether descriptor space naturally separates into groups.

---

## 10. Pareto analysis

The notebook identifies circuits that simultaneously optimize

* expressibility
* entanglement
* trainability

No assumption is made that a single circuit is optimal in every objective.

---

## 11. Stability analysis

PCA is repeated independently for each qubit count.

The objective is to determine whether the latent geometry remains stable as system size increases.

---

# Outputs

The notebook exports

```
pqc_descriptor_dataset.csv

pqc_design_level_dataset.csv

spearman_correlation_matrix.csv

quantum_only_embedding_scores.csv

full_embedding_scores.csv

quantum_only_pca_loadings.csv

full_pca_loadings.csv

controlled_family_comparisons.csv

latent_axis_confound_regression.csv

cluster_quality.csv

pareto_designs.csv

pca_stability_across_qubit_counts.csv

analysis_summary.json
```

All outputs are automatically collected into

```
pqc_latent_results.zip
```

---

# Interpretation

The experiment is designed to answer four questions.

1. Do PQCs occupy a low-dimensional descriptor space?

2. Are latent coordinates determined mainly by circuit depth and size?

3. Does architecture contribute information beyond structural complexity?

4. Can a latent representation provide a compact description of PQC families?

---

# Recommended Full Experiment

For publication-quality results:

```
N_INSTANCES = 10
N_STATE_SAMPLES = 100
N_PAIR_SAMPLES = 5000
N_GRAD_SAMPLES = 100
```

This produces approximately

* 5 circuit families
* 3 qubit counts
* 5 circuit depths
* 10 Monte Carlo repetitions

for roughly **750 architecture evaluations**.

---

# Expected Contribution

If the residualized latent space remains stable across

* circuit families,
* qubit counts,
* and circuit depths,

this would support the hypothesis that parameterized quantum circuits possess an intrinsic **latent descriptor geometry**, providing a compact representation of PQC architectures that is independent of specific downstream applications.


Open to Collaboration and Contributions

This project is intended as an open research effort toward understanding the intrinsic geometry of parameterized quantum circuits. We welcome collaborations from researchers, students, and practitioners interested in quantum computing, quantum machine learning, optimization, and mathematical modeling. Contributions may include new PQC architectures, additional descriptor definitions, theoretical analyses, benchmarking datasets, validation on different quantum hardware or simulators, and improvements to the mathematical framework. Suggestions, discussions, bug reports, and pull requests are all encouraged. Our goal is to build a reproducible and extensible benchmark that can serve as a community resource for studying the latent structure of PQCs.

Contact

For questions, collaboration opportunities, or research discussions, please feel free to connect via LinkedIn:

Marwan Ait Haddou
[LinkedIn Profile](https://www.linkedin.com/in/marwan-ait-haddou-85b796120/)

Contributions, feedback, and collaborative research proposals are always welcome.

