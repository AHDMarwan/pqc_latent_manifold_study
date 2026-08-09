from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


DEFAULT_FAMILIES = (
    "hea_line",
    "hea_ring",
    "brickwall",
    "ttn",
    "symmetry_xy",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the PQC latent-geometry benchmark."""

    qubit_counts: tuple[int, ...] = (4, 6, 8)
    depths: tuple[int, ...] = (1, 2, 3, 4, 5)
    families: tuple[str, ...] = DEFAULT_FAMILIES

    # Fast proof-of-concept defaults.
    n_instances: int = 3
    n_state_samples: int = 24
    n_pair_samples: int = 300
    n_grad_samples: int = 20
    n_bins: int = 40

    # Two non-constant trainability observables are reported.  The second is
    # used as the primary trainability coordinate in the latent-space analysis.
    trainability_observables: tuple[str, ...] = ("local_z", "staggered_z")
    primary_trainability_observable: str = "staggered_z"

    seed: int = 42
    output_dir: Path = Path("pqc_latent_results")

    def validate(self) -> None:
        if not self.qubit_counts:
            raise ValueError("qubit_counts must not be empty")
        if not self.depths:
            raise ValueError("depths must not be empty")
        if not self.families:
            raise ValueError("families must not be empty")
        if any(n < 2 for n in self.qubit_counts):
            raise ValueError("all qubit counts must be >= 2")
        if any(d < 1 for d in self.depths):
            raise ValueError("all depths must be >= 1")
        if self.n_instances < 1:
            raise ValueError("n_instances must be >= 1")
        if self.n_state_samples < 2:
            raise ValueError("n_state_samples must be >= 2")
        if self.n_pair_samples < 1:
            raise ValueError("n_pair_samples must be >= 1")
        if self.n_grad_samples < 2:
            raise ValueError("n_grad_samples must be >= 2")
        if self.n_bins < 5:
            raise ValueError("n_bins must be >= 5")
        if self.primary_trainability_observable not in self.trainability_observables:
            raise ValueError(
                "primary_trainability_observable must appear in trainability_observables"
            )

    @property
    def primary_trainability_feature(self) -> str:
        return f"log10_gradient_variance_mean_{self.primary_trainability_observable}"

    @classmethod
    def publication(cls, **overrides: Any) -> "ExperimentConfig":
        """Publication-scale Monte Carlo settings proposed in the notebook."""
        base = cls(
            n_instances=10,
            n_state_samples=100,
            n_pair_samples=5000,
            n_grad_samples=100,
        )
        return replace(base, **overrides)

    def with_overrides(self, **overrides: Any) -> "ExperimentConfig":
        return replace(self, **overrides)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data
