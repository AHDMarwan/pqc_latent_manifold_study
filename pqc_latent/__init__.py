"""Latent-geometry benchmark for parameterized quantum circuits."""

from .config import ExperimentConfig
from .pipeline import analyze_descriptor_dataset, run_full_pipeline

__all__ = ["ExperimentConfig", "analyze_descriptor_dataset", "run_full_pipeline"]
