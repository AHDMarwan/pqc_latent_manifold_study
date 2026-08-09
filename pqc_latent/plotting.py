from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_embedding(
    clean: pd.DataFrame,
    coordinates: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    color_column: str = "family",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    categories = clean[color_column].astype(str).unique()
    for category in categories:
        mask = clean[color_column].astype(str).to_numpy() == category
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=55,
            alpha=0.85,
            label=category,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_correlation(correlation: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=90)
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title("Descriptor correlation matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
