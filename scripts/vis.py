import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Optional
from scipy.stats import pearsonr

plt.rcParams["font.family"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.alpha"] = 0.3


def plot_influence_distribution(
    influences: Dict[int, List[float]],
    save_dir: str,
    test_idx: int,
    top_n: int = 50,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if test_idx not in influences:
        raise ValueError(f"test_idx {test_idx} not found in influences")

    scores = np.array(influences[test_idx], dtype=np.float32)
    sorted_scores = np.sort(scores)
    helpful = sorted_scores[:top_n]
    harmful = sorted_scores[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.hist(helpful, bins=20, alpha=0.5, label="Helpful (more negative)", color="#2ca02c")
    ax.hist(harmful, bins=20, alpha=0.5, label="Harmful (more positive)", color="#d62728")

    ax.set_xlabel("Influence Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Influence Score Distribution (Test Sample {test_idx})", pad=15)
    ax.legend()
    ax.grid(True, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path = os.path.join(save_dir, f"influence_distribution_test_{test_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    logging.info(f"Saved influence distribution: {save_path}")


def plot_influence_vs_leave_one_out(
    actual_loss_changes: List[float],
    predicted_influences: List[float],
    save_dir: str,
    test_idx: int,
    top_k: Optional[int] = 500,
    model_type: str = "Approx",
) -> None:
    """
    Scatter plot comparing:
      X: actual change in test loss from leave-one-out retraining (LOO)
      Y: predicted influence scores from influence functions
    """
    os.makedirs(save_dir, exist_ok=True)

    actual = np.asarray(actual_loss_changes, dtype=np.float32)
    predicted = np.asarray(predicted_influences, dtype=np.float32)
    if actual.shape[0] != predicted.shape[0]:
        n = min(actual.shape[0], predicted.shape[0])
        actual = actual[:n]
        predicted = predicted[:n]

    if top_k is not None and top_k < len(actual):
        top_idx = np.argsort(np.abs(predicted))[-top_k:]
        actual = actual[top_idx]
        predicted = predicted[top_idx]

    corr, p_value = pearsonr(actual, predicted)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(actual, predicted, alpha=0.6, s=20, color="#1f77b4", edgecolors="none")

    min_val = float(min(np.min(actual), np.min(predicted)))
    max_val = float(max(np.max(actual), np.max(predicted)))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label=f"Ideal (y=x), r={corr:.3f} (p={p_value:.2e})",
    )

    ax.set_xlabel("Actual Change in Loss (LOO)")
    ax.set_ylabel("Predicted Influence (IF)")
    ax.set_title(f"{model_type}: Influence vs. Leave-One-Out (Test {test_idx})", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path = os.path.join(save_dir, f"influence_vs_loo_test_{test_idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    logging.info(f"Saved IF vs LOO plot: {save_path}")
    logging.info(f"Pearson r={corr:.3f}, p={p_value:.2e}")
