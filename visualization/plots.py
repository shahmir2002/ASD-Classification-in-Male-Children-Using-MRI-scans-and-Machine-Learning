"""
Visualisation utilities for the STRP 2 pipeline.

Generates publication-ready plots:
  - Accuracy / F1 heatmaps  (atlases × classifiers)
  - Confusion matrix grids
  - Modality comparison bar charts
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Style defaults ──────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
CMAP_HEAT = "YlOrRd"
FIG_DPI = 150


# ──────────────────────────────────────────────
#  Accuracy / F1 Heatmap
# ──────────────────────────────────────────────

def plot_accuracy_heatmap(
    results_df: pd.DataFrame,
    title: str = "Average Accuracy (%)",
    save_path: str | None = None,
    fmt: str = ".2f",
    as_percentage: bool = True,
) -> None:
    """
    Plot a heatmap of metric values (atlases × classifiers).

    Parameters
    ----------
    results_df : pd.DataFrame
        Rows = atlases, columns = classifiers, values = metric.
    title : str
    save_path : str or None
    fmt : str  — annotation format string
    as_percentage : bool — if True, multiply values by 100 for display
    """
    data = results_df.copy()
    if as_percentage:
        data = data * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=CMAP_HEAT,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Atlas")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("Saved heatmap: %s", save_path)
    plt.close(fig)


# ──────────────────────────────────────────────
#  Confusion Matrix
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    labels: list[str] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot a single confusion matrix as a heatmap.

    Parameters
    ----------
    cm : np.ndarray of shape (2, 2)
    title : str
    labels : list[str]  — class labels (default: ['TD', 'ASD'])
    save_path : str or None
    """
    if labels is None:
        labels = ["TD", "ASD"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("Saved confusion matrix: %s", save_path)
    plt.close(fig)


def plot_confusion_matrices_grid(
    results: dict,
    atlas_name: str,
    save_path: str | None = None,
) -> None:
    """
    Plot confusion matrices for all classifiers for a given atlas
    in a single grid figure.

    Parameters
    ----------
    results : dict  — results[atlas][classifier] = eval_dict
    atlas_name : str
    save_path : str or None
    """
    clf_results = results[atlas_name]
    clf_names = list(clf_results.keys())
    n = len(clf_names)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    labels = ["TD", "ASD"]

    for i, clf_name in enumerate(clf_names):
        # Sum confusion matrices across folds
        cm_sum = sum(clf_results[clf_name]["all_cm"])
        sns.heatmap(
            cm_sum,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[i],
        )
        axes[i].set_title(clf_name, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Confusion Matrices — {atlas_name}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("Saved CM grid: %s", save_path)
    plt.close(fig)


# ──────────────────────────────────────────────
#  Modality Comparison
# ──────────────────────────────────────────────

def plot_modality_comparison(
    structural_df: pd.DataFrame,
    functional_df: pd.DataFrame,
    metric_name: str = "Accuracy",
    save_path: str | None = None,
) -> None:
    """
    Side-by-side grouped bar chart comparing structural vs functional
    performance, averaged across atlases for each classifier.

    Parameters
    ----------
    structural_df, functional_df : pd.DataFrame
        Rows = atlases, columns = classifiers, values = metric.
    metric_name : str
    save_path : str or None
    """
    struct_mean = structural_df.mean(axis=0) * 100
    func_mean = functional_df.mean(axis=0) * 100

    clf_names = struct_mean.index.tolist()
    x = np.arange(len(clf_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, struct_mean.values, width, label="Structural", color="#2196F3")
    bars2 = ax.bar(x + width / 2, func_mean.values, width, label="Functional", color="#FF9800")

    ax.set_xlabel("Classifier")
    ax.set_ylabel(f"Average {metric_name} (%)")
    ax.set_title(
        f"Structural vs Functional — Average {metric_name} by Classifier",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(clf_names, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 100)

    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("Saved modality comparison: %s", save_path)
    plt.close(fig)


# ──────────────────────────────────────────────
#  F1 Bar Chart
# ──────────────────────────────────────────────

def plot_f1_bar_chart(
    results_df: pd.DataFrame,
    title: str = "F1-Score by Atlas and Classifier",
    save_path: str | None = None,
) -> None:
    """
    Grouped bar chart of F1 scores for each atlas–classifier pair.

    Parameters
    ----------
    results_df : pd.DataFrame
        Rows = atlases, columns = classifiers, values = F1 score.
    title : str
    save_path : str or None
    """
    data = results_df * 100  # percent

    fig, ax = plt.subplots(figsize=(14, 7))
    data.plot(kind="bar", ax=ax, width=0.8, edgecolor="white")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Atlas")
    ax.set_ylabel("F1-Score (%)")
    ax.set_xticklabels(data.index, rotation=30, ha="right")
    ax.legend(title="Classifier", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        logger.info("Saved F1 bar chart: %s", save_path)
    plt.close(fig)


# ──────────────────────────────────────────────
#  Generate All Plots
# ──────────────────────────────────────────────

def generate_all_plots(
    structural_results: dict | None,
    functional_results: dict | None,
    output_dir: str,
) -> None:
    """
    Generate the complete set of visualisation outputs.

    Parameters
    ----------
    structural_results, functional_results : dict or None
        Output from evaluation.metrics.run_full_evaluation.
    output_dir : str
        Base directory for saving plots.
    """
    from evaluation.metrics import results_to_dataframe

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    struct_acc_df, struct_f1_df = None, None
    func_acc_df, func_f1_df = None, None

    if structural_results is not None:
        struct_acc_df = results_to_dataframe(structural_results, "accuracy")
        struct_f1_df = results_to_dataframe(structural_results, "f1")

        plot_accuracy_heatmap(
            struct_acc_df,
            title="Structural Data — Average Accuracy (%)",
            save_path=os.path.join(plots_dir, "structural_accuracy_heatmap.png"),
        )
        plot_accuracy_heatmap(
            struct_f1_df,
            title="Structural Data — Average F1-Score (%)",
            save_path=os.path.join(plots_dir, "structural_f1_heatmap.png"),
        )
        plot_f1_bar_chart(
            struct_f1_df,
            title="Structural Data — F1-Score by Atlas and Classifier",
            save_path=os.path.join(plots_dir, "structural_f1_bar.png"),
        )

        # Confusion matrix grids for each atlas
        for atlas_name in structural_results:
            plot_confusion_matrices_grid(
                structural_results, atlas_name,
                save_path=os.path.join(
                    plots_dir, f"structural_cm_{atlas_name}.png"
                ),
            )

    if functional_results is not None:
        func_acc_df = results_to_dataframe(functional_results, "accuracy")
        func_f1_df = results_to_dataframe(functional_results, "f1")

        plot_accuracy_heatmap(
            func_acc_df,
            title="Functional Data — Average Accuracy (%)",
            save_path=os.path.join(plots_dir, "functional_accuracy_heatmap.png"),
        )
        plot_accuracy_heatmap(
            func_f1_df,
            title="Functional Data — Average F1-Score (%)",
            save_path=os.path.join(plots_dir, "functional_f1_heatmap.png"),
        )
        plot_f1_bar_chart(
            func_f1_df,
            title="Functional Data — F1-Score by Atlas and Classifier",
            save_path=os.path.join(plots_dir, "functional_f1_bar.png"),
        )

        for atlas_name in functional_results:
            plot_confusion_matrices_grid(
                functional_results, atlas_name,
                save_path=os.path.join(
                    plots_dir, f"functional_cm_{atlas_name}.png"
                ),
            )

    # Modality comparison (if both modalities present)
    if struct_acc_df is not None and func_acc_df is not None:
        plot_modality_comparison(
            struct_acc_df, func_acc_df,
            metric_name="Accuracy",
            save_path=os.path.join(plots_dir, "modality_comparison_accuracy.png"),
        )

    if struct_f1_df is not None and func_f1_df is not None:
        plot_modality_comparison(
            struct_f1_df, func_f1_df,
            metric_name="F1-Score",
            save_path=os.path.join(plots_dir, "modality_comparison_f1.png"),
        )

    logger.info("All plots saved to %s", plots_dir)
