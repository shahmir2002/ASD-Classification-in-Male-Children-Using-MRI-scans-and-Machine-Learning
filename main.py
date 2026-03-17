#!/usr/bin/env python3
"""
STRP 2 — Diagnosing ASD via Brain Scans Using Machine Learning

Main entry point for running the full classification pipeline.
Supports structural, functional, or both modalities across
7 brain atlases and 7 ML classifiers with stratified k-fold CV.

Usage examples:
    # Run everything (both modalities, all atlases, all classifiers)
    python main.py --data-dir ./data/raw --modality both

    # Run only structural with a specific atlas
    python main.py --data-dir ./data/raw --modality structural --atlas AAL

    # Run functional with a specific classifier
    python main.py --data-dir ./data/raw --modality functional --classifier SVM
"""

import os
import sys
import argparse
import logging
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import ATLASES, RESULTS_DIR, DEFAULT_TR
from data.loader import (
    load_phenotypic,
    load_structural,
    load_functional_timeseries,
)
from data.preprocessing import remove_nan_features
from features.extraction import extract_all_subjects
from models.classifiers import get_classifiers, get_classifier, list_classifier_names
from evaluation.metrics import (
    evaluate_model,
    run_full_evaluation,
    results_to_dataframe,
    save_results,
)
from visualization.plots import generate_all_plots


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "STRP 2: ASD classification using structural and functional "
            "brain imaging data with multiple ML classifiers and brain atlases."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the raw data directory (containing structural/, "
             "functional/, atlases/, and phenotypic.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Directory for results output (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["structural", "functional", "both"],
        default="both",
        help="Data modality to evaluate (default: both)",
    )
    parser.add_argument(
        "--atlas",
        type=str,
        default="all",
        help="Atlas name to use (default: all). "
             f"Options: {', '.join(ATLASES.keys())}",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="all",
        help="Classifier to use (default: all). "
             f"Options: {', '.join(list_classifier_names())}",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10)",
    )
    parser.add_argument(
        "--tr",
        type=float,
        default=DEFAULT_TR,
        help=f"fMRI repetition time in seconds (default: {DEFAULT_TR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )

    return parser.parse_args()


# ──────────────────────────────────────────────
#  Data Loading Helpers
# ──────────────────────────────────────────────

def load_structural_data(
    data_dir: str,
    atlas_names: list[str],
    pheno_df,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load structural features for specified atlases."""
    logger = logging.getLogger(__name__)
    X_dict = {}

    for atlas in atlas_names:
        try:
            X, y, _ = load_structural(atlas, data_dir, pheno_df)
            X, _ = remove_nan_features(X)
            X_dict[atlas] = (X, y)
            logger.info(
                "  %s: %d subjects, %d features",
                atlas, X.shape[0], X.shape[1],
            )
        except FileNotFoundError as e:
            logger.warning("  Skipping atlas %s: %s", atlas, e)

    return X_dict


def load_functional_data(
    data_dir: str,
    atlas_names: list[str],
    pheno_df,
    tr: float,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load functional features for specified atlases."""
    logger = logging.getLogger(__name__)
    X_dict = {}

    for atlas in atlas_names:
        try:
            timeseries, y, _ = load_functional_timeseries(
                atlas, data_dir, pheno_df
            )
            X = extract_all_subjects(timeseries, tr=tr)
            X, _ = remove_nan_features(X)
            X_dict[atlas] = (X, y)
            logger.info(
                "  %s: %d subjects, %d features",
                atlas, X.shape[0], X.shape[1],
            )
        except FileNotFoundError as e:
            logger.warning("  Skipping atlas %s: %s", atlas, e)

    return X_dict


# ──────────────────────────────────────────────
#  Main Pipeline
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("STRP 2 — ASD Classification Pipeline")
    logger.info("=" * 60)
    logger.info("Data dir:    %s", args.data_dir)
    logger.info("Output dir:  %s", args.output_dir)
    logger.info("Modality:    %s", args.modality)
    logger.info("Atlas:       %s", args.atlas)
    logger.info("Classifier:  %s", args.classifier)
    logger.info("CV folds:    %d", args.folds)
    logger.info("TR:          %.2f s", args.tr)
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine which atlases to run
    if args.atlas.lower() == "all":
        atlas_names = list(ATLASES.keys())
    else:
        if args.atlas not in ATLASES:
            logger.error(
                "Unknown atlas '%s'. Available: %s",
                args.atlas, list(ATLASES.keys()),
            )
            sys.exit(1)
        atlas_names = [args.atlas]

    # Load phenotypic data
    logger.info("Loading phenotypic data ...")
    pheno_df = load_phenotypic(args.data_dir)
    n_asd = (pheno_df["label"] == 1).sum()
    n_td = (pheno_df["label"] == 0).sum()
    logger.info("  Subjects: %d ASD, %d TD", n_asd, n_td)

    structural_results = None
    functional_results = None

    start_time = time.time()

    # ── Structural ──
    if args.modality in ("structural", "both"):
        logger.info("\n" + "─" * 40)
        logger.info("STRUCTURAL MODALITY")
        logger.info("─" * 40)

        X_struct = load_structural_data(args.data_dir, atlas_names, pheno_df)

        if X_struct:
            structural_results = run_full_evaluation(
                X_struct, "structural", args.folds
            )
            save_results(structural_results, args.output_dir, "structural")

            # Print summary
            acc_df = results_to_dataframe(structural_results, "accuracy")
            logger.info("\nStructural Accuracy (%%)\n%s", (acc_df * 100).to_string())
        else:
            logger.warning("No structural data loaded — skipping.")

    # ── Functional ──
    if args.modality in ("functional", "both"):
        logger.info("\n" + "─" * 40)
        logger.info("FUNCTIONAL MODALITY")
        logger.info("─" * 40)

        X_func = load_functional_data(
            args.data_dir, atlas_names, pheno_df, args.tr
        )

        if X_func:
            functional_results = run_full_evaluation(
                X_func, "functional", args.folds
            )
            save_results(functional_results, args.output_dir, "functional")

            acc_df = results_to_dataframe(functional_results, "accuracy")
            logger.info("\nFunctional Accuracy (%%)\n%s", (acc_df * 100).to_string())
        else:
            logger.warning("No functional data loaded — skipping.")

    # ── Plots ──
    if not args.no_plots and (structural_results or functional_results):
        logger.info("\nGenerating plots ...")
        generate_all_plots(
            structural_results, functional_results, args.output_dir
        )

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline complete in %.1f seconds", elapsed)
    logger.info("Results saved to: %s", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
