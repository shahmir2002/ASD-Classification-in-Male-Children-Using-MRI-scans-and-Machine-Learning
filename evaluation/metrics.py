"""
Evaluation and cross-validation utilities for the STRP 2 pipeline.

Implements stratified k-fold cross-validation with fold-aware
preprocessing (fit on train, transform on test) and comprehensive
metric collection (accuracy, precision, recall, F1, confusion matrices).
"""

import os
import json
import copy
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config.settings import CV_FOLDS, RANDOM_STATE, ATLASES
from data.preprocessing import create_preprocessing_pipeline
from models.classifiers import get_classifiers

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Single Model Evaluation
# ──────────────────────────────────────────────

def evaluate_model(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    modality: str,
    n_folds: int = CV_FOLDS,
) -> dict:
    """
    Evaluate a single classifier using stratified k-fold cross-validation.

    Preprocessing is applied inside each fold to prevent information
    leakage (fitted on training data, applied to validation data).

    Parameters
    ----------
    clf : sklearn estimator (will be deep-copied per fold)
    X : np.ndarray of shape (n_subjects, n_features)
    y : np.ndarray of shape (n_subjects,)
    modality : str  — 'structural' or 'functional'
    n_folds : int

    Returns
    -------
    dict with keys:
        'per_fold'   : list of per-fold metric dicts
        'average'    : dict of mean metrics across folds
        'best_fold'  : dict of best-fold metrics (by accuracy)
        'all_cm'     : list of confusion matrices (np.ndarray)
    """
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE
    )

    fold_results = []
    all_cm = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit preprocessing on train, transform both
        pipeline = create_preprocessing_pipeline(modality)
        X_train = pipeline.fit_transform(X_train)
        X_val = pipeline.transform(X_val)

        # Train classifier
        model = copy.deepcopy(clf)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        except Exception as e:
            logger.warning("Fold %d failed: %s", fold_idx, e)
            # Record zero metrics for failed folds
            fold_results.append({
                "fold": fold_idx,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            })
            all_cm.append(np.zeros((2, 2), dtype=int))
            continue

        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        cm = confusion_matrix(y_val, y_pred, labels=[0, 1])

        fold_results.append({
            "fold": fold_idx,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })
        all_cm.append(cm)

        logger.debug(
            "  Fold %d — Acc: %.4f  Prec: %.4f  Rec: %.4f  F1: %.4f",
            fold_idx, acc, prec, rec, f1,
        )

    # Aggregate
    metrics = ["accuracy", "precision", "recall", "f1"]
    average = {
        m: np.mean([r[m] for r in fold_results]) for m in metrics
    }
    best_idx = int(np.argmax([r["accuracy"] for r in fold_results]))
    best_fold = fold_results[best_idx]

    return {
        "per_fold": fold_results,
        "average": average,
        "best_fold": best_fold,
        "all_cm": all_cm,
    }


# ──────────────────────────────────────────────
#  Full Evaluation Grid
# ──────────────────────────────────────────────

def run_full_evaluation(
    X_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    modality: str,
    n_folds: int = CV_FOLDS,
) -> dict:
    """
    Run all 7 classifiers across all atlases for a given modality.

    Parameters
    ----------
    X_dict : dict
        atlas_name → (X, y) tuples.
    modality : str  — 'structural' or 'functional'
    n_folds : int

    Returns
    -------
    results : nested dict  →  results[atlas][classifier] = eval_dict
    """
    classifiers = get_classifiers()
    results = {}

    total = len(X_dict) * len(classifiers)
    count = 0

    for atlas_name, (X, y) in X_dict.items():
        results[atlas_name] = {}

        for clf_name, clf in classifiers.items():
            count += 1
            logger.info(
                "[%d/%d] Evaluating %s on %s (%s) ...",
                count, total, clf_name, atlas_name, modality,
            )

            eval_result = evaluate_model(clf, X, y, modality, n_folds)
            results[atlas_name][clf_name] = eval_result

            avg = eval_result["average"]
            logger.info(
                "  → Avg Acc: %.4f  F1: %.4f", avg["accuracy"], avg["f1"]
            )

    return results


# ──────────────────────────────────────────────
#  Results to DataFrame
# ──────────────────────────────────────────────

def results_to_dataframe(
    results: dict,
    metric: str = "accuracy",
) -> pd.DataFrame:
    """
    Convert nested results dict to a DataFrame (rows = atlases, cols = classifiers).

    Parameters
    ----------
    results : dict from run_full_evaluation
    metric : str  — which average metric to extract

    Returns
    -------
    pd.DataFrame
    """
    rows = {}
    for atlas_name, clf_results in results.items():
        rows[atlas_name] = {
            clf_name: eval_dict["average"][metric]
            for clf_name, eval_dict in clf_results.items()
        }

    df = pd.DataFrame(rows).T
    df.index.name = "Atlas"
    return df


# ──────────────────────────────────────────────
#  Save Results
# ──────────────────────────────────────────────

def save_results(results: dict, output_dir: str, modality: str) -> None:
    """
    Save evaluation results to CSV and JSON files.

    Creates:
        - {modality}_accuracy.csv   — average accuracy grid
        - {modality}_f1.csv         — average F1 grid
        - {modality}_full.json      — complete per-fold results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Accuracy table
    acc_df = results_to_dataframe(results, "accuracy")
    acc_path = os.path.join(output_dir, f"{modality}_accuracy.csv")
    acc_df.to_csv(acc_path)
    logger.info("Saved accuracy table: %s", acc_path)

    # F1 table
    f1_df = results_to_dataframe(results, "f1")
    f1_path = os.path.join(output_dir, f"{modality}_f1.csv")
    f1_df.to_csv(f1_path)
    logger.info("Saved F1 table: %s", f1_path)

    # Full results (JSON-serialisable version)
    serialisable = {}
    for atlas, clf_results in results.items():
        serialisable[atlas] = {}
        for clf, eval_dict in clf_results.items():
            serialisable[atlas][clf] = {
                "per_fold": eval_dict["per_fold"],
                "average": eval_dict["average"],
                "best_fold": eval_dict["best_fold"],
                "confusion_matrices": [
                    cm.tolist() for cm in eval_dict["all_cm"]
                ],
            }

    json_path = os.path.join(output_dir, f"{modality}_full_results.json")
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved full results: %s", json_path)
