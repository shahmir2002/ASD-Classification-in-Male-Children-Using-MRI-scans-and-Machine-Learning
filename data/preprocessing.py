"""
Preprocessing utilities for the STRP 2 pipeline.

Provides scikit-learn-compatible pipelines for feature scaling and
dimensionality reduction, designed to be used inside cross-validation
folds (fit on train, transform on test) to prevent information leakage.
"""

import logging

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(
    modality: str,
    pca_variance: float = 0.95,
) -> Pipeline:
    """
    Build a preprocessing pipeline appropriate for the given data modality.

    Parameters
    ----------
    modality : {'structural', 'functional'}
        - structural  →  MinMaxScaler to [0, 1]
        - functional  →  StandardScaler + PCA (retaining `pca_variance`
                          fraction of variance)
    pca_variance : float
        Fraction of variance to retain in PCA (functional only).

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    modality = modality.lower()

    if modality == "structural":
        return Pipeline([
            ("scaler", MinMaxScaler(feature_range=(0, 1))),
        ])

    elif modality == "functional":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=42)),
        ])

    else:
        raise ValueError(
            f"Unknown modality '{modality}'. Choose 'structural' or 'functional'."
        )


def remove_nan_features(X: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove features (columns) that have NaN in more than `threshold`
    fraction of subjects.  Remaining NaNs are imputed with column means.

    Parameters
    ----------
    X : np.ndarray of shape (n_subjects, n_features)
    threshold : float
        Maximum fraction of NaN allowed per feature.

    Returns
    -------
    X_clean : np.ndarray
    kept_indices : np.ndarray of kept column indices
    """
    nan_frac = np.isnan(X).mean(axis=0)
    keep = nan_frac <= threshold
    X_clean = X[:, keep].copy()

    # Impute remaining NaNs with column mean
    col_means = np.nanmean(X_clean, axis=0)
    nan_locs = np.where(np.isnan(X_clean))
    X_clean[nan_locs] = np.take(col_means, nan_locs[1])

    n_removed = (~keep).sum()
    if n_removed > 0:
        logger.info(
            "Removed %d features with >%.0f%% NaN values",
            n_removed, threshold * 100,
        )

    return X_clean, np.where(keep)[0]


def align_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: list[str],
    valid_ids: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Keep only subjects present in `valid_ids` (if given) and ensure
    consistent ordering.

    Returns subset of X, y, subject_ids that are in valid_ids.
    """
    if valid_ids is None:
        return X, y, subject_ids

    mask = np.array([sid in valid_ids for sid in subject_ids])
    return X[mask], y[mask], [s for s, m in zip(subject_ids, mask) if m]
