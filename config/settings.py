"""
Central configuration for the STRP 2 ASD classification pipeline.

Contains atlas definitions, classifier hyperparameters, feature names,
and all project-wide constants.
"""

import os
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# ──────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
ATLAS_DIR = os.path.join(DATA_DIR, "atlases")

# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────
RANDOM_STATE = 42
CV_FOLDS = 10
MAX_ITER = 5000

# ──────────────────────────────────────────────
#  Brain Atlases
# ──────────────────────────────────────────────
# Each atlas maps to its expected number of ROIs and the
# filename pattern used when loading the NIfTI parcellation.
ATLASES = {
    "AAL":        {"n_rois": 116,  "filename": "aal.nii.gz"},
    "Dosenbach":  {"n_rois": 160,  "filename": "dosenbach160.nii.gz"},
    "CC200":      {"n_rois": 200,  "filename": "cc200.nii.gz"},
    "CC400":      {"n_rois": 392,  "filename": "cc400.nii.gz"},
    "EZ":         {"n_rois": 116,  "filename": "ez.nii.gz"},
    "HO":         {"n_rois": 110,  "filename": "ho.nii.gz"},
    "TT":         {"n_rois": 97,   "filename": "tt.nii.gz"},
}

# ──────────────────────────────────────────────
#  Functional Feature Definitions
# ──────────────────────────────────────────────
# 21 features extracted per ROI from the BOLD time series.
FUNCTIONAL_FEATURE_NAMES = [
    # Statistical (5)
    "mean", "std", "median", "skewness", "kurtosis",
    # Temporal dynamics (5)
    "slope", "zero_crossing_rate", "autocorrelation_lag1",
    "time_to_peak", "time_above_threshold",
    # Spectral (5)
    "dominant_frequency", "avg_psd",
    "band_power_slow5", "band_power_slow4", "band_power_slow3",
    # Neuroimaging metrics (2)
    "alff", "falff",
    # Nonlinear dynamics (1)
    "lyapunov_exponent",
    # Additional statistical (3)
    "iqr", "range", "rms",
]

assert len(FUNCTIONAL_FEATURE_NAMES) == 21, (
    f"Expected 21 functional features, got {len(FUNCTIONAL_FEATURE_NAMES)}"
)

# Canonical frequency bands (Hz) for resting-state fMRI
FREQ_BANDS = {
    "slow5": (0.010, 0.027),
    "slow4": (0.027, 0.073),
    "slow3": (0.073, 0.198),
}

# Typical fMRI repetition time (seconds) — override per dataset if needed
DEFAULT_TR = 2.0

# ──────────────────────────────────────────────
#  Classifier Configurations
# ──────────────────────────────────────────────
def get_classifier_configs():
    """
    Return a dictionary of classifier name → sklearn estimator,
    configured with the exact hyperparameters from the STRP 2 report.
    """
    return {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate="adaptive",
            max_iter=MAX_ITER,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            gamma="scale",
            shrinking=True,
            tol=1e-3,
            cache_size=200,
            class_weight="balanced",
            max_iter=MAX_ITER,
            decision_function_shape="ovr",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            oob_score=False,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight=None,
        ),
        "Logistic Regression": LogisticRegression(
            penalty="l2",
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=True,
            solver="lbfgs",
            max_iter=MAX_ITER,
            random_state=RANDOM_STATE,
            n_jobs=None,
        ),
        "QDA": QuadraticDiscriminantAnalysis(
            priors=None,
            reg_param=0.0,
            store_covariance=False,
            tol=1e-4,
        ),
        "GBC": GradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.1,
            n_estimators=100,
            subsample=1.0,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            max_features=None,
            random_state=RANDOM_STATE,
            max_leaf_nodes=None,
            warm_start=False,
            validation_fraction=0.1,
            n_iter_no_change=None,
            tol=1e-4,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=RANDOM_STATE,
        ),
    }
