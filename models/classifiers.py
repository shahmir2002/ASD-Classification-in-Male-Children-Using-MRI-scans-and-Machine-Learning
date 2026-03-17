"""
Classifier factory for the STRP 2 pipeline.

Provides convenience functions to instantiate fresh copies of all seven
classifiers with the exact hyperparameters used in the research report.
"""

import copy
from config.settings import get_classifier_configs


def get_classifiers() -> dict:
    """
    Return a dict of classifier_name → fresh sklearn estimator.

    Each call returns new, unfitted instances so they can be safely
    used inside cross-validation loops.
    """
    return get_classifier_configs()


def get_classifier(name: str):
    """
    Return a single fresh classifier instance by name.

    Parameters
    ----------
    name : str
        One of: 'MLP', 'SVM', 'Random Forest', 'Logistic Regression',
                'QDA', 'GBC', 'AdaBoost'

    Returns
    -------
    sklearn estimator (unfitted)
    """
    configs = get_classifier_configs()
    if name not in configs:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Available: {list(configs.keys())}"
        )
    return copy.deepcopy(configs[name])


def list_classifier_names() -> list[str]:
    """Return the names of all available classifiers."""
    return list(get_classifier_configs().keys())
