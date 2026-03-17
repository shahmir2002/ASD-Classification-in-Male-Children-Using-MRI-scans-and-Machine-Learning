"""
ABIDE data loading utilities.

Handles loading of:
  - Structural data:  per-subject cortical thickness CSVs (one per atlas)
  - Functional data:  per-subject ROI BOLD time-series (.1D files)
  - Phenotypic data:  subject metadata (diagnosis labels, age, sex, etc.)
  - Atlas parcellations: NIfTI files for region-of-interest definitions

Expected data layout under `data_dir`:
    data/raw/
    ├── structural/
    │   ├── AAL/
    │   │   ├── sub-001_thickness.csv   (or .txt)
    │   │   └── ...
    │   ├── CC200/
    │   └── ...
    ├── functional/
    │   ├── AAL/
    │   │   ├── sub-001_rois.1D
    │   │   └── ...
    │   └── ...
    ├── atlases/
    │   ├── aal.nii.gz
    │   ├── cc200.nii.gz
    │   └── ...
    └── phenotypic.csv
"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import nibabel as nib

from config.settings import ATLASES, FUNCTIONAL_FEATURE_NAMES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Phenotypic / Label Loading
# ──────────────────────────────────────────────

def load_phenotypic(data_dir: str) -> pd.DataFrame:
    """
    Load the phenotypic CSV containing subject metadata and diagnosis labels.

    Expected columns (at minimum):
        - SUB_ID   : unique subject identifier
        - DX_GROUP : diagnosis (1 = ASD, 2 = TD)

    Returns
    -------
    pd.DataFrame with a standardised 'label' column (1 = ASD, 0 = TD).
    """
    pheno_path = os.path.join(data_dir, "phenotypic.csv")
    if not os.path.exists(pheno_path):
        raise FileNotFoundError(
            f"Phenotypic file not found at {pheno_path}. "
            "Please place the ABIDE phenotypic CSV there."
        )

    df = pd.read_csv(pheno_path)
    logger.info("Loaded phenotypic data: %d subjects", len(df))

    # Standardise label: ASD = 1, TD = 0
    if "DX_GROUP" in df.columns:
        df["label"] = (df["DX_GROUP"] == 1).astype(int)
    elif "label" in df.columns:
        pass  # already present
    else:
        raise KeyError(
            "Phenotypic file must contain 'DX_GROUP' or 'label' column."
        )

    return df


# ──────────────────────────────────────────────
#  Atlas Loading
# ──────────────────────────────────────────────

def load_atlas(atlas_name: str, atlas_dir: str) -> nib.Nifti1Image:
    """
    Load a brain atlas parcellation NIfTI file.

    Parameters
    ----------
    atlas_name : str
        Key from ATLASES dict (e.g. 'AAL', 'CC200').
    atlas_dir : str
        Directory containing the atlas NIfTI files.

    Returns
    -------
    nibabel.Nifti1Image
    """
    if atlas_name not in ATLASES:
        raise ValueError(
            f"Unknown atlas '{atlas_name}'. Choose from: {list(ATLASES.keys())}"
        )

    filename = ATLASES[atlas_name]["filename"]
    path = os.path.join(atlas_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Atlas file not found: {path}")

    img = nib.load(path)
    logger.info(
        "Loaded atlas '%s' from %s  (shape: %s)", atlas_name, path, img.shape
    )
    return img


# ──────────────────────────────────────────────
#  Structural Data Loading
# ──────────────────────────────────────────────

def load_structural(
    atlas_name: str,
    data_dir: str,
    phenotypic_df: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load structural (cortical thickness) features for a given atlas.

    Expects one file per subject inside:
        data_dir/structural/<atlas_name>/

    Each file is a CSV or whitespace-delimited text with one row of
    regional cortical thickness values (columns = ROIs).

    Parameters
    ----------
    atlas_name : str
    data_dir : str
    phenotypic_df : pd.DataFrame, optional
        If provided, used to assign labels.  Otherwise attempts to
        infer from filenames or a co-located labels file.

    Returns
    -------
    X : np.ndarray of shape (n_subjects, n_rois)
    y : np.ndarray of shape (n_subjects,)  — 1 = ASD, 0 = TD
    subject_ids : list[str]
    """
    struct_dir = os.path.join(data_dir, "structural", atlas_name)
    if not os.path.isdir(struct_dir):
        raise FileNotFoundError(
            f"Structural data directory not found: {struct_dir}"
        )

    files = sorted(
        glob.glob(os.path.join(struct_dir, "*.csv"))
        + glob.glob(os.path.join(struct_dir, "*.txt"))
    )

    if not files:
        raise FileNotFoundError(
            f"No .csv or .txt files found in {struct_dir}"
        )

    features, subject_ids = [], []
    for fpath in files:
        # Extract subject ID from filename (e.g. "sub-001_thickness.csv" → "sub-001")
        basename = os.path.splitext(os.path.basename(fpath))[0]
        sub_id = basename.split("_")[0]
        subject_ids.append(sub_id)

        # Load thickness values
        try:
            vals = np.loadtxt(fpath, delimiter=",")
        except ValueError:
            vals = np.loadtxt(fpath)  # whitespace delimited

        if vals.ndim > 1:
            vals = vals.flatten()
        features.append(vals)

    X = np.array(features, dtype=np.float64)
    logger.info(
        "Loaded structural data for atlas '%s': X.shape = %s",
        atlas_name, X.shape,
    )

    # Remove subjects/features with NaN
    nan_mask = np.isnan(X).any(axis=1)
    if nan_mask.any():
        n_removed = nan_mask.sum()
        logger.warning("Removing %d subjects with NaN values", n_removed)
        X = X[~nan_mask]
        subject_ids = [s for s, m in zip(subject_ids, nan_mask) if not m]

    # Assign labels
    y = _assign_labels(subject_ids, phenotypic_df, data_dir)

    return X, y, subject_ids


# ──────────────────────────────────────────────
#  Functional Data Loading
# ──────────────────────────────────────────────

def load_functional_timeseries(
    atlas_name: str,
    data_dir: str,
    phenotypic_df: pd.DataFrame | None = None,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """
    Load raw ROI BOLD time-series for a given atlas.

    Expects one .1D (or .csv/.txt) file per subject inside:
        data_dir/functional/<atlas_name>/

    Each file has shape (n_timepoints, n_rois).

    Returns
    -------
    timeseries : list[np.ndarray]
        Each element has shape (n_timepoints, n_rois) for one subject.
    y : np.ndarray of shape (n_subjects,) — 1 = ASD, 0 = TD
    subject_ids : list[str]
    """
    func_dir = os.path.join(data_dir, "functional", atlas_name)
    if not os.path.isdir(func_dir):
        raise FileNotFoundError(
            f"Functional data directory not found: {func_dir}"
        )

    files = sorted(
        glob.glob(os.path.join(func_dir, "*.1D"))
        + glob.glob(os.path.join(func_dir, "*.csv"))
        + glob.glob(os.path.join(func_dir, "*.txt"))
    )

    if not files:
        raise FileNotFoundError(
            f"No .1D, .csv, or .txt files found in {func_dir}"
        )

    timeseries, subject_ids = [], []
    for fpath in files:
        basename = os.path.splitext(os.path.basename(fpath))[0]
        sub_id = basename.split("_")[0]
        subject_ids.append(sub_id)

        try:
            ts = np.loadtxt(fpath, delimiter=",")
        except ValueError:
            ts = np.loadtxt(fpath)

        timeseries.append(ts)

    logger.info(
        "Loaded functional time-series for atlas '%s': %d subjects",
        atlas_name, len(timeseries),
    )

    y = _assign_labels(subject_ids, phenotypic_df, data_dir)

    return timeseries, y, subject_ids


# ──────────────────────────────────────────────
#  Label Assignment Helper
# ──────────────────────────────────────────────

def _assign_labels(
    subject_ids: list[str],
    phenotypic_df: pd.DataFrame | None,
    data_dir: str,
) -> np.ndarray:
    """
    Assign binary labels (ASD=1, TD=0) to a list of subject IDs.

    Tries in order:
        1. Look up from the provided phenotypic DataFrame.
        2. Fall back to a labels.csv file in data_dir.
    """
    if phenotypic_df is not None:
        id_col = _find_id_column(phenotypic_df)
        pheno_ids = phenotypic_df[id_col].astype(str).values
        label_map = dict(zip(pheno_ids, phenotypic_df["label"].values))

        labels = []
        for sid in subject_ids:
            # Try exact match, then numeric-only match
            if sid in label_map:
                labels.append(label_map[sid])
            else:
                numeric_id = "".join(filter(str.isdigit, sid))
                matched = [
                    label_map[k] for k in label_map
                    if "".join(filter(str.isdigit, k)) == numeric_id
                ]
                if matched:
                    labels.append(matched[0])
                else:
                    raise KeyError(
                        f"Subject '{sid}' not found in phenotypic data."
                    )
        return np.array(labels, dtype=int)

    # Fallback: labels.csv
    labels_path = os.path.join(data_dir, "labels.csv")
    if os.path.exists(labels_path):
        ldf = pd.read_csv(labels_path)
        id_col = _find_id_column(ldf)
        label_map = dict(zip(ldf[id_col].astype(str), ldf["label"]))
        return np.array([label_map[sid] for sid in subject_ids], dtype=int)

    raise FileNotFoundError(
        "Cannot assign labels: provide phenotypic_df or place labels.csv "
        f"in {data_dir}."
    )


def _find_id_column(df: pd.DataFrame) -> str:
    """Find the subject-ID column in a DataFrame."""
    for col in ["SUB_ID", "sub_id", "Subject", "subject", "ID", "id"]:
        if col in df.columns:
            return col
    raise KeyError(
        f"Cannot find subject-ID column. Columns present: {list(df.columns)}"
    )
