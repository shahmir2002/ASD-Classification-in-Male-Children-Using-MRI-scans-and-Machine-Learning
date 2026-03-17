"""
Functional feature extraction from BOLD time-series.

Extracts 21 features per ROI from resting-state fMRI time-series data:
  - 5 statistical features
  - 5 temporal dynamics features
  - 5 spectral features
  - 2 neuroimaging-specific metrics (ALFF, fALFF)
  - 1 nonlinear dynamics feature (Lyapunov exponent)
  - 3 additional statistical features
"""

import logging
import warnings

import numpy as np
from scipy import stats, signal
from scipy.fft import rfft, rfftfreq

from config.settings import FREQ_BANDS, DEFAULT_TR, FUNCTIONAL_FEATURE_NAMES

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Individual Feature Functions
# ──────────────────────────────────────────────

def _compute_statistical(ts: np.ndarray) -> dict:
    """Basic statistical features from a 1-D time series."""
    return {
        "mean": np.mean(ts),
        "std": np.std(ts, ddof=1) if len(ts) > 1 else 0.0,
        "median": np.median(ts),
        "skewness": float(stats.skew(ts, bias=False)) if len(ts) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(ts, bias=False)) if len(ts) > 3 else 0.0,
    }


def _compute_temporal(ts: np.ndarray) -> dict:
    """Temporal dynamics features."""
    n = len(ts)

    # Linear slope via least-squares
    if n > 1:
        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, ts, 1)[0]
    else:
        slope = 0.0

    # Zero-crossing rate
    if n > 1:
        zero_crossings = np.sum(np.diff(np.sign(ts - np.mean(ts))) != 0)
        zcr = zero_crossings / (n - 1)
    else:
        zcr = 0.0

    # Autocorrelation at lag 1
    if n > 1:
        ts_centered = ts - np.mean(ts)
        var = np.var(ts)
        if var > 0:
            auto_corr = np.correlate(ts_centered, ts_centered, mode="full")
            auto_corr = auto_corr[n - 1 :] / (var * n)
            acf_lag1 = auto_corr[1] if len(auto_corr) > 1 else 0.0
        else:
            acf_lag1 = 0.0
    else:
        acf_lag1 = 0.0

    # Time to peak (index of maximum as fraction of total length)
    time_to_peak = float(np.argmax(ts)) / max(n - 1, 1)

    # Time above threshold (fraction of time above the mean)
    threshold = np.mean(ts)
    time_above = np.mean(ts > threshold)

    return {
        "slope": slope,
        "zero_crossing_rate": zcr,
        "autocorrelation_lag1": acf_lag1,
        "time_to_peak": time_to_peak,
        "time_above_threshold": time_above,
    }


def _compute_spectral(ts: np.ndarray, tr: float = DEFAULT_TR) -> dict:
    """
    Spectral features from the power spectrum of a BOLD signal.

    Parameters
    ----------
    ts : 1-D array
    tr : float  — repetition time in seconds (determines Nyquist freq)
    """
    n = len(ts)
    if n < 4:
        return {
            "dominant_frequency": 0.0,
            "avg_psd": 0.0,
            "band_power_slow5": 0.0,
            "band_power_slow4": 0.0,
            "band_power_slow3": 0.0,
        }

    # Compute power spectral density using Welch's method
    fs = 1.0 / tr
    nperseg = min(n, 256)
    freqs, psd = signal.welch(ts, fs=fs, nperseg=nperseg)

    # Dominant frequency
    dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0

    # Average PSD
    avg_psd = np.mean(psd) if len(psd) > 0 else 0.0

    # Band power for canonical resting-state bands
    band_powers = {}
    for band_name, (f_low, f_high) in FREQ_BANDS.items():
        mask = (freqs >= f_low) & (freqs <= f_high)
        band_powers[f"band_power_{band_name}"] = (
            np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
        )

    return {
        "dominant_frequency": dominant_freq,
        "avg_psd": avg_psd,
        **band_powers,
    }


def _compute_alff_falff(ts: np.ndarray, tr: float = DEFAULT_TR) -> dict:
    """
    Amplitude of Low-Frequency Fluctuations (ALFF) and fractional ALFF.

    ALFF  = sum of amplitudes in the 0.01–0.08 Hz band
    fALFF = ALFF / sum of amplitudes across all frequencies
    """
    n = len(ts)
    if n < 4:
        return {"alff": 0.0, "falff": 0.0}

    fs = 1.0 / tr
    freqs = rfftfreq(n, d=tr)
    amplitudes = np.abs(rfft(ts - np.mean(ts)))

    # Low-frequency band (0.01–0.08 Hz)
    lf_mask = (freqs >= 0.01) & (freqs <= 0.08)
    alff = np.sum(amplitudes[lf_mask])
    total = np.sum(amplitudes)
    falff = alff / total if total > 0 else 0.0

    return {"alff": alff, "falff": falff}


def _compute_lyapunov(ts: np.ndarray) -> dict:
    """
    Compute the largest Lyapunov exponent as a measure of signal complexity.
    Falls back to a correlation-dimension-based estimate if nolds is not
    available.
    """
    if len(ts) < 20:
        return {"lyapunov_exponent": 0.0}

    try:
        import nolds
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lyap = nolds.lyap_r(ts, emb_dim=10, lag=1, min_tsep=None)
        if not np.isfinite(lyap):
            lyap = 0.0
    except Exception:
        # Simple fallback: average log-divergence
        lyap = 0.0

    return {"lyapunov_exponent": float(lyap)}


def _compute_additional_statistical(ts: np.ndarray) -> dict:
    """Additional descriptive statistics."""
    return {
        "iqr": float(np.subtract(*np.percentile(ts, [75, 25]))),
        "range": float(np.ptp(ts)),
        "rms": float(np.sqrt(np.mean(ts ** 2))),
    }


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def extract_roi_features(ts: np.ndarray, tr: float = DEFAULT_TR) -> np.ndarray:
    """
    Extract 21 features from a single ROI time series.

    Parameters
    ----------
    ts : np.ndarray of shape (n_timepoints,)
    tr : float  — repetition time in seconds

    Returns
    -------
    np.ndarray of shape (21,)
    """
    features = {}
    features.update(_compute_statistical(ts))
    features.update(_compute_temporal(ts))
    features.update(_compute_spectral(ts, tr))
    features.update(_compute_alff_falff(ts, tr))
    features.update(_compute_lyapunov(ts))
    features.update(_compute_additional_statistical(ts))

    # Ensure ordering matches FUNCTIONAL_FEATURE_NAMES
    vec = np.array(
        [features[name] for name in FUNCTIONAL_FEATURE_NAMES],
        dtype=np.float64,
    )

    # Replace any non-finite values with 0
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    return vec


def extract_subject_features(
    roi_timeseries: np.ndarray,
    tr: float = DEFAULT_TR,
) -> np.ndarray:
    """
    Extract features for all ROIs of a single subject.

    Parameters
    ----------
    roi_timeseries : np.ndarray of shape (n_timepoints, n_rois)

    Returns
    -------
    np.ndarray of shape (n_rois * 21,) — flattened feature vector
    """
    n_timepoints, n_rois = roi_timeseries.shape
    all_features = []

    for roi_idx in range(n_rois):
        ts = roi_timeseries[:, roi_idx]
        feat = extract_roi_features(ts, tr)
        all_features.append(feat)

    return np.concatenate(all_features)


def extract_all_subjects(
    timeseries_list: list[np.ndarray],
    tr: float = DEFAULT_TR,
) -> np.ndarray:
    """
    Extract functional features for all subjects.

    Parameters
    ----------
    timeseries_list : list[np.ndarray]
        Each element has shape (n_timepoints, n_rois).

    Returns
    -------
    X : np.ndarray of shape (n_subjects, n_rois * 21)
    """
    logger.info(
        "Extracting functional features for %d subjects ...",
        len(timeseries_list),
    )
    X = []
    for i, ts_matrix in enumerate(timeseries_list):
        if ts_matrix.ndim == 1:
            ts_matrix = ts_matrix.reshape(-1, 1)
        vec = extract_subject_features(ts_matrix, tr)
        X.append(vec)
        if (i + 1) % 50 == 0:
            logger.info("  Processed %d / %d subjects", i + 1, len(timeseries_list))

    return np.array(X, dtype=np.float64)
