# Diagnosing ASD via Brain Scans Using Machine Learning

A machine learning pipeline for classifying **Autism Spectrum Disorder (ASD)** in male children using structural and functional brain imaging data from the [ABIDE](http://preprocessed-connectomes-project.org/abide/) dataset.

## Overview

This project evaluates **7 machine learning classifiers** across **7 brain atlases** and **2 data modalities** (structural cortical thickness and functional BOLD time-series) to determine the most effective combination for ASD diagnosis.

| Aspect | Details |
|---|---|
| **Modalities** | Structural (cortical thickness), Functional (BOLD time-series) |
| **Atlases** | AAL, CC200, CC400, Dosenbach, EZ, HO, TT |
| **Classifiers** | MLP, SVM, Random Forest, Logistic Regression, QDA, GBC, AdaBoost |
| **Evaluation** | Stratified 10-fold cross-validation |
| **Metrics** | Accuracy, Precision, Recall, F1-score, Confusion matrices |

## Project Structure

```
├── config/
│   └── settings.py              # Atlas definitions, classifier hyperparameters, constants
├── data/
│   ├── loader.py                # ABIDE data loading (structural + functional)
│   └── preprocessing.py         # Feature scaling (MinMax/PCA), NaN handling
├── features/
│   └── extraction.py            # 21 functional features from BOLD time-series
├── models/
│   └── classifiers.py           # Classifier factory (7 classifiers)
├── evaluation/
│   └── metrics.py               # Stratified k-fold CV, metric computation
├── visualization/
│   └── plots.py                 # Heatmaps, confusion matrices, comparison charts
├── main.py                      # CLI entry point
├── requirements.txt
└── results/                     # Auto-generated output
```

## Installation

```bash
# Clone the repository
git clone https://github.com/shahmir2002/ASD-Classification-in-Male-Children-Using-MRI-scans-and-Machine-Learning.git
cd ASD-Classification-in-Male-Children-Using-MRI-scans-and-Machine-Learning

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

Place your ABIDE data in the following structure:

```
data/raw/
├── structural/
│   ├── AAL/
│   │   ├── sub-001_thickness.csv
│   │   └── ...
│   ├── CC200/
│   └── ...  (one folder per atlas)
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
```

The **phenotypic.csv** must contain at minimum:
- `SUB_ID` — unique subject identifier
- `DX_GROUP` — diagnosis label (1 = ASD, 2 = TD)

## Usage

```bash
# Run the full pipeline (both modalities, all atlases, all classifiers)
python main.py --data-dir ./data/raw --modality both

# Structural only, specific atlas
python main.py --data-dir ./data/raw --modality structural --atlas AAL

# Functional only, specific classifier
python main.py --data-dir ./data/raw --modality functional --classifier SVM

# With custom settings
python main.py --data-dir ./data/raw --modality both --folds 10 --tr 2.0 --verbose

# See all options
python main.py --help
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | *required* | Path to raw data directory |
| `--output-dir` | `./results/` | Output directory for results |
| `--modality` | `both` | `structural`, `functional`, or `both` |
| `--atlas` | `all` | Atlas name or `all` |
| `--classifier` | `all` | Classifier name or `all` |
| `--folds` | `10` | Number of CV folds |
| `--tr` | `2.0` | fMRI repetition time (seconds) |
| `--verbose` | `false` | Enable debug logging |
| `--no-plots` | `false` | Skip plot generation |

## Output

Results are saved to the output directory:

```
results/
├── structural_accuracy.csv          # Accuracy grid (atlases × classifiers)
├── structural_f1.csv                # F1-score grid
├── structural_full_results.json     # Complete per-fold results
├── functional_accuracy.csv
├── functional_f1.csv
├── functional_full_results.json
└── plots/
    ├── structural_accuracy_heatmap.png
    ├── functional_accuracy_heatmap.png
    ├── modality_comparison_accuracy.png
    ├── structural_cm_AAL.png        # Confusion matrices per atlas
    └── ...
```

## Functional Features (21 per ROI)

Each ROI's BOLD time-series is transformed into 21 features:

| # | Feature | Category | Description |
|---|---|---|---|
| 1 | **Mean** | Statistical | Average signal amplitude |
| 2 | **Std** | Statistical | Standard deviation of signal |
| 3 | **Median** | Statistical | Median signal value |
| 4 | **Skewness** | Statistical | Asymmetry of distribution |
| 5 | **Kurtosis** | Statistical | Tailedness of distribution |
| 6 | **Slope** | Temporal | Linear trend via least-squares fit |
| 7 | **Zero-Crossing Rate** | Temporal | Rate of signal sign changes around the mean |
| 8 | **Autocorrelation (lag-1)** | Temporal | Self-similarity at one time step |
| 9 | **Time-to-Peak** | Temporal | Fractional index of maximum amplitude |
| 10 | **Time-Above-Threshold** | Temporal | Fraction of time signal exceeds the mean |
| 11 | **Dominant Frequency** | Spectral | Frequency with highest power (Welch's PSD) |
| 12 | **Average PSD** | Spectral | Mean power spectral density |
| 13 | **Band Power (slow-5)** | Spectral | Power in 0.010–0.027 Hz band |
| 14 | **Band Power (slow-4)** | Spectral | Power in 0.027–0.073 Hz band |
| 15 | **Band Power (slow-3)** | Spectral | Power in 0.073–0.198 Hz band |
| 16 | **ALFF** | Neuroimaging | Amplitude of Low-Frequency Fluctuations (0.01–0.08 Hz) |
| 17 | **fALFF** | Neuroimaging | Fractional ALFF (ALFF / total amplitude) |
| 18 | **Lyapunov Exponent** | Nonlinear | Largest Lyapunov exponent — signal complexity measure |
| 19 | **IQR** | Additional | Interquartile range (75th – 25th percentile) |
| 20 | **Range** | Additional | Peak-to-peak amplitude |
| 21 | **RMS** | Additional | Root mean square of the signal |

## Results

### Average Structural Accuracies (%)

| Atlas | MLP | SVM | RF | LR | QDA | GBC | AdaBoost |
|---|---|---|---|---|---|---|---|
| AAL | 53.28 | 58.37 | 59.18 | 58.35 | 52.93 | 59.58 | 60.87 |
| CC200 | 53.33 | 59.66 | 55.80 | 57.55 | 52.93 | 58.80 | 58.37 |
| CC400 | 56.29 | 58.80 | 57.90 | 60.09 | 53.79 | 55.38 | 60.40 |
| Dosenbach | 54.26 | 58.90 | 55.04 | 55.53 | 56.00 | 55.92 | 51.27 |
| EZ | 53.37 | 58.44 | 58.28 | 58.33 | 52.10 | 57.48 | 55.40 |
| HO | 54.58 | 57.54 | 56.59 | 56.68 | 52.93 | 58.79 | 61.32 |
| TT | 54.95 | 56.68 | 62.54 | 51.25 | 58.33 | 58.77 | 59.60 |

### Average Functional Accuracies (%)

| Atlas | MLP | SVM | RF | LR | QDA | GBC | AdaBoost |
|---|---|---|---|---|---|---|---|
| AAL | 48.25 | 51.35 | 54.68 | 47.16 | 49.24 | 51.49 | 50.38 |
| CC200 | 51.05 | 52.98 | 53.63 | 56.87 | 45.18 | 48.45 | 45.70 |
| CC400 | 51.05 | 52.98 | 51.49 | 56.87 | 49.42 | 53.16 | 46.14 |
| Dosenbach | 56.35 | 52.57 | 54.71 | 52.63 | 47.51 | 46.35 | 51.49 |
| EZ | 50.32 | 53.63 | 54.71 | 48.16 | 44.65 | 53.22 | 53.13 |
| HO | 48.25 | 51.35 | 55.15 | 46.64 | 47.63 | 50.94 | 57.39 |
| TT | 55.20 | 46.90 | 50.47 | 56.43 | 49.94 | 42.31 | 48.65 |

### Best Model Performance

| Modality | Best Model | Accuracy | F1-Score | Recall |
|---|---|---|---|---|
| Structural | SVM | 0.71 | 0.84 | 0.83 |
| Functional | Random Forest | 0.75 | 0.76 | 0.75 |

## Methodology

1. **Data Loading** — Structural cortical thickness and functional BOLD time-series from ABIDE preprocessed (CPAC pipeline, no filter, no GSR)
2. **Feature Extraction** — 21 features per ROI for functional data; cortical thickness values for structural
3. **Preprocessing** — MinMax scaling (structural), StandardScaler + PCA (functional), applied within CV folds
4. **Classification** — 7 classifiers × 7 atlases × 2 modalities = 98 configurations
5. **Evaluation** — Stratified 10-fold cross-validation with accuracy, precision, recall, F1, confusion matrices

## References

- ABIDE Preprocessed: http://preprocessed-connectomes-project.org/abide/
- ANTs Pipeline: https://github.com/ANTsX/ANTs
- CPAC: http://fcp-indi.github.io

## Author

**Shahmir Chaudhry**  
Research Supervisor: **Dr. Saleha Raza**

## License

This project is for academic and research purposes.
