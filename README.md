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
git clone https://github.com/<your-username>/STRP-2.git
cd STRP-2

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

| Category | Features |
|---|---|
| **Statistical** | mean, std, median, skewness, kurtosis |
| **Temporal** | slope, zero-crossing rate, autocorrelation (lag-1), time-to-peak, time-above-threshold |
| **Spectral** | dominant frequency, average PSD, band power (slow-5, slow-4, slow-3) |
| **Neuroimaging** | ALFF, fALFF |
| **Nonlinear** | Lyapunov exponent |
| **Additional** | IQR, range, RMS |

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
