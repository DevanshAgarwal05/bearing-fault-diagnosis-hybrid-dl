# Bearing Fault Diagnosis with Hybrid Deep Learning and Fuzzy Soft Computing

## Description
This project implements a vibration-signal fault diagnosis workflow for rotating machinery (CWRU-style `.mat` datasets). It combines signal processing, deep learning, and soft-computing techniques to classify fault conditions such as `Normal`, `Inner`, `Outer`, and `Ball`.

Two independent training scripts are provided:
- `baseline_pipeline.py`: baseline/standard implementation with CNN, LSTM, hybrid CNN-LSTM, and an autoencoder.
- `novel_fuzzy_pipeline.py`: novel implementation that extends the baseline idea with rough-set-inspired feature selection, fuzzy feature enhancement, and multiple hybrid evaluation variants.

## Features
- Robust recursive loading of `.mat` vibration files and automatic 1D signal extraction.
- Sliding-window segmentation for sample generation.
- Spectrogram conversion for image-like CNN input.
- Sequence modeling with LSTM on raw windowed signals.
- Hybrid CNN-LSTM architecture for fused spectral + temporal learning.
- Fuzzy logic integration:
  - Embedded fuzzy membership layer (`baseline_pipeline.py`).
  - Fuzzy-enhanced engineered features and post-fuzzy prediction adjustment (`novel_fuzzy_pipeline.py`).
- Automatic evaluation artifacts:
  - Accuracy/weighted metrics CSVs.
  - Confusion matrix plots.
  - Training curves (in `novel_fuzzy_pipeline.py`).
  - Saved model files.

## Project Structure
```text
Soft_Computing_Project/
|-- src/
|   |-- baseline_pipeline.py     # baseline training pipeline
|   `-- novel_fuzzy_pipeline.py  # novel (rough-set + fuzzy) pipeline
|-- requirements.txt
|-- scripts/
|   |-- check_gpu.py         # prints TensorFlow version and detected GPUs
|   `-- inspect_mat_files.py # lists keys of every data/raw/*.mat file
|-- data/                    # local only (git-ignored)
|   `-- raw/                 # CWRU .mat vibration files
|-- results_reference/       # tracked: lightweight reference results
|   |-- baseline/            # results_summary.csv + confusion matrices
|   `-- novel/<experiment>/  # <experiment>_summary.csv + confusion matrix
|-- results/                 # local training output of baseline_pipeline.py (git-ignored)
`-- results_novel/           # local training output of novel_fuzzy_pipeline.py (git-ignored)
```

Model files (`*.h5`), pickles (`*.pkl`), `*.npz` archives, `data/` and course documents are kept local and are not tracked.

## Methodology
### 1) Data preprocessing
- Load all `.mat` files recursively from the data directory.
- Extract first valid 1D vector from each file.
- Infer labels from file/folder naming patterns (`normal`, `inner`, `outer`, `ball`, etc.).
- Segment each signal into fixed windows (`window_size`, `step`).
- Encode labels using `LabelEncoder`.

### 2) Representation building
- **Sequence representation**: normalized raw windows for LSTM.
- **Spectral representation**: log-scaled spectrograms for CNN/hybrid models.
- **Engineered features (`novel_fuzzy_pipeline.py`)**:
  - Time-domain: mean, std, RMS, peak-to-peak, skewness, kurtosis, crest factor.
  - Frequency-domain (Welch PSD): centroid, bandwidth, entropy, rolloff.

### 3) Model training approach
- Train/test split with stratification.
- Supervised classification with sparse categorical cross-entropy.
- Confusion-matrix based qualitative evaluation.
- Summary CSV generation for experiment comparison.

## Training Scripts
### `baseline_pipeline.py` (baseline / standard)
Purpose:
- Establishes baseline deep models and fuzzy-augmented variants on the same segmented dataset.

Workflow:
1. Load and segment `.mat` vibration data.
2. Build normalized sequential and spectrogram inputs.
3. Train/evaluate:
   - `lstm_baseline`
   - `lstm_embedded` (LSTM + fuzzy membership embedding)
   - `cnn_baseline`
   - `cnn_embedded` (CNN + fuzzy membership embedding)
   - `hybrid_cnn_lstm`
   - `autoencoder` (reconstruction error only)
4. Save confusion matrices, label encoder, raw result object, and summary CSV.

Notes:
- CPU-only mode is enforced (`CUDA_VISIBLE_DEVICES=-1`).
- Includes a defined post-fuzzy rule system helper, while model comparison primarily uses embedded fuzzy membership for LSTM/CNN variants.

### `novel_fuzzy_pipeline.py` (novel implementation)
Purpose:
- Implements a broader experimental framework combining rough-set-style feature relevance and fuzzy-enhanced hybrid modeling.

What makes it novel (from code behavior):
1. **Rough-set-inspired greedy feature selection**
   - Discretizes engineered features.
   - Maximizes dependency degree via positive region analysis.
2. **Fuzzy enhancement of selected engineered features**
   - Converts selected features into membership activations (Gaussian-like memberships).
3. **Comparative five-model pipeline**
   - `roughset_cnn_lstm`
   - `hybrid_cnn_lstm_fuzzy` (hybrid network + fuzzy feature input branch)
   - `cnn_lstm_post_fuzzy`
   - `confidence_fuzzy`
   - `gradcam_fuzzy`
4. **Extended reporting**
   - Weighted precision/recall/F1 in addition to accuracy.
   - Training history plots per experiment.
   - Saved model per experiment directory.

Important implementation detail:
- The `gradcam_fuzzy` experiment currently prints an informational placeholder for Grad-CAM generation (not a full Grad-CAM map implementation in this script).

## How to Run
### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run baseline training (`baseline_pipeline.py`)
```bash
python src/baseline_pipeline.py
```

Optional arguments:
```bash
python src/baseline_pipeline.py --data-dir ./data/raw --result-dir ./results --epochs 10
```

Note: the built-in default for `--data-dir` is `./data/CWRU`, so pass `--data-dir ./data/raw` explicitly.

### 3) Run novel pipeline (`novel_fuzzy_pipeline.py`)
`novel_fuzzy_pipeline.py` requires explicit input and output paths:
```bash
python src/novel_fuzzy_pipeline.py --data-dir ./data/raw --result-dir ./results_novel --epochs 10
```

### 4) Helper scripts
```bash
python scripts/check_gpu.py          # TensorFlow version / GPU detection
python scripts/inspect_mat_files.py  # list keys in data/raw/*.mat
```

## Data expectation
- Input should be `.mat` vibration files under the specified `--data-dir` (recursive search). The expected location is `data/raw/` (10 CWRU files: Normal, plus Ball/Inner/Outer faults at 0.007, 0.014 and 0.021 in.).
- `data/` is not tracked; place the files there locally.

## Results and Outputs
### Baseline (`results/`)
- `results_summary.csv` with model-level accuracy.
- Confusion matrices per classification model.
- `label_encoder.pkl`, `results_raw.pkl`.

### Novel (`results_novel/<experiment>/`)
- `<experiment>_summary.csv` with:
  - accuracy
  - precision_weighted
  - recall_weighted
  - f1_weighted
- `<experiment>_confusion.png`
- `<experiment>_accuracy_loss.png`
- `<experiment>_model.h5`

### Reference metrics
Lightweight reference outputs (summary CSVs and confusion matrices) are tracked in `results_reference/`.

| Pipeline | Model | Accuracy |
|---|---|---|
| Baseline (`baseline_pipeline.py`) | `lstm_baseline` | 33.45% |
| Baseline (`baseline_pipeline.py`) | `lstm_embedded` | 43.43% |
| Baseline (`baseline_pipeline.py`) | `cnn_baseline` | 98.94% |
| Baseline (`baseline_pipeline.py`) | `cnn_embedded` | 98.59% |
| Baseline (`baseline_pipeline.py`) | `hybrid_cnn_lstm` (best) | **99.30%** |
| Novel (`novel_fuzzy_pipeline.py`) | `cnn_lstm_post_fuzzy` | 98.12% |
| Novel (`novel_fuzzy_pipeline.py`) | `gradcam_fuzzy` | 98.47% |
| Novel (`novel_fuzzy_pipeline.py`) | `confidence_fuzzy` | 99.18% |
| Novel (`novel_fuzzy_pipeline.py`) | `roughset_cnn_lstm` | 99.41% |
| Novel (`novel_fuzzy_pipeline.py`) | `hybrid_cnn_lstm_fuzzy` (best) | **99.65%** |

Best baseline ~99.30% vs best novel ~99.65%.

## Tech Stack
- Python
- NumPy, SciPy, pandas
- TensorFlow / Keras
- scikit-learn
- scikit-fuzzy
- Matplotlib, Seaborn
- joblib

## Future Improvements
- Implement full Grad-CAM map extraction and visualization in `gradcam_fuzzy`.
- Add fixed random seeds for end-to-end reproducibility across all frameworks.
- Add experiment configuration files (YAML/JSON) for cleaner benchmark management.
- Add unit tests for feature extraction and rough-set selection utilities.
