# Bearing Fault Diagnosis with Hybrid Deep Learning and Fuzzy Soft Computing

## Description
This project implements a vibration-signal fault diagnosis workflow for rotating machinery (CWRU-style `.mat` datasets). It combines signal processing, deep learning, and soft-computing techniques to classify fault conditions such as `Normal`, `Inner`, `Outer`, and `Ball`.

Two independent training scripts are provided:
- `main.py`: baseline/standard implementation with CNN, LSTM, hybrid CNN-LSTM, and an autoencoder.
- `main2.py`: novel implementation that extends the baseline idea with rough-set-inspired feature selection, fuzzy feature enhancement, and multiple hybrid evaluation variants.

## Features
- Robust recursive loading of `.mat` vibration files and automatic 1D signal extraction.
- Sliding-window segmentation for sample generation.
- Spectrogram conversion for image-like CNN input.
- Sequence modeling with LSTM on raw windowed signals.
- Hybrid CNN-LSTM architecture for fused spectral + temporal learning.
- Fuzzy logic integration:
  - Embedded fuzzy membership layer (`main.py`).
  - Fuzzy-enhanced engineered features and post-fuzzy prediction adjustment (`main2.py`).
- Automatic evaluation artifacts:
  - Accuracy/weighted metrics CSVs.
  - Confusion matrix plots.
  - Training curves (in `main2.py`).
  - Saved model files.

## Project Structure
```text
Soft_Computing_Project/
|-- main.py
|-- main2.py
|-- test.py
|-- test2.py
|-- requirements.txt
|-- data/
|   `-- feature_time_48k_2048_load_1.csv
|-- results/
|   `-- results_summary.csv
`-- results_novel/
    |-- all_features_preview.csv
    |-- roughset_cnn_lstm/
    |-- hybrid_cnn_lstm_fuzzy/
    |-- cnn_lstm_post_fuzzy/
    |-- confidence_fuzzy/
    `-- gradcam_fuzzy/
```

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
- **Engineered features (`main2.py`)**:
  - Time-domain: mean, std, RMS, peak-to-peak, skewness, kurtosis, crest factor.
  - Frequency-domain (Welch PSD): centroid, bandwidth, entropy, rolloff.

### 3) Model training approach
- Train/test split with stratification.
- Supervised classification with sparse categorical cross-entropy.
- Confusion-matrix based qualitative evaluation.
- Summary CSV generation for experiment comparison.

## Training Scripts
### `main.py` (baseline / standard)
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

### `main2.py` (novel implementation)
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

### 2) Run baseline training (`main.py`)
```bash
python main.py
```

Optional arguments:
```bash
python main.py --data-dir ./data/CWRU --result-dir ./results --epochs 10
```

### 3) Run novel pipeline (`main2.py`)
`main2.py` requires explicit input and output paths:
```bash
python main2.py --data-dir ./data/CWRU --result-dir ./results_novel --epochs 10
```

## Data expectation
- Input should be `.mat` vibration files under the specified `--data-dir` (recursive search).

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

### Existing recorded metrics (from current repository outputs)
- Baseline best shown in `results/results_summary.csv`: `hybrid_cnn_lstm` accuracy ~ `0.993`.
- Novel best shown in `results_novel/*_summary.csv`: `hybrid_cnn_lstm_fuzzy` accuracy ~ `0.9965`.

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
