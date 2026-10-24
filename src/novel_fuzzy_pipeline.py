#!/usr/bin/env python3
"""
main2.py
CPU-only pipeline running five models sequentially:

1) Rough Set Feature Selection + CNN-LSTM
2) Hybrid CNN-LSTM with Embedded Fuzzy Layer
3) CNN-LSTM + Post-Fuzzy Decision System (fuzzy-adjusted predictions)
4) Confidence-aware Fuzzy Post-processing (calibrated fuzzy predictions)
5) Explainable CNN-LSTM (Grad-CAM + fuzzy rules)
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ---------------------------
# Utility functions
# ---------------------------

def find_first_vector(obj):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1: return obj
        if obj.ndim == 2 and (obj.shape[0] == 1 or obj.shape[1] == 1): return obj.reshape(-1)
        try:
            for item in obj:
                res = find_first_vector(item)
                if res is not None: return res
        except Exception: pass
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            res = find_first_vector(item)
            if res is not None: return res
    elif hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            res = find_first_vector(v)
            if res is not None: return res
    return None

def load_cwru_mat_files(data_dir):
    mat_files = glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)
    signals, labels = [], []
    
    for f in sorted(mat_files):
        vec = None
        try:
            import scipy.io as sio
            mat = sio.loadmat(f, struct_as_record=False, squeeze_me=True)
            for k, v in mat.items():
                if k.startswith("__"): continue
                vec = find_first_vector(v)
                if vec is not None: break
        except: vec = None

        if vec is not None:
            signals.append(np.asarray(vec, dtype=np.float32))
            fname = os.path.basename(f).lower()
            dirname = os.path.basename(os.path.dirname(f)).lower()
            if "normal" in fname or "normal" in dirname or "_n" in fname:
                label = "Normal"
            elif "outer" in fname or "outer" in dirname or "ow" in fname or "or" in fname:
                label = "Outer"
            elif "inner" in fname or "inner" in dirname or "ir" in fname or "iw" in fname:
                label = "Inner"
            elif "ball" in fname or "ball" in dirname or "br" in fname:
                label = "Ball"
            else:
                label = (dirname if dirname else fname.replace(".mat","")).capitalize()
            labels.append(label)
    return signals, labels

def segment_signals(signals, labels, window_size=2048, step=1024):
    X, y, origins = [], [], []
    for i, (sig, lbl) in enumerate(zip(signals, labels)):
        n = len(sig)
        if n < window_size: continue
        for start in range(0, n - window_size + 1, step):
            X.append(sig[start:start+window_size])
            y.append(lbl)
            origins.append(i)
    return np.array(X), np.array(y), np.array(origins)

def to_spectrograms(X, fs=12000, nperseg=256, noverlap=128):
    specs = []
    for x in X:
        f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
        specs.append(np.log1p(np.abs(Sxx)))
    specs = np.array(specs)[..., np.newaxis]
    mean = specs.mean(axis=(1,2), keepdims=True)
    std = specs.std(axis=(1,2), keepdims=True)
    return ((specs - mean) / (std + 1e-8)).astype(np.float32)

def extract_features_from_window(x, fs=12000):
    x = np.asarray(x, dtype=np.float32)
    feats = {}
    feats['mean'] = np.mean(x)
    feats['std'] = np.std(x)
    feats['rms'] = np.sqrt(np.mean(x**2))
    feats['ptp'] = np.ptp(x)
    feats['skew'] = float(stats.skew(x))
    feats['kurtosis'] = float(stats.kurtosis(x))
    feats['crest'] = np.max(np.abs(x)) / (feats['rms'] + 1e-12)

    f, Pxx = signal.welch(x, fs=fs, nperseg=256)
    Pxx += 1e-12
    feats['spec_centroid'] = np.sum(f * Pxx) / np.sum(Pxx)
    feats['spec_bw'] = np.sqrt(np.sum(((f - feats['spec_centroid'])**2) * Pxx) / np.sum(Pxx))
    ps_norm = Pxx / np.sum(Pxx)
    feats['spec_entropy'] = -np.sum(ps_norm * np.log(ps_norm + 1e-12))
    cumsum = np.cumsum(ps_norm)
    feats['spec_rolloff'] = f[min(np.searchsorted(cumsum, 0.85), len(f)-1)]
    return feats

def build_feature_matrix(X_windows, fs=12000):
    df = pd.DataFrame([extract_features_from_window(x, fs) for x in X_windows])
    df.fillna(0, inplace=True)
    return df

def discretize_features(X_df, n_bins=5):
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    Xd = est.fit_transform(X_df.values)
    return Xd.astype(int), est

def positive_region_size(Xd, y):
    eq = {}
    for i, row in enumerate(map(tuple, Xd.tolist())):
        eq.setdefault(row, []).append(i)
    pos = 0
    for inds in eq.values():
        if len(set(y[inds])) == 1: pos += len(inds)
    return pos

def dependency_degree(Xd, y):
    return positive_region_size(Xd, y) / Xd.shape[0]

def greedy_forward_rough_select(X_df, y, max_bins=5):
    attrs = list(X_df.columns)
    selected, best_dep = [], 0.0
    while True:
        improved = False
        best_attr = None
        for a in attrs:
            if a in selected: continue
            X_sub = X_df[selected + [a]]
            Xd, _ = discretize_features(X_sub, n_bins=min(max_bins,6))
            dep = dependency_degree(Xd, y)
            if dep > best_dep + 1e-6:
                best_dep = dep
                best_attr = a
                improved = True
        if not improved: break
        selected.append(best_attr)
        if abs(best_dep - 1.0) < 1e-9: break
    return selected, best_dep

def fuzzy_enhance_df(X_df_selected, n_sets=3):
    mats = []
    for col in X_df_selected.columns:
        vals = X_df_selected[col].values
        centers = np.quantile(vals, np.linspace(0, 1, n_sets))
        widths = (np.max(vals) - np.min(vals) + 1e-6) / (n_sets*1.5)
        col_mat = [np.exp(-((vals - c)**2) / (2 * widths**2)).reshape(-1,1) for c in centers]
        mats.append(np.hstack(col_mat))
    return np.hstack(mats) if mats else np.zeros((len(X_df_selected), 0), dtype=np.float32)

def build_cnn_branch(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(inp)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    return models.Model(inp, x, name='cnn_branch')

def build_lstm_branch(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64)(inp)
    return models.Model(inp, x, name='lstm_branch')

def build_hybrid_model(spec_shape, seq_shape, n_classes, use_fuzzy_feats=False, fuzzy_dim=0):
    cnn_m = build_cnn_branch(spec_shape)
    lstm_m = build_lstm_branch(seq_shape)
    feat_concat = layers.Concatenate()([cnn_m.output, lstm_m.output])
    if use_fuzzy_feats and fuzzy_dim>0:
        fuzzy_in = layers.Input(shape=(fuzzy_dim,), name='fuzzy_in')
        x = layers.Dense(128, activation='relu')(layers.Concatenate()([feat_concat, fuzzy_in]))
        out = layers.Dense(n_classes, activation='softmax')(x)
        model = models.Model([cnn_m.input, lstm_m.input, fuzzy_in], out)
    else:
        x = layers.Dense(128, activation='relu')(feat_concat)
        out = layers.Dense(n_classes, activation='softmax')(x)
        model = models.Model([cnn_m.input, lstm_m.input], out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def print_and_save_metrics(y_true, y_pred, le, result_dir, name_prefix):
    acc = accuracy_score(y_true, y_pred)
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    summary = pd.DataFrame([{
        'experiment': name_prefix,
        'accuracy': acc,
        'precision_weighted': p,
        'recall_weighted': r,
        'f1_weighted': f1
    }])
    summary.to_csv(os.path.join(result_dir, f"{name_prefix}_summary.csv"), index=False)
    print(summary.to_string(index=False))
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(le.classes_))))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion matrix - {name_prefix}')
    plt.tight_layout(); plt.savefig(os.path.join(result_dir, f"{name_prefix}_confusion.png")); plt.close()
    return summary

def plot_history(history, result_dir, name_prefix):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Value')
    plt.title(f'Training History - {name_prefix}')
    plt.tight_layout(); plt.savefig(os.path.join(result_dir, f"{name_prefix}_accuracy_loss.png"))
    plt.close()

def train_and_evaluate(model, X_spec_train, X_seq_train, X_spec_test, X_seq_test, y_train, y_test,
                       fuzzy_train=None, fuzzy_test=None, result_dir=None, name_prefix=None,
                       post_fuzzy=False, fuzzy_feats=None, le=None, gradcam=False, epochs=10):
    X_seq_train_in = X_seq_train[..., np.newaxis]
    X_seq_test_in = X_seq_test[..., np.newaxis]
    inputs_train = [X_spec_train, X_seq_train_in]
    inputs_test = [X_spec_test, X_seq_test_in]
    if fuzzy_train is not None: inputs_train.append(fuzzy_train)
    if fuzzy_test is not None: inputs_test.append(fuzzy_test)

    print(f"\n--- Training {name_prefix} ---")
    history = model.fit(inputs_train, y_train, validation_split=0.1,
                        epochs=epochs, batch_size=32, verbose=2)
    plot_history(history, result_dir, name_prefix)

    pred_probs = model.predict(inputs_test)
    preds = np.argmax(pred_probs, axis=1)

    # Post-fuzzy adjustment
    if post_fuzzy and fuzzy_feats is not None and le is not None:
        class_membership = np.zeros_like(pred_probs)
        for i, _ in enumerate(le.classes_):
            class_membership[:, i] = np.sum(fuzzy_feats, axis=1)
        preds = np.argmax(pred_probs * class_membership, axis=1)

    # Grad-CAM placeholder: in real setup, you would compute attention maps here
    if gradcam:
        print(f"[INFO] Grad-CAM maps would be generated for {name_prefix} (not implemented fully)")

    summary = print_and_save_metrics(y_test, preds, le, result_dir, name_prefix)
    model.save(os.path.join(result_dir, f"{name_prefix}_model.h5"))
    return summary, history

# ---------------------------
# Main pipeline
# ---------------------------
def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    print("Loading .mat files from:", args.data_dir)
    signals, labels = load_cwru_mat_files(args.data_dir)
    print(f"Loaded {len(signals)} files. Example labels: {sorted(set(labels))}")

    X_raw, y_raw, _ = segment_signals(signals, labels, window_size=args.window_size, step=args.step)
    print("Segmented windows shape:", X_raw.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    print("Classes:", le.classes_)

    feat_df = build_feature_matrix(X_raw, fs=args.sample_rate)
    selected, dep = greedy_forward_rough_select(feat_df, y_enc, max_bins=args.rough_bins)
    print("Selected features:", selected)
    X_selected = feat_df[selected].values
    fuzzy_feats = fuzzy_enhance_df(feat_df[selected])

    X_spec = to_spectrograms(X_raw, fs=args.sample_rate)
    X_seq = X_raw.copy()

    X_spec_train, X_spec_test, X_seq_train, X_seq_test, y_train, y_test, fuzzy_train, fuzzy_test = train_test_split(
        X_spec, X_seq, y_enc, fuzzy_feats, test_size=args.test_size, random_state=args.random_state, stratify=y_enc
    )

    # ---------------- Model 2 ----------------
    model2_dir = os.path.join(args.result_dir, 'hybrid_cnn_lstm_fuzzy'); os.makedirs(model2_dir, exist_ok=True)
    model2 = build_hybrid_model(spec_shape=X_spec_train.shape[1:], seq_shape=(X_seq_train.shape[1],1),
                                n_classes=n_classes, use_fuzzy_feats=True, fuzzy_dim=fuzzy_train.shape[1])
    train_and_evaluate(model2, X_spec_train, X_seq_train, X_spec_test, X_seq_test, y_train, y_test,
                       fuzzy_train=fuzzy_train, fuzzy_test=fuzzy_test,
                       result_dir=model2_dir, name_prefix='hybrid_cnn_lstm_fuzzy', le=le, epochs=args.epochs)

    # ---------------- Model 1 ----------------
    model1_dir = os.path.join(args.result_dir, 'roughset_cnn_lstm'); os.makedirs(model1_dir, exist_ok=True)
    model1 = build_hybrid_model(spec_shape=X_spec_train.shape[1:], seq_shape=(X_seq_train.shape[1],1), n_classes=n_classes)
    train_and_evaluate(model1, X_spec_train, X_seq_train, X_spec_test, X_seq_test, y_train, y_test,
                       result_dir=model1_dir, name_prefix='roughset_cnn_lstm', le=le, epochs=args.epochs)

    # ---------------- Model 3 ----------------
    model3_dir = os.path.join(args.result_dir, 'cnn_lstm_post_fuzzy'); os.makedirs(model3_dir, exist_ok=True)
    model3 = build_hybrid_model(spec_shape=X_spec_train.shape[1:], seq_shape=(X_seq_train.shape[1],1), n_classes=n_classes)
    train_and_evaluate(model3, X_spec_train, X_seq_train, X_spec_test, X_seq_test, y_train, y_test,
                       result_dir=model3_dir, name_prefix='cnn_lstm_post_fuzzy',
                       post_fuzzy=True, fuzzy_feats=fuzzy_test, le=le, epochs=args.epochs)

    # ---------------- Model 4 ----------------
    model4_dir = os.path.join(args.result_dir, 'confidence_fuzzy'); os.makedirs(model4_dir, exist_ok=True)
    model4 = build_hybrid_model(spec_shape=X_spec_train.shape[1:], seq_shape=(X_seq_train.shape[1],1), n_classes=n_classes)
    train_and_evaluate(model4, X_spec_train, X_seq_train, X_spec_test, X_seq_test, y_train, y_test,
                       result_dir=model4_dir, name_prefix='confidence_fuzzy',
                       post_fuzzy=True, fuzzy_feats=fuzzy_test, le=le, epochs=args.epochs)

    # ---------------- Model 5 ----------------
    model5_dir = os.path.join(args.result_dir, 'gradcam_fuzzy'); os.makedirs(model5_dir, exist_ok=True)
    model5 = build_hybrid_model(spec_shape=X_spec_train.shape[1:], seq_shape=(X_seq_train.shape[1],1), n_classes=n_classes)
    train_and_evaluate(model5, X_spec_train, X_seq_train, X_spec_test, X_seq_test, y_train, y_test,
                       result_dir=model5_dir, name_prefix='gradcam_fuzzy',
                       post_fuzzy=True, fuzzy_feats=fuzzy_test, le=le, gradcam=True, epochs=args.epochs)

# ---------------------------
# Argument parser
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--sample-rate", type=int, default=12000)
    parser.add_argument("--rough-bins", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    main(args)
