#!/usr/bin/env python3
"""
run_experiment.py
...
"""
import os
import glob
import zipfile
import argparse
import networkx as nx
import numpy as np
import scipy.io as sio
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import tensorflow as tf



# ===== GPU CHECK =====
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)
# ====================

from tensorflow.keras import layers, models, backend as K
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ... rest of your code ...

# ---------------------------
# Helper: recursive search for first 1D numeric array
# ---------------------------
def find_first_vector(obj):
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return obj
        if obj.ndim == 2 and (obj.shape[0] == 1 or obj.shape[1] == 1):
            return obj.reshape(-1)
        try:
            for item in obj:
                res = find_first_vector(item)
                if res is not None:
                    return res
        except Exception:
            pass
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            res = find_first_vector(item)
            if res is not None:
                return res
    elif hasattr(obj, "__dict__"):
        for v in obj.__dict__.values():
            res = find_first_vector(v)
            if res is not None:
                return res
    return None

# ---------------------------
# Load .mat files robustly
# ---------------------------
def load_cwru_mat_files(data_dir):
    mat_files = glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True)
    signals = []
    labels = []
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {data_dir}.")
    for f in sorted(mat_files):
        try:
            mat = sio.loadmat(f, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            print("Skipping", f, "due to load error:", e)
            continue
        found_vec = None
        for k, v in mat.items():
            if k.startswith("__"):
                continue
            res = find_first_vector(v)
            if res is not None:
                found_vec = res
                break
        if found_vec is not None:
            signals.append(np.asarray(found_vec, dtype=np.float32))
            fname = os.path.basename(f).lower()
            dirname = os.path.basename(os.path.dirname(f)).lower()
            label = None
            if "normal" in fname or "normal" in dirname or "_n" in fname:
                label = "Normal"
            elif "outer" in fname or "outer" in dirname or "ow" in fname or "or" in fname:
                label = "Outer"
            elif "inner" in fname or "inner" in dirname or "ir" in fname or "iw" in fname:
                label = "Inner"
            elif "ball" in fname or "ball" in dirname or "br" in fname:
                label = "Ball"
            else:
                label_candidate = dirname if dirname and dirname != '' else fname.replace(".mat", "")
                label = label_candidate.capitalize()
            labels.append(label)
        else:
            print("No 1D signal found in", f)
    return signals, labels

# ---------------------------
# Segment windows
# ---------------------------
def segment_signals(signals, labels, window_size=2048, step=1024):
    X = []
    y = []
    for sig, lbl in zip(signals, labels):
        n = len(sig)
        if n < window_size:
            continue
        for start in range(0, n - window_size + 1, step):
            seg = sig[start:start + window_size]
            X.append(seg)
            y.append(lbl)
    return np.array(X), np.array(y)

# ---------------------------
# Spectrogram conversion
# ---------------------------
def to_spectrograms(X, fs=12000, nperseg=256, noverlap=128):
    specs = []
    for x in X:
        f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling='spectrum')
        Sxx = np.log1p(np.abs(Sxx))
        specs.append(Sxx)
    specs = np.array(specs)[..., np.newaxis]
    mean = specs.mean(axis=(1,2), keepdims=True)
    std = specs.std(axis=(1,2), keepdims=True)
    specs = (specs - mean) / (std + 1e-8)
    return specs

# ---------------------------
# Fuzzy logic system
# ---------------------------
def build_post_fuzzy_system():
    prob = ctrl.Antecedent(np.linspace(0,1,101), 'prob')
    conf = ctrl.Antecedent(np.linspace(0,1,101), 'conf')
    decision = ctrl.Consequent(np.linspace(0,1,101), 'decision')
    prob['low'] = fuzz.trimf(prob.universe, [0,0,0.5])
    prob['med'] = fuzz.trimf(prob.universe, [0,0.5,1])
    prob['high'] = fuzz.trimf(prob.universe, [0.5,1,1])
    conf['poor'] = fuzz.trimf(conf.universe, [0,0,0.5])
    conf['ok'] = fuzz.trimf(conf.universe, [0,0.5,1])
    conf['good'] = fuzz.trimf(conf.universe, [0.5,1,1])
    decision['reject'] = fuzz.trimf(decision.universe, [0,0,0.5])
    decision['uncertain'] = fuzz.trimf(decision.universe, [0,0.5,1])
    decision['confirm'] = fuzz.trimf(decision.universe, [0.5,1,1])
    rules = [ctrl.Rule(prob['high'] & conf['good'], decision['confirm']),
             ctrl.Rule(prob['med'] & conf['good'], decision['confirm']),
             ctrl.Rule(prob['high'] & conf['ok'], decision['confirm']),
             ctrl.Rule(prob['med'] & conf['ok'], decision['uncertain']),
             ctrl.Rule(prob['low'] | conf['poor'], decision['reject'])]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))

def post_process_with_fis(sim, prob_val, conf_val):
    sim.input['prob'] = float(prob_val)
    sim.input['conf'] = float(conf_val)
    sim.compute()
    return float(sim.output['decision'])

# ---------------------------
# Embedded fuzzy layer
# ---------------------------
class FuzzyMembershipLayer(layers.Layer):
    def __init__(self, n_sets=3, **kwargs):
        super().__init__(**kwargs)
        self.n_sets = n_sets
    def build(self, input_shape):
        features = int(input_shape[-1])
        self.centers = self.add_weight(shape=(features, self.n_sets),
                                       initializer='random_uniform', trainable=True)
        self.widths = self.add_weight(shape=(features, self.n_sets),
                                      initializer='random_uniform', trainable=True)
        super().build(input_shape)
    def call(self, inputs):
        x = inputs
        if K.ndim(x) == 3:
            x_col = K.mean(x, axis=1)
        else:
            x_col = x
        x_exp = K.expand_dims(x_col, axis=-1)
        c = K.expand_dims(self.centers, axis=0)
        w = K.expand_dims(K.abs(self.widths)+1e-6, axis=0)
        mem = K.exp(-K.square(x_exp - c)/(2*K.square(w)))
        batch = K.shape(x_col)[0]
        feats = int(x_col.shape[-1])
        return K.reshape(mem, (batch, feats*self.n_sets))
    def compute_output_shape(self, input_shape):
        return (input_shape[0], int(input_shape[-1])*self.n_sets)

# ---------------------------
# Model builders
# ---------------------------
def build_lstm(input_shape, n_classes, use_fuzzy_embedded=False):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64)(inp)
    x = layers.Dropout(0.3)(x)
    if use_fuzzy_embedded:
        x = layers.Concatenate()([x, FuzzyMembershipLayer()(x)])
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile('adam','sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_shape, n_classes, use_fuzzy_embedded=False):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(16,(3,3),activation='relu',padding='same')(inp)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Flatten()(x)
    if use_fuzzy_embedded:
        x = layers.Concatenate()([x, FuzzyMembershipLayer()(x)])
    x = layers.Dense(64,activation='relu')(x)
    out = layers.Dense(n_classes,activation='softmax')(x)
    model = models.Model(inp,out)
    model.compile('adam','sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_hybrid_cnn_lstm(spec_shape, seq_shape, n_classes, use_fuzzy_embedded=False):
    spec_inp = layers.Input(shape=spec_shape)
    seq_inp = layers.Input(shape=seq_shape)
    # CNN branch
    x1 = layers.Conv2D(16,(3,3),activation='relu',padding='same')(spec_inp)
    x1 = layers.MaxPool2D((2,2))(x1)
    x1 = layers.Conv2D(32,(3,3),activation='relu',padding='same')(x1)
    x1 = layers.MaxPool2D((2,2))(x1)
    x1 = layers.Flatten()(x1)
    # LSTM branch
    x2 = layers.LSTM(64)(seq_inp)
    if use_fuzzy_embedded:
        x2 = layers.Concatenate()([x2, FuzzyMembershipLayer()(x2)])
    x = layers.Concatenate()([x1,x2])
    x = layers.Dense(64,activation='relu')(x)
    out = layers.Dense(n_classes,activation='softmax')(x)
    model = models.Model([spec_inp,seq_inp], out)
    model.compile('adam','sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_autoencoder(input_shape, latent_dim=32):
    inp = layers.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(input_shape[0], activation='linear')(x)
    model = models.Model(inp,out)
    model.compile('adam','mse')
    return model

# ---------------------------
# Confusion matrix plot
# ---------------------------
def plot_conf_matrix(y_true, y_pred, labels, title, fname, result_dir):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,fname))
    plt.close()

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.makedirs(args.result_dir,exist_ok=True)
    # auto-extract zip if present
    zip_path = "/mnt/data/archive.zip"
    if os.path.exists(zip_path):
        print("Extracting archive.zip to", args.data_dir)
        with zipfile.ZipFile(zip_path,'r') as z:
            z.extractall(args.data_dir)
    # load signals
    signals, labels = load_cwru_mat_files(args.data_dir)
    print(f"Loaded {len(signals)} signals. Labels: {sorted(set(labels))}")
    X_raw, y_raw = segment_signals(signals, labels, window_size=args.window_size, step=args.step)
    print("Segmented windows:", X_raw.shape)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    # normalize
    X_scaled = (X_raw - X_raw.mean(axis=1,keepdims=True)) / (X_raw.std(axis=1,keepdims=True)+1e-8)
    X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_enc,test_size=args.test_size,stratify=y_enc,random_state=args.random_state)
    # ---------------------------
    # GPU + mixed precision setup
    # ---------------------------
    # ---------------------------
    # CPU setup (disable GPU & mixed precision)
    # ---------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable GPU

    from tensorflow.keras import mixed_precision

    # Convert data to float32 tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    # Use float32 policy (no mixed precision)
    policy = mixed_precision.Policy('float32')
    mixed_precision.set_global_policy(policy)
    print("Running on CPU only with policy:", mixed_precision.global_policy())

    results = {}

    # LSTM baseline + fuzzy
    if args.use_lstm:
        print("--- LSTM baseline ---")
        X_tr_l = X_train[...,np.newaxis]; X_te_l = X_test[...,np.newaxis]
        lstm = build_lstm((X_tr_l.shape[1],1), n_classes, use_fuzzy_embedded=False)
        lstm.fit(X_tr_l,y_train,validation_split=0.1,epochs=args.epochs,batch_size=args.batch_size,verbose=2)
        preds = np.argmax(lstm.predict(X_te_l),axis=1)
        results['lstm_baseline'] = (preds, None, accuracy_score(y_test,preds))

        lstm_emb = build_lstm((X_tr_l.shape[1],1), n_classes, use_fuzzy_embedded=True)
        lstm_emb.fit(X_tr_l,y_train,validation_split=0.1,epochs=args.epochs,batch_size=args.batch_size,verbose=2)
        preds_emb = np.argmax(lstm_emb.predict(X_te_l),axis=1)
        results['lstm_embedded'] = (preds_emb, None, accuracy_score(y_test,preds_emb))

    # CNN baseline + fuzzy
    if args.use_cnn:
        specs_tr = to_spectrograms(X_train, fs=args.sample_rate, nperseg=args.nperseg, noverlap=args.noverlap)
        specs_te = to_spectrograms(X_test, fs=args.sample_rate, nperseg=args.nperseg, noverlap=args.noverlap)
        cnn = build_cnn(specs_tr.shape[1:], n_classes, use_fuzzy_embedded=False)
        cnn.fit(specs_tr,y_train,validation_split=0.1,epochs=args.epochs,batch_size=args.batch_size,verbose=2)
        preds_c = np.argmax(cnn.predict(specs_te),axis=1)
        results['cnn_baseline'] = (preds_c,None,accuracy_score(y_test,preds_c))

        cnn_emb = build_cnn(specs_tr.shape[1:], n_classes, use_fuzzy_embedded=True)
        cnn_emb.fit(specs_tr,y_train,validation_split=0.1,epochs=args.epochs,batch_size=args.batch_size,verbose=2)
        preds_ce = np.argmax(cnn_emb.predict(specs_te),axis=1)
        results['cnn_embedded'] = (preds_ce,None,accuracy_score(y_test,preds_ce))

    # Hybrid CNN+LSTM
    hybrid = build_hybrid_cnn_lstm(specs_tr.shape[1:], (X_tr_l.shape[1],1), n_classes, use_fuzzy_embedded=True)
    hybrid.fit([specs_tr,X_tr_l], y_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    preds_h = np.argmax(hybrid.predict([specs_te,X_te_l]),axis=1)
    results['hybrid_cnn_lstm'] = (preds_h,None,accuracy_score(y_test,preds_h))

    # Autoencoder (unsupervised)
    ae = build_autoencoder((X_train.shape[1],))
    ae.fit(X_train,X_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size, verbose=2)
    reconstructed = ae.predict(X_test)
    mse = np.mean(np.square(X_test-reconstructed),axis=1)
    results['autoencoder'] = (None, mse, None)

    # Plot confusion matrices
    labels_list = list(le.classes_)
    for k,v in results.items():
        if 'preds' in k or 'cnn' in k or 'lstm' in k or 'hybrid' in k:
            preds = v[0]
            if preds is not None:
                plot_conf_matrix(y_test, preds, labels_list, k, f"cm_{k}.png", args.result_dir)

    # Save encoder and results
    joblib.dump(le, os.path.join(args.result_dir,'label_encoder.pkl'))
    joblib.dump(results, os.path.join(args.result_dir,'results_raw.pkl'))
    pd.DataFrame([{'name':k,'accuracy':v[2]} for k,v in results.items() if v[2] is not None]).to_csv(
        os.path.join(args.result_dir,'results_summary.csv'),index=False)
    print("All results saved in", args.result_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data/CWRU")
    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--sample-rate", type=int, default=12000)
    parser.add_argument("--window-size", type=int, default=2048)
    parser.add_argument("--step", type=int, default=1024)
    parser.add_argument("--nperseg", type=int, default=256)
    parser.add_argument("--noverlap", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--use-lstm", action="store_true", default=True)
    parser.add_argument("--use-cnn", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
