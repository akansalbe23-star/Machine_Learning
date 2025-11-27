import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GRU, Dropout
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import shap
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# Basic EEG parameters
# -----------------------------
FS = 128
EPOCH_SEC = 3
EPOCH_LEN = FS * EPOCH_SEC
EEG_CHANNELS = 16
ALPHA_LOW, ALPHA_HIGH = 7.5, 15

# -----------------------------
# Filter utilities
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype="band")

def bandpass_filter(data, lowcut=ALPHA_LOW, highcut=ALPHA_HIGH, fs=FS, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return lfilter(b, a, data, axis=0)

# -----------------------------
# Load and segment .eea EEG files
# -----------------------------
def load_and_segment_eeg(filepath, label):
    df = pd.read_csv(filepath, header=None)
    flat = df.values.flatten()

    if len(flat) != 122880:
        return [], []

    eeg = flat.reshape(16, 7680).T
    eeg = bandpass_filter(eeg)

    segments, labels = [], []
    for start in range(0, eeg.shape[0] - EPOCH_LEN + 1, EPOCH_LEN):
        seg = eeg[start:start + EPOCH_LEN]
        segments.append(seg)
        labels.append(label)

    return segments, labels

def load_dataset(folder, label):
    X, y = [], []
    for fname in os.listdir(folder):
        if fname.endswith(".eea"):
            fpath = os.path.join(folder, fname)
            segs, labels = load_and_segment_eeg(fpath, label)
            X.extend(segs)
            y.extend(labels)
    return np.array(X), np.array(y)

NORMAL_DIR = "/kaggle/input/eeg-schizophrenia/Schizophrenia/norm"
SCHIZO_DIR = "/kaggle/input/eeg-schizophrenia/Schizophrenia/sch"

X_norm, y_norm = load_dataset(NORMAL_DIR, 0)
X_sch, y_sch = load_dataset(SCHIZO_DIR, 1)

X = np.concatenate([X_norm, X_sch], axis=0)
y = np.concatenate([y_norm, y_sch], axis=0)

# -----------------------------
# Normalization
# -----------------------------
scaler = StandardScaler()
X_reshaped = X.reshape(-1, X.shape[2])
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)

# -----------------------------
# CNN-GRU deep feature extractor
# -----------------------------
input_layer = Input(shape=(384, 16))
x = Conv1D(32, kernel_size=5, activation="relu", padding="same")(input_layer)
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.2)(x)
x = tf.keras.layers.Bidirectional(GRU(64, return_sequences=False))(x)

deep_feature_model = Model(inputs=input_layer, outputs=x)
X_deep = deep_feature_model.predict(X, verbose=1)

# -----------------------------
# Handcrafted features
# -----------------------------
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def compute_hjorth_parameters(signal):
    first = np.diff(signal, axis=0)
    second = np.diff(first, axis=0)
    activity = np.var(signal, axis=0)
    mobility = np.sqrt(np.var(first, axis=0) / activity)
    complexity = np.sqrt(np.var(second, axis=0) / np.var(first, axis=0)) / mobility
    return activity, mobility, complexity

def compute_handcrafted_features(X):
    features = []
    for segment in X:
        fv = []
        for ch in range(segment.shape[1]):
            x = segment[:, ch]
            fv.extend([np.mean(x), np.std(x), skew(x), kurtosis(x)])

            a, m, c = compute_hjorth_parameters(x)
            fv.extend([a, m, c])

            f, psd = welch(x, fs=FS, nperseg=128)
            for band in [(4,7), (7.5,15), (15,30), (30,45)]:
                bp = np.trapz(psd[(f >= band[0]) & (f <= band[1])],
                              f[(f >= band[0]) & (f <= band[1])])
                fv.append(bp)
        features.append(fv)
    return np.array(features)

X_hand = compute_handcrafted_features(X)

# -----------------------------
# Feature fusion + classifier
# -----------------------------
from xgboost import XGBClassifier

X_fused = np.concatenate([X_deep, X_hand], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X_fused, y, test_size=0.2, stratify=y, random_state=42
)

clf = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=20,
    verbose=False
)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0,1], ["Healthy", "Schizophrenia"])
plt.yticks([0,1], ["Healthy", "Schizophrenia"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.show()

# -----------------------------
# SHAP Interpretability
# -----------------------------
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test[:100])

feature_names = (
    [f"d{i}" for i in range(X_deep.shape[1])] +
    [f"h{i}" for i in range(X_hand.shape[1])]
)

shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, max_display=20)

# -----------------------------
# Saliency map for CNN-GRU
# -----------------------------
sample = X[0:1]
with tf.GradientTape() as tape:
    inp = tf.convert_to_tensor(sample)
    tape.watch(inp)
    preds = deep_feature_model(inp)
    idx = tf.argmax(preds[0])
    out = preds[:, idx]
    grads = tape.gradient(out, inp)

sal = tf.reduce_mean(tf.abs(grads), axis=2).numpy()[0]
plt.plot(sal)
plt.title("Saliency Map")
plt.xlabel("Time")
plt.ylabel("Importance")
plt.show()
