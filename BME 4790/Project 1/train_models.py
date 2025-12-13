# ---- Editable constants ----
DATA_NPZ = "./artifacts/rbc_data.npz"
OUT_DIR  = "./artifacts"
EPOCHS   = 10
BATCH    = 32
# ---------------------------

import os, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def load_npz(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return d["X_train"], d["y_train"], d["X_val"], d["y_val"], d["X_test"], d["y_test"], meta

def build_tiny_cnn(input_shape, num_classes):
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])

def build_logreg_baseline(input_shape, num_classes):
    return tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax")
    ])

def train_and_plot(model, X_train, Y_train, X_val, Y_val, out_prefix, epochs=EPOCHS, batch_size=BATCH):
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)

    plt.figure(); plt.plot(hist.history["accuracy"]); plt.plot(hist.history["val_accuracy"])
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(["train","val"]); plt.title("Accuracy")
    plt.savefig(out_prefix + "_acc.png", dpi=150, bbox_inches="tight"); plt.close()

    plt.figure(); plt.plot(hist.history["loss"]); plt.plot(hist.history["val_loss"])
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(["train","val"]); plt.title("Loss")
    plt.savefig(out_prefix + "_loss.png", dpi=150, bbox_inches="tight"); plt.close()

def evaluate(model, X_test, y_test_1based, class_names, out_prefix, out_txt):
    y_true = (y_test_1based - 1).astype(int)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right"); plt.yticks(ticks, class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(out_prefix + "_cm.png", dpi=150, bbox_inches="tight"); plt.close()

    rep = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n"); f.write(rep)
    return acc

def main():
    X_tr, y_tr1, X_va, y_va1, X_te, y_te1, meta = load_npz(DATA_NPZ)
    class_names = meta["class_names"]; img_size = meta["img_size"]
    num_classes = len(class_names)
    input_shape = (img_size, img_size, 3)

    y_tr = (y_tr1 - 1).astype(int)
    y_va = (y_va1 - 1).astype(int)
    Y_tr = to_categorical(y_tr, num_classes=num_classes)
    Y_va = to_categorical(y_va, num_classes=num_classes)

    out_root = Path(OUT_DIR)
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)

    cnn = build_tiny_cnn(input_shape, num_classes)
    train_and_plot(cnn, X_tr, Y_tr, X_va, Y_va, str(out_root / "plots" / "tiny_cnn"))
    cnn.save(str(out_root / "models" / "tiny_cnn.keras"))
    _ = evaluate(cnn, X_te, y_te1, class_names, str(out_root / "plots" / "tiny_cnn"), str(out_root / "reports" / "tiny_cnn.txt"))

    logreg = build_logreg_baseline(input_shape, num_classes)
    train_and_plot(logreg, X_tr, Y_tr, X_va, Y_va, str(out_root / "plots" / "logreg"))
    logreg.save(str(out_root / "models" / "logreg.keras"))
    _ = evaluate(logreg, X_te, y_te1, class_names, str(out_root / "plots" / "logreg"), str(out_root / "reports" / "logreg.txt"))

    print("Done. Artifacts under:", str(out_root))

