"""Evaluasi model (.keras atau .tflite) pada dataset uji.

    python evaluate.py --model outputs/best.keras --data-dir ../dataset/test
    python evaluate.py --model exported/freshness_mnv4s_r224_fp16.tflite --data-dir ../dataset/test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402

from config import LABELS_FILENAME  # noqa: E402
from dataset import load_eval_dataset  # noqa: E402


def read_labels(model_path: Path):
    f = model_path.parent / LABELS_FILENAME
    return [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    model_path = Path(args.model)
    out_dir = Path(args.output_dir) if args.output_dir else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = read_labels(model_path)
    ds, class_names = load_eval_dataset(args.data_dir, args.batch_size, class_names=labels)

    images, y_true = [], []
    for x, y in ds:
        images.append(x.numpy())
        y_true.append(y.numpy().argmax(1))
    images, y_true = np.concatenate(images), np.concatenate(y_true)

    t0 = time.perf_counter()
    if model_path.suffix == ".tflite":
        from export_tflite import tflite_predict
        probs = tflite_predict(model_path.read_bytes(), images)
    else:
        import keras
        probs = keras.models.load_model(model_path, compile=False).predict(images, verbose=0)
    ms_per_img = (time.perf_counter() - t0) * 1000 / len(images)
    y_pred = probs.argmax(1)

    idx = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=idx, target_names=class_names,
                                   digits=4, output_dict=True, zero_division=0)
    print(classification_report(y_true, y_pred, labels=idx, target_names=class_names,
                                digits=4, zero_division=0))
    print(f"Rata-rata waktu inferensi (host): {ms_per_img:.2f} ms/gambar")

    cm = confusion_matrix(y_true, y_pred, labels=idx)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", xticks_rotation=30)
    plt.title(f"Confusion Matrix - {model_path.name}")
    plt.tight_layout()
    stem = model_path.stem
    plt.savefig(out_dir / f"confusion_matrix_{stem}.png", dpi=120)
    (out_dir / f"eval_{stem}.json").write_text(
        json.dumps({"report": report, "confusion_matrix": cm.tolist(), "ms_per_image": ms_per_img}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
