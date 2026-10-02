"""Training MobileNetV4-Conv-Small R224 untuk deteksi kesegaran daging ayam.

Contoh:
    # Transfer learning (disarankan) - jalankan convert_timm_weights.py dulu
    python train.py --data-dir ../dataset --pretrained-weights weights/mnv4_conv_small_imagenet.weights.h5

    # Dari nol (butuh dataset besar)
    python train.py --data-dir ../dataset --epochs-head 0 --epochs-finetune 150
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from keras import layers  # noqa: E402

from config import IMAGE_SIZE, LABELS_FILENAME, MODEL_NAME  # noqa: E402
from dataset import compute_class_weights, load_datasets  # noqa: E402
from mobilenetv4 import build_classifier  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--pretrained-weights", default=None,
                   help="File .weights.h5 backbone hasil convert_timm_weights.py")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--epochs-head", type=int, default=10, help="Fase 1: backbone dibekukan")
    p.add_argument("--epochs-finetune", type=int, default=40, help="Fase 2: fine-tune penuh")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-finetune", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cosine_schedule(lr: float, epochs: int, steps_per_epoch: int, warmup_epochs: int = 2):
    total = max(1, epochs * steps_per_epoch)
    warmup = min(total // 2, warmup_epochs * steps_per_epoch)
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr * 0.01, decay_steps=total - warmup, alpha=0.01,
        warmup_target=lr, warmup_steps=warmup)


def compile_model(model, lr, epochs, steps, args):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=cosine_schedule(lr, epochs, steps),
                                         weight_decay=args.weight_decay),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")],
    )


def callbacks(out: Path, phase: str, patience: int, best: float | None = None):
    return [
        keras.callbacks.ModelCheckpoint(out / "best.keras", monitor="val_accuracy",
                                        mode="max", save_best_only=True, verbose=1,
                                        initial_value_threshold=best),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.CSVLogger(out / f"history_{phase}.csv"),
        keras.callbacks.TensorBoard(log_dir=str(out / "tensorboard" / phase)),
    ]


def plot_history(histories, out: Path):
    merged: dict[str, list] = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    if not merged:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(merged.get(key, []), label=f"train_{key}")
        ax.plot(merged.get(f"val_{key}", []), label=f"val_{key}")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=120)


def main():
    args = parse_args()
    keras.utils.set_random_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, counts = load_datasets(
        args.data_dir, args.batch_size, args.val_split, args.seed)
    print("Kelas:", class_names)
    print("Jumlah data train per kelas:", dict(zip(class_names, counts.tolist())))
    (out / LABELS_FILENAME).write_text("\n".join(class_names) + "\n", encoding="utf-8")

    class_weight = None if args.no_class_weight else compute_class_weights(counts)
    steps = int(train_ds.cardinality().numpy())

    model, backbone = build_classifier(len(class_names), dropout=args.dropout)
    histories = []

    if args.pretrained_weights:
        backbone.load_weights(args.pretrained_weights)
        print(f"Bobot pretrained dimuat dari {args.pretrained_weights}")

        if args.epochs_head > 0:
            backbone.trainable = False
            compile_model(model, args.lr_head, args.epochs_head, steps, args)
            print("\n=== Fase 1: training classifier head (backbone beku) ===")
            histories.append(model.fit(train_ds, validation_data=val_ds,
                                       epochs=args.epochs_head, class_weight=class_weight,
                                       callbacks=callbacks(out, "head", args.patience)))

        # BatchNorm tetap beku (mode inferensi) -> stabil untuk dataset kecil.
        backbone.trainable = True
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    elif args.epochs_head > 0:
        print("Tanpa bobot pretrained: fase 1 dilewati, seluruh model dilatih dari nol.")

    if args.epochs_finetune > 0:
        compile_model(model, args.lr_finetune if args.pretrained_weights else args.lr_head,
                      args.epochs_finetune, steps, args)
        print("\n=== Fase 2: fine-tuning seluruh backbone ===")
        best = max(histories[-1].history["val_accuracy"]) if histories else None
        histories.append(model.fit(train_ds, validation_data=val_ds,
                                   epochs=args.epochs_finetune, class_weight=class_weight,
                                   callbacks=callbacks(out, "finetune", args.patience, best)))

    model.save(out / "final.keras")
    plot_history(histories, out)

    def evaluate(ds):
        return {k: float(v) for k, v in model.evaluate(ds, verbose=0, return_dict=True).items()}

    results = {"val": evaluate(val_ds)}
    if test_ds is not None:
        results["test"] = evaluate(test_ds)
    print("Hasil:", json.dumps(results, indent=2))

    meta = {
        "model": MODEL_NAME,
        "image_size": IMAGE_SIZE,
        "input": "RGB float32 [0,255], NHWC, center-crop 1:1 lalu resize bilinear",
        "class_names": class_names,
        "train_counts": counts.tolist(),
        "args": vars(args),
        "results": results,
    }
    (out / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSelesai. Model terbaik: {out / 'best.keras'}")


if __name__ == "__main__":
    main()
