"""Konversi model Keras (.keras) ke TensorFlow Lite (.tflite) untuk C++/Android.

    python export_tflite.py --model outputs/best.keras --quant fp16
    python export_tflite.py --model outputs/best.keras --quant int8 --data-dir ../dataset/train

Mode kuantisasi:
    none    : float32 (paling akurat, ~10 MB)
    fp16    : bobot float16 (~5 MB, akurasi hampir sama)       <- default
    dynamic : bobot int8, aktivasi float (~3 MB)
    int8    : full integer, input/output uint8 (~3 MB, tercepat di CPU/NPU)
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from config import IMAGE_SIZE, LABELS_FILENAME, TFLITE_BASENAME
from dataset import load_eval_dataset


def representative_dataset(data_dir: str, num_samples: int):
    ds, _ = load_eval_dataset(data_dir, batch_size=1)
    ds = ds.unbatch().shuffle(1000, seed=0).take(num_samples).batch(1)

    def gen():
        for images, _ in ds:
            yield [tf.cast(images, tf.float32)]
    return gen


def convert(model_path: str, quant: str, data_dir: str | None, num_samples: int):
    """Return (tflite_bytes, keras_model)."""
    model = keras.models.load_model(model_path, compile=False)

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = str(Path(tmp) / "saved_model")
        # Batch statis = 1 agar graph TFLite sederhana di perangkat mobile.
        export = keras.export.ExportArchive()
        export.track(model)
        export.add_endpoint(
            name="serve",
            fn=lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec([1, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32, name="image")],
        )
        export.write_out(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        if quant == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif quant == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quant == "int8":
            if not data_dir:
                raise SystemExit("--data-dir wajib untuk kuantisasi int8 (representative dataset)")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset(data_dir, num_samples)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
        return converter.convert(), model


def tflite_predict(tflite_bytes: bytes, images: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    preds = []
    for img in images:
        x = img[None].astype(np.float32)
        if inp["dtype"] != np.float32:
            scale, zp = inp["quantization"]
            x = np.clip(np.round(x / scale + zp), 0, 255).astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        y = interp.get_tensor(out["index"])[0]
        if out["dtype"] != np.float32:
            scale, zp = out["quantization"]
            y = (y.astype(np.float32) - zp) * scale
        preds.append(y)
    return np.stack(preds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--quant", choices=["none", "fp16", "dynamic", "int8"], default="fp16")
    p.add_argument("--data-dir", default=None, help="Folder berisi subfolder per kelas")
    p.add_argument("--num-calib", type=int, default=300)
    p.add_argument("--output-dir", default="exported")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tflite_bytes, model = convert(args.model, args.quant, args.data_dir, args.num_calib)
    suffix = "" if args.quant == "none" else f"_{args.quant}"
    tflite_path = out_dir / f"{TFLITE_BASENAME}{suffix}.tflite"
    tflite_path.write_bytes(tflite_bytes)
    print(f"TFLite disimpan: {tflite_path} ({len(tflite_bytes) / 1e6:.2f} MB)")

    labels_src = Path(args.model).parent / LABELS_FILENAME
    if labels_src.exists():
        shutil.copy(labels_src, out_dir / LABELS_FILENAME)

    # Sanity check: bandingkan output Keras vs TFLite.
    if args.data_dir:
        ds, _ = load_eval_dataset(args.data_dir, batch_size=16)
        images = next(iter(ds))[0].numpy()
    else:
        images = np.random.default_rng(0).uniform(0, 255, (4, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)
    k = model.predict(images, verbose=0)
    t = tflite_predict(tflite_bytes, images)
    agree = (k.argmax(1) == t.argmax(1)).mean() * 100
    print(f"Max abs diff probabilitas Keras vs TFLite: {np.abs(k - t).max():.4f} | "
          f"kesesuaian argmax: {agree:.1f}%")


if __name__ == "__main__":
    main()
