"""Porting bobot ImageNet MobileNetV4-Conv-Small dari timm (PyTorch) ke Keras.

Keras tidak menyediakan bobot pretrained MNV4, sedangkan Google merilis checkpoint
resmi lewat timm (`mobilenetv4_conv_small.e2400_r224_in1k`). Script ini menyalin
bobot tersebut ke backbone Keras lalu memverifikasi output fiturnya identik.

    pip install -r requirements-convert.txt
    python convert_timm_weights.py --output weights/mnv4_conv_small_imagenet.weights.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from keras import layers

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from mobilenetv4 import build_backbone

TIMM_NAME = "mobilenetv4_conv_small.e2400_r224_in1k"


def port(backbone, timm_map, state_dict):
    used = set()

    def take(key):
        if key not in state_dict:
            raise KeyError(f"Parameter timm '{key}' tidak ditemukan")
        used.add(key)
        return state_dict[key].detach().cpu().numpy()

    for layer in backbone.layers:
        prefix = timm_map.get(layer.name)
        if prefix is None:
            continue
        if isinstance(layer, layers.DepthwiseConv2D):
            # torch (C, 1, kH, kW) -> keras (kH, kW, C, 1)
            layer.set_weights([take(f"{prefix}.weight").transpose(2, 3, 0, 1)])
        elif isinstance(layer, layers.Conv2D):
            # torch (O, I, kH, kW) -> keras (kH, kW, I, O)
            layer.set_weights([take(f"{prefix}.weight").transpose(2, 3, 1, 0)])
        elif isinstance(layer, layers.BatchNormalization):
            layer.set_weights([take(f"{prefix}.{n}") for n in
                               ("weight", "bias", "running_mean", "running_var")])

    leftovers = [k for k in state_dict
                 if k not in used and not k.endswith("num_batches_tracked")
                 and not k.startswith("classifier")]
    if leftovers:
        raise RuntimeError(f"Parameter timm tidak terpakai (arsitektur tidak cocok): {leftovers}")
    print(f"{len(used)} tensor berhasil diporting.")


def verify(backbone, tmodel):
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, size=(2, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)
    norm = (img / 255.0 - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)

    keras_feat = np.asarray(backbone(norm, training=False))
    with torch.no_grad():
        t_in = torch.from_numpy(norm.transpose(0, 3, 1, 2).copy())
        torch_feat = tmodel.forward_head(tmodel.forward_features(t_in), pre_logits=True).numpy()

    diff = np.abs(keras_feat - torch_feat).max()
    print(f"Max abs diff fitur Keras vs timm: {diff:.6f}")
    if diff > 1e-3:
        raise RuntimeError("Verifikasi gagal: output tidak identik.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="weights/mnv4_conv_small_imagenet.weights.h5")
    args = p.parse_args()

    tmodel = timm.create_model(TIMM_NAME, pretrained=True).eval()
    backbone, timm_map = build_backbone()
    port(backbone, timm_map, tmodel.state_dict())
    verify(backbone, tmodel)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    backbone.save_weights(out)
    print(f"Bobot backbone disimpan ke: {out}")


if __name__ == "__main__":
    main()
