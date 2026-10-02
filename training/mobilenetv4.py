"""MobileNetV4-Conv-Small (Qin et al., 2024) untuk Keras 3 / TensorFlow.

Arsitektur mengikuti spesifikasi resmi `MNV4ConvSmall` dan implementasi timm
`mobilenetv4_conv_small`, sehingga bobot ImageNet dari timm dapat diporting
1:1 (lihat convert_timm_weights.py).

Stem (112px) -> stage0 (56px) -> stage1 (28px) -> stage2 (14px) -> stage3 (7px)
-> Conv1x1 960 -> GAP -> Conv1x1 1280 -> classifier.
"""

from __future__ import annotations

import keras
from keras import layers

from config import IMAGENET_MEAN, IMAGENET_STD, INPUT_SHAPE

BN_MOMENTUM = 0.99
BN_EPSILON = 1e-5  # sama dengan PyTorch/timm agar bobot porting identik


def make_divisible(v: float, divisor: int = 8, round_limit: float = 0.9) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


# ("cn", kernel, stride, out_ch)                         -> Conv-BN-ReLU
# ("uib", start_dw_k, mid_dw_k, stride, expand, out_ch)  -> Universal Inverted Bottleneck
#   start=0,mid>0 : Inverted Bottleneck   | start>0,mid>0 : ExtraDW
#   start>0,mid=0 : ConvNext-like         | start=0,mid=0 : FFN
MNV4_CONV_SMALL_STAGES = [
    [("cn", 3, 2, 32), ("cn", 1, 1, 32)],
    [("cn", 3, 2, 96), ("cn", 1, 1, 64)],
    [("uib", 5, 5, 2, 3.0, 96)]
    + [("uib", 0, 3, 1, 2.0, 96)] * 4
    + [("uib", 3, 0, 1, 4.0, 96)],
    [
        ("uib", 3, 3, 2, 6.0, 128),
        ("uib", 5, 5, 1, 4.0, 128),
        ("uib", 0, 5, 1, 4.0, 128),
        ("uib", 0, 5, 1, 3.0, 128),
        ("uib", 0, 3, 1, 4.0, 128),
        ("uib", 0, 3, 1, 4.0, 128),
    ],
    [("cn", 1, 1, 960)],
]
STEM_CHANNELS = 32
HEAD_CHANNELS = 1280


class _Builder:
    def __init__(self):
        # nama layer keras -> prefix parameter timm (untuk porting bobot)
        self.timm_map: dict[str, str] = {}

    def conv_bn(self, x, filters, kernel, stride, name, timm_conv, timm_bn,
                depthwise=False, act=True):
        if stride > 1 and kernel > 1:
            # Padding simetris (gaya PyTorch) agar hasil identik dengan bobot timm.
            x = layers.ZeroPadding2D(kernel // 2, name=f"{name}_pad")(x)
            padding = "valid"
        else:
            padding = "same"

        if depthwise:
            conv = layers.DepthwiseConv2D(kernel, strides=stride, padding=padding,
                                          use_bias=False, name=f"{name}_dw")
        else:
            conv = layers.Conv2D(filters, kernel, strides=stride, padding=padding,
                                 use_bias=False, name=f"{name}_conv")
        x = conv(x)
        bn = layers.BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                       name=f"{name}_bn")
        x = bn(x)
        if act:
            x = layers.ReLU(name=f"{name}_relu")(x)

        self.timm_map[conv.name] = timm_conv
        self.timm_map[bn.name] = timm_bn
        return x

    def uib(self, x, in_ch, out_ch, start_k, mid_k, stride, expand, name, prefix):
        shortcut = x
        if start_k:
            x = self.conv_bn(x, in_ch, start_k, 1 if mid_k else stride, f"{name}_start",
                             f"{prefix}.dw_start.conv", f"{prefix}.dw_start.bn",
                             depthwise=True, act=False)
        mid_ch = make_divisible(in_ch * expand)
        x = self.conv_bn(x, mid_ch, 1, 1, f"{name}_expand",
                         f"{prefix}.pw_exp.conv", f"{prefix}.pw_exp.bn")
        if mid_k:
            x = self.conv_bn(x, mid_ch, mid_k, stride, f"{name}_mid",
                             f"{prefix}.dw_mid.conv", f"{prefix}.dw_mid.bn",
                             depthwise=True)
        x = self.conv_bn(x, out_ch, 1, 1, f"{name}_proj",
                         f"{prefix}.pw_proj.conv", f"{prefix}.pw_proj.bn", act=False)
        if stride == 1 and in_ch == out_ch:
            x = layers.Add(name=f"{name}_add")([shortcut, x])
        return x


def build_backbone(input_shape=INPUT_SHAPE):
    """Backbone MNV4-Conv-Small. Input: tensor yang sudah dinormalisasi ImageNet.

    Return: (keras.Model dengan output fitur 1280-d, timm_map)
    """
    b = _Builder()
    inputs = keras.Input(shape=input_shape, name="normalized_image")
    x = b.conv_bn(inputs, STEM_CHANNELS, 3, 2, "stem", "conv_stem", "bn1")

    in_ch = STEM_CHANNELS
    for s, stage in enumerate(MNV4_CONV_SMALL_STAGES):
        for i, spec in enumerate(stage):
            name, prefix = f"s{s}_b{i}", f"blocks.{s}.{i}"
            if spec[0] == "cn":
                _, k, stride, out_ch = spec
                x = b.conv_bn(x, out_ch, k, stride, name, f"{prefix}.conv", f"{prefix}.bn1")
            else:
                _, start_k, mid_k, stride, expand, out_ch = spec
                x = b.uib(x, in_ch, out_ch, start_k, mid_k, stride, expand, name, prefix)
            in_ch = out_ch

    x = layers.GlobalAveragePooling2D(keepdims=True, name="gap")(x)
    x = b.conv_bn(x, HEAD_CHANNELS, 1, 1, "head", "conv_head", "norm_head")
    x = layers.Flatten(name="features")(x)
    return keras.Model(inputs, x, name="mnv4_conv_small_backbone"), b.timm_map


def build_classifier(num_classes: int, input_shape=INPUT_SHAPE, dropout: float = 0.2):
    """Model lengkap: RGB float32 [0..255] -> softmax(num_classes).

    Preprocessing (rescale + normalisasi ImageNet) ada di dalam graph sehingga
    aplikasi C++/Kotlin cukup mengirim pixel RGB mentah.
    """
    backbone, _ = build_backbone(input_shape)
    inputs = keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = layers.Normalization(mean=list(IMAGENET_MEAN),
                             variance=[s ** 2 for s in IMAGENET_STD],
                             name="imagenet_norm")(x)
    x = backbone(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32",
                           name="predictions")(x)
    return keras.Model(inputs, outputs, name="mnv4_conv_small_freshness"), backbone


if __name__ == "__main__":
    m, bb = build_classifier(3)
    bb.summary()
    print(f"Backbone params: {bb.count_params():,}")
    print(f"Total params   : {m.count_params():,}")
