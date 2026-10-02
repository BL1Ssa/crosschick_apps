"""Konstanta bersama untuk training, export, dan inference."""

MODEL_NAME = "mobilenetv4_conv_small"
IMAGE_SIZE = 224  # R224
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Normalisasi ImageNet (sama dengan bobot timm `mobilenetv4_conv_small.e2400_r224_in1k`).
# Normalisasi ditanam di dalam model, sehingga input model = RGB float32 0..255.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TFLITE_BASENAME = "freshness_mnv4s_r224"
LABELS_FILENAME = "labels.txt"
