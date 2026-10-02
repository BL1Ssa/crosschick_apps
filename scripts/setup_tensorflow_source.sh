#!/usr/bin/env bash
# Clone source TensorFlow (hanya untuk membangun TensorFlow Lite C++).
# Tag HARUS sama dengan versi tensorflow di training/requirements.txt.
set -euo pipefail

TAG="${1:-v2.17.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/tensorflow"

STB_DIR="$ROOT/third_party/stb"
mkdir -p "$(dirname "$DEST")"

# stb_image.h: dipakai CLI desktop classify_image
if [[ ! -f "$STB_DIR/stb_image.h" ]]; then
  git clone --depth 1 https://github.com/nothings/stb.git "$STB_DIR"
fi

if [[ -f "$DEST/tensorflow/lite/CMakeLists.txt" ]]; then
  echo "Source TensorFlow sudah ada di $DEST"
  exit 0
fi

git clone --depth 1 --branch "$TAG" https://github.com/tensorflow/tensorflow.git "$DEST"
echo "Selesai: $DEST ($TAG)"
