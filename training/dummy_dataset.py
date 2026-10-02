import numpy as np
from pathlib import Path
from PIL import Image

rng = np.random.default_rng(0)
# Warna dasar beda per kelas agar model benar-benar bisa belajar -> membuktikan pipeline bekerja
classes = {"segar": (230, 150, 140), "kurang_segar": (200, 140, 100), "busuk": (140, 120, 80)}

for split, n in [("train", 30), ("val", 8), ("test", 8)]:
    for cls, rgb in classes.items():
        d = Path("../dataset_dummy") / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = np.clip(np.array(rgb) + rng.normal(0, 25, (300, 400, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(d / f"{cls}_{i:03d}.jpg")