"""Fetch and verify the model files once. Inference reads them from disk."""

import hashlib
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

MODEL_DIR = Path(__file__).resolve().parent / ".models"
VEHICLE = "yolo11s.pt"
CLASSIFIER = "open_clip_model.safetensors"
PLATE = "yolo-v9-t-640-license-plates-end2end.onnx"
OCR = "cct_xs_v2_global.onnx"
OCR_CONFIG = "cct_xs_v2_global_plate_config.yaml"

OCR_RELEASE = "https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates"
FILES = (
    (VEHICLE, f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{VEHICLE}",
     "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"),
    (CLASSIFIER, "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/resolve/"
     f"a6f597a30f7b82c51704746581f9a4e41421e878/{CLASSIFIER}",
     "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"),
    (PLATE, f"https://github.com/ankandrew/open-image-models/releases/download/assets/{PLATE}",
     "c3c1026ca7d0585dd88084d68182dd897113712fa734ae1557ca70174440c076"),
    (OCR, f"{OCR_RELEASE}/{OCR}",
     "8031afb5fdc6b4d80462c9d542f1284ebd2cfddf5dbacd62609848d7e2855f44"),
    (OCR_CONFIG, f"{OCR_RELEASE}/{OCR_CONFIG}",
     "0335c74a305173bb6f393efed0fde03cadeaa0b649ed8e19f431016d8232d0a6"),
)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    for name, url, expected in FILES:
        target = MODEL_DIR / name
        if target.is_file() and digest(target) == expected:
            continue
        print(f"Downloading {name} ...", flush=True)
        with NamedTemporaryFile(dir=MODEL_DIR, delete=False) as temp:
            partial = Path(temp.name)
        try:
            with urlopen(url, timeout=90) as source, partial.open("wb") as out:
                shutil.copyfileobj(source, out)
            if digest(partial) != expected:
                raise ValueError(f"Checksum mismatch for {name}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
    print("Models ready. Run python demo.py")


if __name__ == "__main__":
    main()
