"""Run the recognition steps from the camera on one local still image."""

import argparse
import os
from pathlib import Path

from download_models import CLASSIFIER, FILES, MODEL_DIR, OCR, OCR_CONFIG, PLATE, VEHICLE

EXAMPLE = Path(__file__).resolve().parent / "examples" / "studio-cruiser.jpg"
POLICE = [
    "a photo of a police car with emergency lights",
    "a police cruiser",
    "a police patrol vehicle with law enforcement markings",
    "a sheriff's car",
    "a police SUV",
]
OTHER = [
    "a photo of a regular civilian car",
    "an ordinary passenger car on the street",
    "a taxi cab",
    "a delivery van",
    "a pickup truck",
    "a family SUV",
    "an ambulance",
    "a fire truck",
    "a tow truck",
    "a white van with company decals",
    "a black SUV with tinted windows",
    "a car with a roof rack",
    "a city bus",
]


def largest_vehicle(photo):
    from ultralytics import YOLO

    detector = YOLO(MODEL_DIR / VEHICLE)
    result = detector.predict(photo, classes=[2, 5, 7], imgsz=640, conf=0.25,
                              device="cpu", save=False, verbose=False)[0]
    if not len(result.boxes):
        return None
    largest = result.boxes.xywh[:, 2:].prod(dim=1).argmax().item()
    box = result.boxes[largest]
    return result.names[int(box.cls.item())], photo.crop(tuple(box.xyxy[0].int().tolist()))


def marked_vehicle_score(crop):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(MODEL_DIR / CLASSIFIER))
    model.eval()
    tokens = open_clip.get_tokenizer("ViT-B-32")(POLICE + OTHER)
    with torch.inference_mode():
        text = model.encode_text(tokens, normalize=True)
        image = model.encode_image(preprocess(crop).unsqueeze(0), normalize=True)
        scores = (100 * image @ text.T).softmax(dim=-1)
    return scores[0, :len(POLICE)].sum().item()


def plate_crop(vehicle):
    import cv2
    import numpy as np
    from open_image_models import create_detector

    bgr = cv2.cvtColor(np.asarray(vehicle), cv2.COLOR_RGB2BGR)
    detector = create_detector(MODEL_DIR / PLATE, backend="yolo_v9",
                               class_labels=["plate"], conf_thresh=0.3,
                               providers=["CPUExecutionProvider"])
    candidates = detector.predict(bgr)
    if not candidates:
        return None
    box = max(candidates, key=lambda d: d.confidence).bounding_box
    x1, y1, x2, y2 = map(int, (box.x1, box.y1, box.x2, box.y2))
    pad_x = int((x2 - x1) * 0.08) + 4
    pad_y = int((y2 - y1) * 0.15) + 4
    patch = bgr[max(0, y1-pad_y):min(vehicle.height, y2+pad_y),
                max(0, x1-pad_x):min(vehicle.width, x2+pad_x)]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) if patch.size else None


def run(photo, read_demo_plate):
    vehicle = largest_vehicle(photo)
    if vehicle is None:
        print("No car, bus or truck detected.")
        return
    kind, crop = vehicle
    print(f"Vehicle: {kind}")
    score = marked_vehicle_score(crop)
    print(f"Marked-vehicle appearance score: {score:.3f}")
    if score < 0.90:
        print("Below the 0.90 threshold. Plate step skipped.")
        return
    patch = plate_crop(crop)
    if patch is None:
        print("No plate found.")
        return
    print("Plate region found.")
    if read_demo_plate:
        from fast_plate_ocr import LicensePlateRecognizer

        reader = LicensePlateRecognizer(onnx_model_path=MODEL_DIR / OCR,
                                        plate_config_path=MODEL_DIR / OCR_CONFIG,
                                        providers=["CPUExecutionProvider"])
        text = reader.run_one(patch).plate.strip()
        print("Fictional plate read:", text or "(unreadable)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", nargs="?", type=Path,
                        help="optional staged JPEG or PNG (vehicle and plate detection only)")
    args = parser.parse_args()
    from PIL import Image, ImageOps

    path = args.photo or EXAMPLE
    if not path.is_file():
        parser.error("Choose one local JPEG or PNG file.")
    try:
        with Image.open(path, formats=["JPEG", "PNG"]) as src:
            photo = ImageOps.exif_transpose(src).convert("RGB")
    except (OSError, ValueError) as exc:
        parser.error(f"Could not read the photo: {exc}")

    if not all((MODEL_DIR / name).is_file() for name, _, _ in FILES):
        parser.exit(1, "Model files missing. Run python download_models.py first.\n")
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(MODEL_DIR)
    os.environ["HF_HUB_OFFLINE"] = "1"
    run(photo, read_demo_plate=args.photo is None)


if __name__ == "__main__":
    main()
