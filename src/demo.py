import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

BEST_THRESHOLD_JSON = Path("outputs/results/best_threshold.json")


def load_best_thresholds():
    try:
        data = json.loads(BEST_THRESHOLD_JSON.read_text())
        return float(data["conf"]), float(data["iou"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0.25, 0.45


def get_weights_path() -> Path:
    candidates = [
        Path("runs/detect/outputs/checkpoints/baseline/weights/best.pt"),
        Path("outputs/checkpoints/baseline/weights/best.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find trained weights. Run `make train-mock` or `make train-baseline` first."
    )


def get_image_dir(mock: bool) -> Path:
    if mock:
        return Path("data/mock/images/val")
    return Path("data/processed/images/val")


def collect_images(image_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(sorted(image_dir.glob(pattern)))
    return image_paths


def run_demo(mock: bool) -> None:
    weights_path = get_weights_path()
    image_dir = get_image_dir(mock)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = collect_images(image_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    output_dir = Path("outputs/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using weights: {weights_path}")
    print(f"Looking for images in: {image_dir}")
    print(f"Found {len(image_paths)} images")
    print(f"Saving demo outputs to: {output_dir}")

    conf, iou = load_best_thresholds()
    print(f"Using thresholds: conf={conf} iou={iou}")

    model = YOLO(str(weights_path))

    for img_path in image_paths[:5]:
        print(f"\nProcessing: {img_path.name}")
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            save=False,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is not None and len(boxes):
            names = results[0].names
            detections = [
                f"{names[int(cls)]} {float(score):.2f}"
                for cls, score in zip(boxes.cls, boxes.conf)
            ]
            print(f"  Detected: {', '.join(detections)}")
        else:
            print("  No detections.")

        annotated = results[0].plot()
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), annotated)

    print("\nDemo complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO demo inference on validation images.")
    parser.add_argument("--mock", action="store_true", help="Run demo on mock dataset")
    args = parser.parse_args()

    try:
        run_demo(mock=args.mock)
    except Exception as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
