"""
evaluate.py — Evaluate trained YOLOv8m model on wildfire smoke/fire dataset.

Usage:
    python src/evaluate.py [--mock]

--mock: Evaluate on mock dataset using baseline weights.
Normal: Evaluate on processed dataset using baseline weights.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
from ultralytics import YOLO


try:
    from src.utils.paths import (
        BASELINE_WEIGHTS,
        DATASET_YAML,
        MOCK_DATASET_YAML,
        RESULTS_DIR,
        BASELINE_CONFIG,
        BEST_THRESHOLD_JSON,
    )
except ImportError:
    BASELINE_WEIGHTS = Path("outputs/checkpoints/baseline/weights/best.pt")
    DATASET_YAML = Path("data/processed/dataset.yaml")
    MOCK_DATASET_YAML = Path("data/mock/dataset.yaml")
    RESULTS_DIR = Path("outputs/results")
    BASELINE_CONFIG = Path("configs/baseline.yaml")
    BEST_THRESHOLD_JSON = Path("outputs/results/best_threshold.json")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_device() -> str:
    try:
        cfg = load_config(BASELINE_CONFIG)
        return cfg.get("device", "cpu")
    except FileNotFoundError:
        return "cpu"


def load_best_thresholds() -> Tuple[float, float]:
    """Return (conf, iou) from the threshold sweep, or safe defaults."""
    try:
        data = json.loads(BEST_THRESHOLD_JSON.read_text())
        return float(data["conf"]), float(data["iou"])
    except (FileNotFoundError, KeyError, ValueError):
        return 0.25, 0.45


def extract_metrics(results) -> Dict[str, Any]:
    """Extract metrics from YOLO Results object."""
    box = results.box
    metrics = {
        "overall": {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
        },
        "per_class": {},
    }

    try:
        class_names = list(results.names.values())
    except AttributeError:
        class_names = ["fire", "smoke"]
    for i, name in enumerate(class_names):
        try:
            class_result = box.class_result(i)
            metrics["per_class"][name] = {
                "precision": round(float(class_result[0]), 4),
                "recall": round(float(class_result[1]), 4),
                "mAP50": round(float(class_result[2]), 4),
                "mAP50-95": round(float(class_result[3]), 4),
            }
        except (IndexError, AttributeError):
            metrics["per_class"][name] = {
                "precision": 0.0,
                "recall": 0.0,
                "mAP50": 0.0,
                "mAP50-95": 0.0,
            }

    return metrics


def print_summary(metrics: Dict[str, Any], mock: bool) -> None:
    """Print clean summary to terminal."""
    mode = "Mock" if mock else "Full"
    print(f"\n{'='*50}")
    print(f"{mode} Evaluation Results")
    print(f"{'='*50}")

    overall = metrics["overall"]
    print("Overall Metrics:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  mAP@50:    {overall['mAP50']:.4f}")
    print(f"  mAP@50-95: {overall['mAP50-95']:.4f}")

    print("\nPer-Class Metrics:")
    for cls, vals in metrics["per_class"].items():
        print(f"  {cls.capitalize()}:")
        print(f"    Precision: {vals['precision']:.4f}")
        print(f"    Recall:    {vals['recall']:.4f}")
        print(f"    mAP@50:    {vals['mAP50']:.4f}")
        print(f"    mAP@50-95: {vals['mAP50-95']:.4f}")

    print(f"{'='*50}")


def evaluate_model(weights_path: Path, data_yaml: Path, mock: bool) -> Dict[str, Any]:
    """Run YOLO evaluation and return metrics."""
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found at {weights_path}. "
            "Run training first: make train-baseline or make train-mock"
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found at {data_yaml}. "
            "Run data preparation: python src/data/prepare_dataset.py"
        )

    device = get_device()
    conf, iou = load_best_thresholds()
    print(f"Evaluating on device: {device}")
    print(f"Weights: {weights_path}")
    print(f"Dataset: {data_yaml}")
    print(f"Thresholds: conf={conf} iou={iou}")

    model = YOLO(str(weights_path))
    results = model.val(
        data=str(data_yaml),
        device=device,
        conf=conf,
        iou=iou,
        workers=0,
        verbose=False,
    )

    return extract_metrics(results)


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save metrics to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8m model")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Evaluate on mock dataset"
    )
    args = parser.parse_args()

    mock = args.mock
    weights_path = BASELINE_WEIGHTS
    data_yaml = MOCK_DATASET_YAML if mock else DATASET_YAML
    output_json = RESULTS_DIR / ("mock_evaluation_metrics.json" if mock else "evaluation_metrics.json")

    try:
        metrics = evaluate_model(weights_path, data_yaml, mock)
        save_metrics(metrics, output_json)
        print_summary(metrics, mock)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
