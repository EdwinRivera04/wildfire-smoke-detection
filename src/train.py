"""
train.py — Train YOLOv8m on the wildfire smoke/fire dataset.

Usage:
    python src/train.py --config configs/baseline.yaml [--mock]

--mock: 1 epoch, batch=2, imgsz=64, mock dataset. Completes in <60s.
"""

import argparse
import json
import time
from pathlib import Path

import yaml

try:
    from src.utils.paths import (
        BEST_THRESHOLD_JSON,
        CHECKPOINTS_DIR,
        MOCK_DATASET_YAML,
        RESULTS_DIR,
    )
except ModuleNotFoundError:
    from utils.paths import (
        BEST_THRESHOLD_JSON,
        CHECKPOINTS_DIR,
        MOCK_DATASET_YAML,
        RESULTS_DIR,
    )


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_mock_overrides(cfg: dict) -> dict:
    import tempfile

    cfg = cfg.copy()
    cfg["epochs"] = 1
    cfg["batch"] = 2
    cfg["imgsz"] = 64
    cfg["cache"] = False
    cfg["fraction"] = 1.0

    mock_abs = MOCK_DATASET_YAML.resolve()
    mock_cfg = yaml.safe_load(mock_abs.read_text())
    mock_cfg["path"] = str(mock_abs.parent.resolve())

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(mock_cfg, tmp)
    tmp.flush()
    cfg["data"] = tmp.name
    return cfg


def threshold_sweep(model, cfg: dict) -> None:
    """Grid search over conf/iou thresholds and save best F1 to JSON."""
    conf_vals = [0.25, 0.35, 0.45]
    iou_vals = [0.40, 0.50, 0.60]
    best = {"conf": None, "iou": None, "f1": 0.0}

    print("\nRunning threshold sweep...")
    for conf in conf_vals:
        for iou in iou_vals:
            results = model.val(
                data=cfg.get("data"),
                conf=conf,
                iou=iou,
                device=cfg.get("device", "mps"),
                workers=0,
                verbose=False,
            )
            p = results.box.mp
            r = results.box.mr
            f1 = 2 * p * r / (p + r + 1e-9)
            print(f"  conf={conf} iou={iou} → P={p:.3f} R={r:.3f} F1={f1:.3f}")

            if f1 > best["f1"]:
                best = {
                    "conf": conf,
                    "iou": iou,
                    "f1": round(f1, 4),
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BEST_THRESHOLD_JSON, "w") as f:
        json.dump(best, f, indent=2)

    print(f"Best threshold → conf={best['conf']} iou={best['iou']} F1={best['f1']}")
    print(f"Saved: {BEST_THRESHOLD_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mock", action="store_true", help="Quick smoke-test run")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mock:
        cfg = apply_mock_overrides(cfg)

    if cfg.get("augment"):
        try:
            from src.data.augment import get_augmentation_pipeline
        except ModuleNotFoundError:
            from data.augment import get_augmentation_pipeline

        try:
            _ = get_augmentation_pipeline()
            print("Augmentation pipeline loaded.")
        except ImportError:
            print("augment.py not available — proceeding without custom augmentation.")

    model_weights = cfg.pop("model", "yolov8m.pt")
    cfg.pop("augment", None)
    cfg.pop("fl_gamma", None)

    from ultralytics import YOLO

    model = YOLO(model_weights)

    t_start = time.time()

    def on_train_epoch_end(trainer) -> None:
        if trainer.epoch == 0:
            elapsed = time.time() - t_start
            est_total = elapsed * trainer.epochs
            print(
                f"\n⏱  Epoch 1: {elapsed:.0f}s | "
                f"Est. total: {est_total / 60:.1f}m | "
                f"Budget: 35-40 min"
            )

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    t0 = time.time()
    results = model.train(**cfg)
    total_time = time.time() - t0

    project = Path(cfg.get("project", str(CHECKPOINTS_DIR)))
    name = cfg.get("name", "baseline")

    checkpoint = Path("runs/detect") / project / name / "weights" / "best.pt"
    if not checkpoint.exists():
        checkpoint = project / name / "weights" / "best.pt"

    print(f"\n{'=' * 50}")
    print(f"Training complete: {total_time / 60:.1f} min")
    print(f"Checkpoint: {checkpoint}")
    if hasattr(results, "box"):
        print(f"mAP@50: {results.box.map50:.4f}")
    print(f"{'=' * 50}")

    try:
        if checkpoint.exists():
            target_dir = project / name / "weights"
            target_dir.mkdir(parents=True, exist_ok=True)

            target_path = target_dir / "best.pt"
            if checkpoint.resolve() != target_path.resolve():
                import shutil

                shutil.copy2(checkpoint, target_path)
                print(f"Copied checkpoint to: {target_path}")
    except Exception as e:
        print(f"Warning: could not copy checkpoint → {e}")

    if "improved" in name:
        threshold_sweep(model, cfg)


if __name__ == "__main__":
    main()