"""
prepare_dataset.py — Validate and copy the Kaggle smoke-fire YOLO dataset.

Source layout (data/raw/data/):
    {train,val,test}/images/*.jpg
    {train,val,test}/labels/*.txt   (raw: 0=smoke, 1=fire)

Output layout (data/processed/):
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt   (remapped: 0=fire, 1=smoke)

Usage:
    python src/data/prepare_dataset.py [--mock]

--mock: validates data/mock/ structure only, no file copying.
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import (
    DATA_MOCK,
    DATA_PROCESSED,
    DATA_RAW,
    DATASET_YAML,
    MOCK_DATASET_YAML,
)

# Raw dataset lives one level deeper
RAW_DATA = DATA_RAW / "data"

# Raw class mapping:  0=smoke, 1=fire
# Project mapping:    0=fire,  1=smoke
REMAP = {0: 1, 1: 0}


# ── helpers ──────────────────────────────────────────────────────────────────

def remap_label(src: Path, dst: Path) -> None:
    """Copy label file swapping class IDs to match project convention."""
    lines = src.read_text().strip().splitlines() if src.stat().st_size else []
    remapped = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            parts[0] = str(REMAP.get(int(parts[0]), int(parts[0])))
        remapped.append(" ".join(parts))
    dst.write_text("\n".join(remapped) + ("\n" if remapped else ""))


def get_classes_from_label(lbl_path: Path) -> set:
    """Return set of (remapped) class IDs present in a label file."""
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return set()
    classes = set()
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if parts:
            classes.add(REMAP.get(int(parts[0]), int(parts[0])))
    return classes


def bucket_name(classes: set) -> str:
    has_fire  = 0 in classes
    has_smoke = 1 in classes
    if has_fire and has_smoke:
        return "both"
    if has_fire:
        return "fire_only"
    if has_smoke:
        return "smoke_only"
    return "neither"


# ── mock validation ───────────────────────────────────────────────────────────

def validate_mock():
    for split in ["train", "val", "test"]:
        img_dir = DATA_MOCK / f"images/{split}"
        lbl_dir = DATA_MOCK / f"labels/{split}"
        assert img_dir.exists(), f"Missing: {img_dir}"
        assert lbl_dir.exists(), f"Missing: {lbl_dir}"
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / f"{img.stem}.txt"
            assert lbl.exists(), f"Missing label for {img.name}"

    cfg = yaml.safe_load(MOCK_DATASET_YAML.read_text())
    assert cfg["nc"] == 2
    assert cfg["names"] == ["fire", "smoke"]

    counts = {s: len(list((DATA_MOCK / f"images/{s}").glob("*.jpg")))
              for s in ["train", "val", "test"]}
    assert counts["train"] == 6, f"Expected 6 train images, got {counts['train']}"
    assert counts["val"]   == 2, f"Expected 2 val images, got {counts['val']}"
    assert counts["test"]  == 2, f"Expected 2 test images, got {counts['test']}"

    print("Mock validation passed.")


# ── full pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline():
    assert RAW_DATA.exists(), (
        f"{RAW_DATA} not found.\n"
        "Download the dataset from Kaggle and unzip into data/raw/ so that\n"
        "data/raw/data/train/, data/raw/data/val/, data/raw/data/test/ exist."
    )

    image_exts = {".jpg", ".jpeg", ".png"}
    split_stats: dict[str, dict] = {}

    for split in ["train", "val", "test"]:
        src_img_dir = RAW_DATA / split / "images"
        src_lbl_dir = RAW_DATA / split / "labels"
        dst_img_dir = DATA_PROCESSED / "images" / split
        dst_lbl_dir = DATA_PROCESSED / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = [p for p in sorted(src_img_dir.glob("*"))
                  if p.suffix.lower() in image_exts]

        valid = missing = 0
        fire = smoke = both = neither = 0

        for img in images:
            src_lbl = src_lbl_dir / f"{img.stem}.txt"
            dst_lbl = dst_lbl_dir / f"{img.stem}.txt"

            if not src_lbl.exists():
                missing += 1
                continue

            valid += 1
            shutil.copy2(img, dst_img_dir / img.name)
            remap_label(src_lbl, dst_lbl)

            classes = get_classes_from_label(dst_lbl)
            b = bucket_name(classes)
            if b == "fire_only":   fire   += 1
            elif b == "smoke_only": smoke  += 1
            elif b == "both":       both   += 1
            else:                   neither += 1

        split_stats[split] = dict(total=valid, fire=fire, smoke=smoke,
                                   both=both, neither=neither,
                                   missing=missing)
        print(f"{split}: {len(images)} images | {valid} valid pairs | {missing} missing labels")

    # Write dataset.yaml
    DATASET_YAML.parent.mkdir(parents=True, exist_ok=True)
    dataset_cfg = {
        "path": str(DATA_PROCESSED),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    2,
        "names": ["fire", "smoke"],
    }
    DATASET_YAML.write_text(yaml.dump(dataset_cfg))

    # Print distribution table
    print(f"\n{'Split':<8} {'Total':>7} {'Fire':>6} {'Smoke':>7} {'Both':>6} {'Neither':>9}")
    print("-" * 50)
    for split, s in split_stats.items():
        print(f"{split:<8} {s['total']:>7} {s['fire']:>6} {s['smoke']:>7} "
              f"{s['both']:>6} {s['neither']:>9}")

    print(f"\nDataset written to {DATA_PROCESSED}")
    print("Class mapping applied: raw 0=smoke→1, raw 1=fire→0 (project: 0=fire, 1=smoke)")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if args.mock:
        validate_mock()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
