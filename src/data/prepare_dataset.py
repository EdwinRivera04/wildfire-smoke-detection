"""
prepare_dataset.py — Download D-Fire, validate, and split into train/val/test.

Usage:
    python src/data/prepare_dataset.py [--mock]

--mock: validates data/mock/ structure only, no download.
"""

import argparse
import shutil
import subprocess
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

DFIRE_REPO = "https://github.com/gaiasd/DFireDataset"
SEED = 42


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

    train_count = len(list((DATA_MOCK / "images/train").glob("*.jpg")))
    val_count   = len(list((DATA_MOCK / "images/val").glob("*.jpg")))
    test_count  = len(list((DATA_MOCK / "images/test").glob("*.jpg")))
    assert train_count == 6, f"Expected 6 train images, got {train_count}"
    assert val_count   == 2, f"Expected 2 val images, got {val_count}"
    assert test_count  == 2, f"Expected 2 test images, got {test_count}"

    print("Mock validation passed.")


def get_image_classes(label_path: Path) -> set:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return set()
    classes = set()
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if parts:
            classes.add(int(parts[0]))
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


def split_list(items, train_ratio=0.8, seed=SEED):
    import random
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    n_val   = (len(shuffled) - n_train) // 2
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


def run_full_pipeline():
    # 1. Download
    if not DATA_RAW.exists() or not any(DATA_RAW.iterdir()):
        print("Cloning D-Fire dataset...")
        subprocess.run(
            ["git", "clone", DFIRE_REPO, str(DATA_RAW)],
            check=True,
        )
    else:
        print(f"D-Fire already present at {DATA_RAW}, skipping clone.")

    # 2. Discover images and validate pairs
    image_exts = {".jpg", ".jpeg", ".png"}
    all_images = [
        p for p in DATA_RAW.rglob("*")
        if p.suffix.lower() in image_exts
    ]
    valid_pairs = []
    missing_labels = []
    for img in all_images:
        lbl = img.with_suffix(".txt")
        if lbl.exists():
            valid_pairs.append(img)
        else:
            missing_labels.append(img)

    print(
        f"{len(all_images)} images | "
        f"{len(valid_pairs)} valid pairs | "
        f"{len(missing_labels)} missing labels"
    )

    # 3. Stratified split by class presence
    buckets: dict[str, list] = {"fire_only": [], "smoke_only": [], "both": [], "neither": []}
    for img in valid_pairs:
        lbl = img.with_suffix(".txt")
        classes = get_image_classes(lbl)
        buckets[bucket_name(classes)].append(img)

    train_imgs, val_imgs, test_imgs = [], [], []
    for name, items in buckets.items():
        tr, va, te = split_list(items, seed=SEED)
        train_imgs.extend(tr)
        val_imgs.extend(va)
        test_imgs.extend(te)

    # 4. Copy to processed/
    split_map = {"train": train_imgs, "val": val_imgs, "test": test_imgs}
    for split, imgs in split_map.items():
        (DATA_PROCESSED / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (DATA_PROCESSED / f"labels/{split}").mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, DATA_PROCESSED / f"images/{split}" / img.name)
            lbl = img.with_suffix(".txt")
            shutil.copy2(lbl, DATA_PROCESSED / f"labels/{split}" / lbl.name)

    # 5. Write dataset.yaml
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

    # 6. Print distribution table
    print(f"\n{'Split':<8} {'Total':>6} {'Fire':>6} {'Smoke':>6} {'Both':>6} {'Neither':>8}")
    print("-" * 46)
    for split, imgs in split_map.items():
        fire = smoke = both = neither = 0
        for img in imgs:
            lbl = img.with_suffix(".txt")
            classes = get_image_classes(lbl)
            b = bucket_name(classes)
            if b == "fire_only":  fire += 1
            elif b == "smoke_only": smoke += 1
            elif b == "both":     both += 1
            else:                 neither += 1
        print(f"{split:<8} {len(imgs):>6} {fire:>6} {smoke:>6} {both:>6} {neither:>8}")

    print(f"\nDataset written to {DATA_PROCESSED}")


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
