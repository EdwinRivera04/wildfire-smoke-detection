import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import DATA_MOCK, MOCK_DATASET_YAML


def test_mock_folder_structure():
    for split in ["train", "val", "test"]:
        assert (DATA_MOCK / f"images/{split}").exists()
        assert (DATA_MOCK / f"labels/{split}").exists()


def test_mock_split_counts():
    assert len(list((DATA_MOCK / "images/train").glob("*.jpg"))) == 6
    assert len(list((DATA_MOCK / "images/val").glob("*.jpg"))) == 2
    assert len(list((DATA_MOCK / "images/test").glob("*.jpg"))) == 2


def test_every_image_has_label():
    for split in ["train", "val", "test"]:
        for img in (DATA_MOCK / f"images/{split}").glob("*.jpg"):
            lbl = DATA_MOCK / f"labels/{split}/{img.stem}.txt"
            assert lbl.exists(), f"Missing label for {img.name}"


def test_label_format():
    for split in ["train", "val", "test"]:
        for lbl in (DATA_MOCK / f"labels/{split}").glob("*.txt"):
            text = lbl.read_text().strip()
            if not text:
                continue
            for line in text.splitlines():
                parts = line.split()
                assert len(parts) == 5, f"Bad label in {lbl}: {line}"
                assert int(parts[0]) in (0, 1), f"Invalid class in {lbl}: {parts[0]}"
                assert all(0.0 <= float(v) <= 1.0 for v in parts[1:]), \
                    f"Coords out of range in {lbl}: {parts[1:]}"


def test_dataset_yaml_keys():
    cfg = yaml.safe_load(MOCK_DATASET_YAML.read_text())
    assert cfg["nc"] == 2
    assert cfg["names"] == ["fire", "smoke"]
    assert "train" in cfg
    assert "val" in cfg
    assert "test" in cfg


def test_prepare_dataset_mock_exits_zero():
    result = subprocess.run(
        [sys.executable, "src/data/prepare_dataset.py", "--mock"],
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode()
