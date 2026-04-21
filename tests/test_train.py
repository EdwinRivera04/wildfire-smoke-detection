import subprocess
import sys
import time
from pathlib import Path

import yaml


def test_mock_completes_under_90s():
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "src/train.py", "--config", "configs/baseline.yaml", "--mock"],
        capture_output=True,
        timeout=120,
    )
    elapsed = time.time() - t0
    assert result.returncode == 0, result.stderr.decode()
    assert elapsed < 90, f"--mock took {elapsed:.0f}s, expected <90s"


def test_checkpoint_saved():
    checkpoints = list(Path("outputs/checkpoints").rglob("best.pt"))
    assert len(checkpoints) > 0, "No best.pt found in outputs/checkpoints/"


def test_timing_output_printed():
    result = subprocess.run(
        [sys.executable, "src/train.py", "--config", "configs/baseline.yaml", "--mock"],
        capture_output=True,
        timeout=120,
    )
    output = result.stdout.decode() + result.stderr.decode()
    assert "Epoch 1" in output or "\u23f1" in output, "Timing line not printed after epoch 1"


def test_baseline_config_mps_safe():
    cfg = yaml.safe_load(open("configs/baseline.yaml"))
    assert cfg["workers"] == 0,   f"workers must be 0 for MPS, got {cfg['workers']}"
    assert cfg["device"]  == "mps", f"device must be mps, got {cfg['device']}"


def test_mock_uses_mock_dataset():
    result = subprocess.run(
        [sys.executable, "src/train.py", "--config", "configs/baseline.yaml", "--mock"],
        capture_output=True,
        timeout=120,
    )
    output = result.stdout.decode() + result.stderr.decode()
    assert "mock" in output.lower(), "Expected mock dataset path in output"
