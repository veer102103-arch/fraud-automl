"""Smoke test without AutoGluon (fast)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_download_data():
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_data.py"), "--rows", "1000"])
    assert (ROOT / "data" / "raw" / "creditcard_sample.csv").exists()
