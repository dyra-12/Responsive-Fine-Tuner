import subprocess
import sys
import tempfile
import os
from types import SimpleNamespace

from backend.data_processor import DataProcessor


def make_config():
    data = SimpleNamespace(supported_formats=['.txt', '.csv'], max_file_size_mb=1, test_split=0.2)
    cfg = SimpleNamespace(data=data)
    return cfg


def test_validate_file_empty(tmp_path):
    cfg = make_config()
    dp = DataProcessor(cfg)
    f = tmp_path / "empty.txt"
    f.write_text("", encoding='utf-8')
    assert not dp.validate_file(str(f))


def test_validate_file_unsupported(tmp_path):
    cfg = make_config()
    dp = DataProcessor(cfg)
    f = tmp_path / "file.exe"
    f.write_text("dummy")
    assert not dp.validate_file(str(f))


def test_process_text_file_short_content(tmp_path):
    cfg = make_config()
    dp = DataProcessor(cfg)
    f = tmp_path / "short.txt"
    f.write_text("short", encoding='utf-8')
    docs = dp.process_text_file(str(f))
    assert docs == []


def test_run_phase6_exec():
    res = subprocess.run([sys.executable, 'run_phase6.py'], capture_output=True, text=True)
    assert res.returncode == 0
