"""Tests for project configuration."""

from pathlib import Path

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, PROJECT_ROOT


def test_project_paths_exist() -> None:
    assert PROJECT_ROOT.exists()
    assert DATA_RAW_DIR.exists()
    assert DATA_PROCESSED_DIR.exists()
    assert MODELS_DIR.exists()


def test_project_root_is_directory() -> None:
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "src").is_dir()
    assert (PROJECT_ROOT / "tests").is_dir()
