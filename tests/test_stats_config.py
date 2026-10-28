import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from penguin_saccades import config
from penguin_saccades import constants as C
from penguin_saccades import stats


def test_bootstrap_ci_brackets_median():
    x = np.r_[np.arange(1, 50, dtype=float), np.nan]
    med, lo, hi = stats.bootstrap_median_ci(x, n_boot=500)
    assert lo <= med <= hi and med == 25


def test_summarise_counts_finite_only():
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, np.nan, 3.0]})
    out = stats.summarise(df, "v", ["g"])
    assert out.n.tolist() == [1, 1]


def test_fisher_tests_separate_clear_groups(monkeypatch):
    monkeypatch.setattr(C, "N_PERMUTATIONS", 300)
    a, b = np.random.default_rng(0).normal(0, 1, 15), np.random.default_rng(1).normal(5, 1, 15)
    assert stats.two_sample_fisher(a, b) < 0.05
    df = pd.DataFrame({"g": ["a"] * 15 + ["b"] * 15 + ["c"] * 1, "v": np.r_[a, b, 1.0]})
    out = stats.pairwise_fisher(df, "v", "g")
    assert len(out) == 1   # group c dropped: fewer than two values


def test_config_fails_loudly_without_paths_file(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(config.ENV_PATHS_FILE, str(tmp_path / "nope.json"))
    monkeypatch.delenv(config.ENV_DATA_ROOT, raising=False)
    with pytest.raises(config.PathConfigError, match="local_paths.template.json"):
        config.data_root()


def test_config_reads_profile(monkeypatch, tmp_path: Path):
    paths = tmp_path / "local_paths.json"
    paths.write_text(json.dumps({"active_profile": "local", "profiles": {
        "local": {"data_root": str(tmp_path), "output_dir": str(tmp_path / "out")}}}), encoding="utf-8")
    monkeypatch.setenv(config.ENV_PATHS_FILE, str(paths))
    monkeypatch.delenv(config.ENV_DATA_ROOT, raising=False)
    assert config.data_root() == tmp_path and config.output_dir() == tmp_path / "out"
    with pytest.raises(config.PathConfigError, match="closed_dlc_dir"):
        config.get_path("closed_dlc_dir")
    monkeypatch.setenv(config.ENV_DATA_ROOT, str(tmp_path / "env"))
    assert config.data_root() == tmp_path / "env"
    assert os.environ[config.ENV_DATA_ROOT]
