import json
from pathlib import Path

import pandas as pd

from penguin_saccades import cli, config
from penguin_saccades import constants as C
from tests.conftest import dlc_track, open_track, write_dlc_csv


def test_pipeline_end_to_end(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(C, "N_PERMUTATIONS", 100)
    monkeypatch.setattr(C, "N_BOOTSTRAP", 200)
    open_dir, dlc_dir, out = tmp_path / "open", tmp_path / "dlc", tmp_path / "out"
    open_dir.mkdir()
    dlc_dir.mkdir()
    open_track(saccades=[(3.0, 60.0), (7.0, -60.0)]).to_hdf(open_dir / "Gentoo_morning.h5", key="trace")
    open_track(saccades=[(4.0, 70.0)]).to_hdf(open_dir / "Rockhopper_morning.h5", key="trace")
    write_dlc_csv(dlc_track(), dlc_dir / "07_gentooDLC_resnet50.csv")
    write_dlc_csv(dlc_track(body_saccades=[(6.0, -60.0)], head_saccades=[(5.9, -60.0), (2.5, 35.0)]),
                  dlc_dir / "08_gentooDLC_resnet50.csv")
    paths = tmp_path / "local_paths.json"
    paths.write_text(json.dumps({"active_profile": "local", "profiles": {"local": {
        "data_root": str(tmp_path), "open_trajectories_dir": str(open_dir),
        "closed_dlc_dir": str(dlc_dir), "output_dir": str(out)}}}), encoding="utf-8")
    monkeypatch.setenv(config.ENV_PATHS_FILE, str(paths))

    assert cli.main(["all"]) == 0

    traj = pd.read_csv(out / "trajectories.csv")
    assert set(traj[C.COL_DATASET]) == {"open", "closed"} and traj[C.COL_TRAJECTORY].nunique() == 4
    sacc = pd.read_csv(out / "saccades.csv")
    assert (sacc[C.COL_SOURCE] == "body").sum() >= 5 and (sacc[C.COL_SOURCE] == "head").sum() >= 2
    assert "associated" in set(sacc[C.COL_SACCADE_TYPE])
    ints = pd.read_csv(out / "intervals.csv")
    assert set(ints[C.COL_INTERVAL_CLASS]) == {"any", "body", "head"}
    for name in ("saccade_content.csv", "interval_content.csv", "saccade_budget.csv",
                 "head_body_compensation.csv", "isi_chunks.csv", "summary_tables.csv", "pairwise_tests.csv"):
        assert (out / name).exists(), name
    figures = list((out / "figures").glob("*.svg"))
    assert figures and all(f.with_suffix(".csv").exists() for f in figures)
