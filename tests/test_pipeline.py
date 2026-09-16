import numpy as np
import pandas as pd
import pytest

from src.pipeline import ROOT, build_features, chronological_split, evaluate, load_data


def test_current_and_future_stats_do_not_change_earlier_features():
    raw = load_data(ROOT / "data/sample_player_games.csv")
    before, columns = build_features(raw)
    changed = raw.copy()
    cutoff = raw["gameDate"].sort_values().iloc[-12]
    changed.loc[changed["gameDate"] >= cutoff, "points"] = 9999
    after, _ = build_features(changed)
    pd.testing.assert_frame_equal(
        before.loc[before.gameDate <= cutoff, columns],
        after.loc[after.gameDate <= cutoff, columns],
    )


def test_rolling_history_stays_with_each_player():
    raw = load_data(ROOT / "data/sample_player_games.csv")
    frame, _ = build_features(raw)
    for player, games in raw.groupby("player"):
        first = frame.loc[frame.player == player].iloc[0]
        assert first["points_prior_5"] == games.iloc[0]["points"]


def test_date_split_and_metrics_use_the_same_holdout():
    frame, columns = build_features(load_data(ROOT / "data/sample_player_games.csv"))
    train, test = chronological_split(frame)
    assert train.gameDate.max() < test.gameDate.min()
    report, predictions, models = evaluate(frame, columns)
    assert report["test_rows"] == len(test) == len(predictions)
    assert set(report["metrics"]) == {
        "prior_5_mean",
        "linear_regression",
        "random_forest",
    }
    assert np.isfinite(predictions.select_dtypes("number")).all().all()
    assert models["linear_regression"].named_steps["scaler"].n_samples_seen_ == len(
        train
    )


def test_invalid_schema_and_unknown_player_fail_clearly(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("points\n10\n")
    with pytest.raises(ValueError, match="Missing CSV columns"):
        load_data(path)
    with pytest.raises(ValueError, match="No game rows"):
        load_data(ROOT / "data/sample_player_games.csv", "Unknown Player")
