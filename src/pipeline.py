"""Build pregame features and evaluate fixed models on later game dates."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGETS = ("points", "reboundsTotal", "assists")
STATS = ("points", "reboundsTotal", "assists", "numMinutes")
ROOT = Path(__file__).resolve().parents[1]


def load_data(path, player=None):
    """Read a local box-score CSV and validate the input schema."""
    frame = pd.read_csv(path)
    required = {"firstName", "lastName", "gameDate", *STATS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    if frame[["firstName", "lastName", "gameDate"]].isna().any().any():
        raise ValueError("Player names and game dates must be present.")
    frame["player"] = (
        frame["firstName"].str.strip() + " " + frame["lastName"].str.strip()
    )
    if player:
        frame = frame.loc[frame["player"].str.casefold() == player.casefold()].copy()
    if frame.empty:
        raise ValueError("No game rows match the selected player or dataset.")
    frame["gameDate"] = pd.to_datetime(frame["gameDate"], utc=True).dt.normalize()
    for stat in STATS:
        frame[stat] = pd.to_numeric(frame[stat], errors="raise")
        if not np.isfinite(frame[stat]).all() or (frame[stat] < 0).any():
            raise ValueError(f"{stat} must contain finite, nonnegative values.")
    if frame.duplicated(["player", "gameDate"]).any():
        raise ValueError(
            "Expected one row per player and game date; deduplicate the CSV."
        )
    return frame.sort_values(["player", "gameDate"]).reset_index(drop=True)


def build_features(frame, target="points"):
    """Use earlier games only. The target is the statistic in the current row."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    out = frame.sort_values(["player", "gameDate"]).copy()
    features = []
    for stat in STATS:
        name = f"{stat}_prior_5"
        out[name] = out.groupby("player")[stat].transform(
            lambda values: values.shift(1).rolling(5, min_periods=1).mean()
        )
        features.append(name)
    out["target_prior_3"] = out.groupby("player")[target].transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    )
    out["days_since_last_game"] = out.groupby("player")["gameDate"].diff().dt.days
    out["game_month"] = out["gameDate"].dt.month
    out["game_dayofweek"] = out["gameDate"].dt.dayofweek
    features += [
        "target_prior_3",
        "days_since_last_game",
        "game_month",
        "game_dayofweek",
    ]
    return out.dropna(subset=[f"{target}_prior_5", target]), features


def chronological_split(frame, test_size=0.2):
    """Keep all players on a given date in the same partition."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    dates = sorted(frame["gameDate"].unique())
    if len(dates) < 5:
        raise ValueError("At least five dates with prior game history are required.")
    cut = max(1, min(len(dates) - 1, int(len(dates) * (1 - test_size))))
    return (
        frame.loc[frame["gameDate"] < dates[cut]].copy(),
        frame.loc[frame["gameDate"] >= dates[cut]].copy(),
    )


def evaluate(frame, features, target="points", test_size=0.2):
    """Compare regressors with a prior-five-game mean on the same holdout."""
    train, test = chronological_split(frame, test_size)
    models = {
        "linear_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100, min_samples_leaf=3, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
    }
    predictions = (
        test[["player", "gameDate", target]].rename(columns={target: "actual"}).copy()
    )
    predictions["prior_5_mean"] = test[f"{target}_prior_5"]
    for name, model in models.items():
        model.fit(train[features], train[target])
        predictions[name] = model.predict(test[features])
    metrics = {}
    for name in ["prior_5_mean", *models]:
        metrics[name] = {
            "mae": float(mean_absolute_error(predictions["actual"], predictions[name])),
            "rmse": float(
                np.sqrt(mean_squared_error(predictions["actual"], predictions[name]))
            ),
        }
    return (
        {
            "target": target,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_end": str(train["gameDate"].max().date()),
            "test_start": str(test["gameDate"].min().date()),
            "test_end": str(test["gameDate"].max().date()),
            "seed": 42,
            "features": features,
            "metrics": metrics,
        },
        predictions,
        models,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/sample_player_games.csv"
    )
    parser.add_argument(
        "--player", help="Optional full name, matched without case sensitivity"
    )
    parser.add_argument("--target", choices=TARGETS, default="points")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/demo")
    args = parser.parse_args()
    try:
        raw = load_data(args.data, args.player)
        frame, features = build_features(raw, args.target)
        report, predictions, _ = evaluate(frame, features, args.target, args.test_size)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report["dataset"] = args.data.name
    report["data_kind"] = (
        "synthetic demonstration"
        if args.data.resolve() == (ROOT / "data/sample_player_games.csv").resolve()
        else "user supplied; provenance must be documented separately"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    print(json.dumps(report, indent=2))
    print(f"Saved metrics.json and predictions.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
