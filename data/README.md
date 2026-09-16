# Input data

`sample_player_games.csv` is a synthetic fixture with 144 rows: three fictional players, each with 48 games. It uses NumPy's `default_rng(42)` and exists to demonstrate the workflow without a download or API key. Its metrics do not measure NBA prediction performance.

The original exploration referenced [Historical NBA Data and Player Box Scores](https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores), particularly `PlayerStatistics.csv`. Download real data separately and check its provider's usage terms. The original filtered local CSVs are not committed.

| Column | Meaning |
| --- | --- |
| `firstName`, `lastName` | Player identity |
| `gameDate` | Game date, parsed in UTC |
| `points`, `reboundsTotal`, `assists` | Nonnegative game totals |
| `numMinutes` | Nonnegative minutes played |

Extra columns are allowed. Use one row per player per UTC game date. Missing required values and duplicate player/date rows raise an error. Each player's first row is omitted from modeling because it has no earlier games.
