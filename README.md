# NBA Player Performance Forecasting

A Python project for comparing how well recent game history predicts a player's next performance. I built this around NBA box scores to explore data preparation, regression models, and statistical evaluation. The current workflow makes those experiments reproducible from a CSV through a saved holdout report.

**Python · pandas · scikit-learn · time-series feature engineering · model evaluation · GitHub Actions**

## What to look at

- [Modeling pipeline](src/pipeline.py): lagged player features, train-only preprocessing, chronological evaluation, and CSV/JSON outputs.
- [Exploration notebook](notebooks/01_data_exploration.ipynb) and [model comparison](notebooks/02_modeling_results.ipynb): a short walkthrough of the same reusable code.
- [Regression tests](tests/test_pipeline.py): protect against current-game leakage and mixing player histories.
- [Data schema and provenance](data/README.md): use the bundled demo or supply actual box scores.

## Run the demo

Requires Python 3.11 or 3.12. Run commands from the repository root.

```bash
git clone https://github.com/davislaroque/Sports_Prediction_Model.git
cd Sports_Prediction_Model
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.pipeline
python -m pytest -q
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The demo trains linear regression and random forest models, compares them with a prior-five-game average, and writes `outputs/demo/metrics.json` and `outputs/demo/predictions.csv`. The bundled data contains **fictional players and synthetic games**. These outputs demonstrate execution and evaluation, not real NBA accuracy.

To use your own dataset:

```bash
python -m src.pipeline --data data/PlayerStatistics.csv --player "Anthony Edwards" --target reboundsTotal --output-dir outputs/edwards
```

Supported targets: `points`, `reboundsTotal`, and `assists`. Open the walkthrough with `jupyter lab` and run the numbered notebooks from top to bottom.

## Modeling decisions

| Decision | Reason |
| --- | --- |
| Shift player statistics before rolling averages | The predicted game's box score must not appear in its own inputs |
| Split by unique game date | Every player on the same date belongs to the same partition |
| Fit imputation and scaling on training data only | Later observations cannot influence preprocessing |
| Compare both models with a recent-average baseline | Added model complexity should earn its place |
| Save MAE, RMSE, date boundaries, seed, and predictions | Results can be inspected and reproduced |

The last 20% of available game dates form the default holdout. Models are fit once; each later prediction uses the games observed before that date. This is sequential one-game-ahead evaluation, not a forecast of an entire season made on one date.

## Scope and results

The original notebooks explored regression, weighted recent performance, matchup adjustments, and Monte Carlo calculations. The reusable workflow focuses on pregame forecasting and removes same-game inputs from model evaluation. Earlier code remains available in Git history.

The [original results workbook](outputs/results_summary.xls) is retained as a historical artifact. It has not been revalidated against this pipeline and is not used to estimate current accuracy or uncertainty. A real-data benchmark still requires the original dataset, documented date range, and a fresh run.

Next improvements are walk-forward retraining, opponent features available before tipoff, and calibrated prediction intervals. No betting-return or production-deployment claims are made by the demo.

Related work: [Sports Data Integration and Forecasting Pipeline](https://github.com/davislaroque/Sports-Data-Integration-and-Forecasting-Pipeline) covers external API ingestion and odds analysis.
