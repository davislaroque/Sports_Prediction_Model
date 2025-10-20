🏀 Sports Prediction Model

Overview

This project uses machine learning to predict NBA player performance in individual games. It demonstrates skills in data wrangling, feature engineering, model building, and statistical evaluation.

I tested multiple models (Linear Regression, Random Forest Regressor, etc.) and compared them using error metrics and prediction intervals. The goal was to evaluate how well different approaches can capture player-level performance patterns.

✨ Key Features

Built an end-to-end ML pipeline: data loading → cleaning → feature engineering → modeling → evaluation.

Implemented and compared multiple regression models.

Evaluated performance with metrics such as MAE, RMSE, and prediction intervals.

Produced interpretable results with tables and visualizations.

Designed for reproducibility and modular improvements.



📂 Project Structure
Sports_Prediction_Model/

├── data/                <- Raw or processed data (not tracked in repo)

├── notebooks/           <- Jupyter notebooks (exploration, modeling, results)

├── src/                 <- Reusable Python scripts (feature engineering, models)

├── outputs/             <- Saved figures, result tables, and reports

├── README.md            <- Project documentation (this file)

├── requirements.txt     <- Python dependencies

└── .gitignore           <- Ignore unnecessary files




🚀 Installation & Setup

Clone this repository:

git clone https://github.com/davislaroque/Sports_Prediction_Model.git
cd Sports_Prediction_Model


Run the notebooks in order:

01_data_exploration.ipynb – initial data analysis and cleaning

02_feature_engineering.ipynb – feature creation and processing

03_modeling_results.ipynb – training models and evaluating results

Example (in Jupyter):

jupyter notebook notebooks/03_modeling_results.ipynb


🔮 Future Work

Incorporate additional features (player fatigue, back-to-back games, etc.)

Expand dataset to multiple seasons

Experiment with deep learning models (e.g., LSTMs for time series)

Deploy as an interactive web app (Flask/Streamlit)

⚙️ Tech Stack

Python (pandas, scikit-learn, matplotlib, seaborn)

Jupyter Notebook for analysis & visualization

Git/GitHub for version control
