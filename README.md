Cryptocurrency 24-Hour Return Prediction Model

This is a machine learning project built for the Avenir HKU Web3 Quant Competition. The core objective of the project is to predict the 24-hour forward returns for a portfolio of 355 cryptocurrencies, using the provided historical K-line data.

This project uses Python as its primary language and leverages libraries such as Pandas, LightGBM, and Scikit-learn to build a complete end-to-end quantitative strategy pipeline, from data processing and feature engineering to model training and prediction.

1. Methodology

The project follows a standard and robust machine learning workflow:

1.1 Data Processing
Data Loading: Loaded historical K-line data for all 355 cryptocurrencies from `.parquet` files.
Data Cleaning: Standardized column names (e.g., `close_price` to `close`) and converted all numeric data to the `float32` data type to ensure computational accuracy while significantly optimizing memory usage.

1.2 Feature Engineering
To enable the model to learn effective patterns from raw price and volume data, several categories of features were engineered:

Technical Indicators: A comprehensive set of classic indicators were calculated, including Momentum (RSI, MACD, ROC), Volatility (Bollinger Bands, ATR), Volume (On-Balance Volume - OBV), and Trend (multiple rolling windows of 10, 30, and 60 periods).
Time-based Features: Extracted temporal features such as the hour of the day (`hour`) and the day of the week (`dayofweek`) from the timestamp to capture potential cyclical market patterns.
Cross-sectional Features: A key feature, the cross-sectional return rank (`return_rank`), was engineered. For each timestamp, this feature ranks a cryptocurrency's return relative to the entire market of 355 assets. This provides a powerful, market-neutral signal of relative strength.

1.3 Validation Strategy
To avoid data leakage and create a realistic backtesting environment, a simple random split was avoided.
A more robust `TimeSeriesSplit` from Scikit-learn was used for cross-validation. The data was split into 5 time-ordered folds, where the model is always trained on past data and validated on future data, respecting the temporal nature of the problem.

1.4 Model Training
Model Choice: LightGBM Regressor was selected for its high performance, efficiency, and excellent handling of tabular data.
Objective Function: The model was trained to optimize for Mean Absolute Error (MAE), using the `regression_l1` objective.
Ensemble Model: Five separate LightGBM models were trained, one for each fold of the `TimeSeriesSplit`. The final prediction is the average (ensemble) of the predictions from these five models, leading to a more robust and generalized result.
Overfitting Prevention: `early_stopping` was utilized during training. The model's performance was monitored on the validation set of each fold, and training was halted automatically if performance did not improve for 100 consecutive rounds.

2. Results
Successfully built an end-to-end pipeline capable of processing large-scale time-series data, training a model, and generating predictions.
Overcame significant memory constraints ("Kernel Died" errors) in the Kaggle environment by implementing robust memory optimization techniques, including using `float32` precision, explicit garbage collection (`del`, `gc.collect()`), and a symbol-by-symbol loop for the final prediction phase.
The final output is a `submission.csv` file that conforms to the competition's required format.

3. How to Run
1.  Environment: This project is designed to run in a Kaggle Notebook environment.
2.  Data: The required data (`kline_data` and `submission_id.csv`) is provided by the competition and is expected to be located in the `/kaggle/input/avenir-hku-web/` directory.
3.  Execution: Simply run all cells in the notebook (`notebookf25843c59d.ipynb`) sequentially from top to bottom.

4. Conclusion and Future Work
This project establishes a solid and robust baseline model for the quantitative trading task. Potential areas for future improvement include:
More advanced feature engineering (e.g., feature interactions, volatility of volatility).
Automated hyperparameter tuning using libraries like Optuna.
Experimenting with different models (XGBoost, CatBoost) and more complex ensemble techniques.
Implementing post-processing on predictions, such as market neutralization.
