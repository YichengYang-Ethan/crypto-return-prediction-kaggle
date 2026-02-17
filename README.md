# Cryptocurrency 24-Hour Return Prediction

Machine learning pipeline for predicting 24-hour forward returns across 355 cryptocurrencies, built for the **Avenir HKU Web3 Quant Competition**.

Uses LightGBM ensemble with time-series cross-validation and a feature engineering framework aligned with [clawdfolio](https://github.com/YichengYang-Ethan/clawdfolio)'s technical indicator library (RSI, MACD, Bollinger Bands, momentum signals).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2980B9?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## Methodology

### 1. Data Processing

- **Source**: Historical K-line data for 355 cryptocurrencies (`.parquet` format)
- **Cleaning**: Standardized column names, converted to `float32` for memory optimization
- **Scale**: Handled large-scale time-series data under Kaggle's memory constraints using explicit garbage collection and symbol-by-symbol prediction loops

### 2. Feature Engineering

| Category | Features | Shared with clawdfolio |
|----------|----------|----------------------|
| **Momentum** | RSI, MACD, ROC | RSI, SMA/EMA |
| **Volatility** | Bollinger Bands, ATR | Bollinger Bands |
| **Volume** | On-Balance Volume (OBV) | — |
| **Trend** | Rolling windows (10, 30, 60 periods) | SMA (20, 50, 200) |
| **Temporal** | Hour-of-day, day-of-week | — |
| **Cross-sectional** | Return rank across 355 assets | HHI concentration |

> The technical indicator methodology (RSI smoothing, Bollinger Band parameterization) follows the same conventions used in [clawdfolio](https://github.com/YichengYang-Ethan/clawdfolio)'s production analytics.

### 3. Validation Strategy

- **TimeSeriesSplit** (5 folds) — always train on past, validate on future
- No data leakage — strict temporal ordering respected
- Mimics realistic backtesting conditions

### 4. Model Training

- **Model**: LightGBM Regressor (`regression_l1` objective, optimizing MAE)
- **Ensemble**: 5-fold time-series CV, final prediction = average of 5 models
- **Regularization**: Early stopping (100 rounds patience) on validation MAE

## Results

- **Validation**: 5-fold TimeSeriesSplit CV with MAE reported per fold (run on Kaggle to see exact scores)
- **Ensemble**: Final predictions averaged across all 5 fold models for robustness
- End-to-end pipeline: data ingestion, feature engineering, training, prediction
- Overcame Kaggle memory constraints via `float32` precision, explicit `gc.collect()`, and batched prediction
- Output: `submission.csv` conforming to competition format

## How to Run

1. **Environment**: Kaggle Notebook
2. **Data**: Competition data (`kline_data`, `submission_id.csv`) in `/kaggle/input/avenir-hku-web/`
3. **Execution**: Run all cells in `crypto_return_prediction.ipynb` sequentially

## Future Work

- Feature interactions and volatility-of-volatility features
- Hyperparameter tuning with Optuna
- Model comparison (XGBoost, CatBoost) and stacking ensembles
- Post-processing: market neutralization of predictions

## Related Projects

| Project | Relationship |
|---------|-------------|
| [clawdfolio](https://github.com/YichengYang-Ethan/clawdfolio) | **Shared methodology** — RSI, Bollinger, SMA indicators used in both feature engineering and production monitoring |
| [ESG-Driven-Stock-Value-Prediction](https://github.com/YichengYang-Ethan/ESG-Driven-Stock-Value-Prediction) | **Complementary research** — this project predicts short-term momentum; ESG predicts long-term value |
| [investment-dashboard](https://github.com/YichengYang-Ethan/investment-dashboard) | Portfolio visualization frontend powered by clawdfolio |
| [QQQ-200D-Deviation-Dashboard](https://github.com/YichengYang-Ethan/QQQ-200D-Deviation-Dashboard) | Single-strategy dashboard using SMA deviation |

## License

MIT
