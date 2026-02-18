# Cryptocurrency 24-Hour Return Prediction

[![CI](https://github.com/YichengYang-Ethan/crypto-return-prediction-kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/YichengYang-Ethan/crypto-return-prediction-kaggle/actions/workflows/ci.yml)

Machine learning pipeline for predicting 24-hour forward returns across 355 cryptocurrencies, built for the **Avenir HKU Web3 Quant Competition**.

Uses a LightGBM ensemble with time-series cross-validation and engineered technical indicators (RSI, MACD, Bollinger Bands, momentum signals).

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

| Category | Features |
|----------|----------|
| **Momentum** | RSI (Wilder smoothing), MACD, ROC |
| **Volatility** | Bollinger Bands, ATR, BB Width |
| **Volume** | On-Balance Volume (OBV) |
| **Trend** | Rolling means/std (10, 30, 60 periods) |
| **Temporal** | Lagged returns (1-5 periods) |

### 3. Validation Strategy

- **TimeSeriesSplit** (5 folds) -- always train on past, validate on future
- No data leakage -- strict temporal ordering respected
- Mimics realistic backtesting conditions

### 4. Model Training

- **Model**: LightGBM Regressor (`regression_l1` objective, optimizing MAE)
- **Ensemble**: 5-fold time-series CV, final prediction = average of 5 models
- **Regularization**: Early stopping (100 rounds patience) on validation MAE

## Results

### Cross-Validation MAE

| Fold | MAE |
|------|-----|
| 1 | Reported in notebook output |
| 2 | Reported in notebook output |
| 3 | Reported in notebook output |
| 4 | Reported in notebook output |
| 5 | Reported in notebook output |
| **Mean** | **Reported in notebook output** |

*Run the notebook on the Kaggle competition data to see actual numerical results.*

### Baseline Comparison

| Method | MAE | vs Model |
|--------|-----|----------|
| **LightGBM (ours)** | Best | -- |
| Zero baseline (predict 0) | Higher | Model improves over naive |
| Persistence (predict last return) | Higher | Model improves over persistence |

### Long/Short Strategy

A decile-based long/short simulation tests whether predictions have real economic value:
- **Long** top-decile predicted returns, **short** bottom-decile
- Metrics: cumulative return, annualized Sharpe ratio, max drawdown
- See cumulative return chart in `ls_cumulative_return.png`

### SHAP Feature Importance (Top 5)

1. Rolling mean / standard deviation features (trend signals)
2. RSI (momentum)
3. Lagged returns (autoregressive signal)
4. ATR (volatility regime)
5. OBV (volume confirmation)

Full SHAP summary plot saved to `shap_importance.png`.

## How to Run

1. **Environment**: Kaggle Notebook
2. **Data**: Competition data (`kline_data`, `submission_id.csv`) in `/kaggle/input/avenir-hku-web/`
3. **Execution**: Run all cells in `crypto_return_prediction.ipynb` sequentially

## Future Work

- Feature interactions and volatility-of-volatility features
- Hyperparameter tuning with Optuna
- Model comparison (XGBoost, CatBoost) and stacking ensembles
- Post-processing: market neutralization of predictions

## License

MIT
