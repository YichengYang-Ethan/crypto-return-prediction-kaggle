# Cryptocurrency 24-Hour Return Prediction

[![CI](https://github.com/YichengYang-Ethan/crypto-return-prediction-kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/YichengYang-Ethan/crypto-return-prediction-kaggle/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2980B9?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

LightGBM ensemble for predicting 24-hour forward returns across **355 cryptocurrencies**, built for the **Avenir HKU Web3 Quant Competition**.

Uses 5-fold TimeSeriesSplit cross-validation (always train on past, validate on future) with engineered technical indicators.

## Methodology

### Feature Engineering

| Category | Features |
|----------|----------|
| **Momentum** | RSI (Wilder smoothing), MACD, ROC |
| **Volatility** | Bollinger Bands, ATR, BB Width |
| **Volume** | On-Balance Volume (OBV) |
| **Trend** | Rolling means/std (10, 30, 60 periods) |
| **Temporal** | Lagged returns (1–5 periods) |

### Model

- **LightGBM Regressor** with `regression_l1` objective (optimizing MAE)
- 5-fold TimeSeriesSplit — strict temporal ordering, no data leakage
- Final prediction = average of 5 fold models
- Early stopping (100 rounds patience) on validation MAE

### Evaluation

The model is benchmarked against two naive baselines on the last CV fold:

| Method | Description |
|--------|-------------|
| **LightGBM (ours)** | 5-fold ensemble, per-fold MAE reported at runtime |
| Zero baseline | Always predict 0 return |
| Persistence | Predict last known return |

The model improves over both baselines on MAE. Exact metrics are generated at runtime from the competition dataset (not redistributable) — run the notebook on Kaggle to reproduce.

### Long/Short Strategy

A decile-based simulation tests whether predictions have real economic value:
- **Long** top-decile predicted returns, **short** bottom-decile (equal weight)
- Metrics: cumulative return, annualized Sharpe ratio, max drawdown

### SHAP Feature Importance

Top drivers by mean |SHAP| value:
1. Rolling mean / standard deviation features (trend signals)
2. RSI (momentum)
3. Lagged returns (autoregressive signal)
4. ATR (volatility regime)
5. OBV (volume confirmation)

## How to Run

1. **Environment**: Kaggle Notebook (competition dataset required)
2. **Data**: `kline_data` + `submission_id.csv` in `/kaggle/input/avenir-hku-web/`
3. **Execution**: Run all cells in `crypto_return_prediction.ipynb` sequentially

## License

MIT
