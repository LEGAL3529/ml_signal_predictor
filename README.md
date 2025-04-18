
# 📈 ML Signal Predictor

This project demonstrates a basic signal prediction model using logistic regression. The script:
- Loads time series market data
- Trains a classifier on `close` and `volume`
- Predicts BUY / SELL / HOLD signals
- Generates a graph of the prediction

## Files Included

- `predict_chart.py` — Main script for training and plotting
- `data/sample_data_with_signals.csv` — Sample data with labeled signals
- `model.pkl` — Saved trained model
- `preview.png` — Visual preview of prediction signals

## Run it

```bash
python predict_chart.py
```
