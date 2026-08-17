# Compact GRU / GRU-TCN — local run instructions

Confirmatory model for Phase 2 (connects to the base paper's GRU-TCN).
CPU is fine. From the repo:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
cd "D:\GHI Forecasting\03_code"
python models/deep_gru_tcn.py --epochs 15 --subsample 150000 --horizons 1 3 6 12
```

- Reads cleaned data + regimes from 02_data/ and GBM predictions from 02_data/interim/.
- Writes 04_results/tables/p2_deep_metrics.csv (RMSE/MAE/R2 + Diebold-Mariano vs GBM).
- Reduce --subsample or --epochs if slow; raise for the final journal run.
Send the CSV back to fold into the Phase 2 report.
