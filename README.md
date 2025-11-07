# Personal Loan Propensity – Streamlit App

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Expected input
`UniversalBank.csv` with columns:
ID, Personal Loan, Age, Experience, Income, Zip code, Family, CCAvg, Education, Mortgage, Securities, CDAccount, Online, CreditCard.

## Tabs
1. **Insights (5 charts)**: income deciles, 2D heatmap (Income×CCAvg), Education×CD segmentation, threshold sweep, lift curve.
2. **Modeling**: DT/RF/GBDT with stratified 5-fold OOF, metrics table, ROC (all), confusion matrices, feature importances.
3. **Predict**: upload new CSV, score with best model (or quick GBDT fallback), download predictions.
