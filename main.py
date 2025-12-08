from src.data_loader import load_data
from src.preprocess import preprocess
from src.train_xgboost import train_xgboost
from src.detect_anomaly import detect_anomaly
from src.depreciation import fit_depreciation
from src.visualize import plot_xgboost, plot_depreciation
from src.util import create_snapshot

df = load_data("data/car_24_database.xls")
df = preprocess(df)

model, preds, y_test = train_xgboost(df)
df = detect_anomaly(df)

print(df[["Year", "Age", "Price", "Distance", "Anomaly"]].head(10))


plot_xgboost(y_test, preds)


for brand in df["BrandName"].unique():
    sub = df[df["BrandName"] == brand]

    params = fit_depreciation(sub)
    if params is None:
        print(f"Skipping {brand}, not enough data")
        continue

    plot_depreciation(sub, params, brand)


create_snapshot(df)