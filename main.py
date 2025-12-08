from src.data_loader import load_data
from src.preprocess import preprocess
from src.train_xgboost import train_xgboost
from src.detect_anomaly import detect_anomaly
from src.depreciation import fit_depreciation
from src.visualize import plot_xgboost, plot_depreciation
from src.util import create_snapshot

df = load_data("data/car_24_database.xls")
df, brand_mapping = preprocess(df)

model, preds, y_test = train_xgboost(df)
df = detect_anomaly(df)

print(df[["Year", "Age", "Price", "Distance", "Anomaly"]].head(10))


plot_xgboost(y_test, preds)



# Reverse the mapping (encoded number → original brand name)
inv_brand_mapping = {v: k for k, v in brand_mapping.items()}

for encoded_brand in df["BrandName"].unique():
    sub = df[df["BrandName"] == encoded_brand]

    params, annual_pct_drop = fit_depreciation(sub)
    if params is None:
        print(f"Skipping {encoded_brand}, not enough data")
        continue

    brand_name = inv_brand_mapping[encoded_brand]
    print(f"{brand_name}: {annual_pct_drop:.2f}% average price decrease per year")
    plot_depreciation(sub, params, brand_name)


create_snapshot(df)