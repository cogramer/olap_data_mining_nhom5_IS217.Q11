from src.data_loader import load_data
from src.preprocess import preprocess
from src.train_xgboost import train_xgboost
from src.detect_anomaly import detect_anomaly
from src.depreciation import fit_depreciation
from src.visualize import plot_xgboost

df = load_data("data/car_24_database.xls")
df = preprocess(df)

model, preds, y_test = train_xgboost(df)
df = detect_anomaly(df)

print(df.head())

plot_xgboost(y_test, preds)