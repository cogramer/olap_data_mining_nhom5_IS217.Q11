from sklearn.ensemble import IsolationForest

def detect_anomaly(df):
    iso = IsolationForest(contamination=0.02)
    df["Anomaly"] = iso.fit_predict(df[["Price", "Distance", "Age"]])
    return df
