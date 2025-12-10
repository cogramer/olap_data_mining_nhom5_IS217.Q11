from sklearn.ensemble import IsolationForest

def detect_anomaly(df):
    iso = IsolationForest(contamination=0.02, random_state=42)
    df["Anomaly"] = iso.fit_predict(df[["Price", "Distance", "Age"]])

    df["Anomaly_Type"] = "normal"

    anomalies = df["Anomaly"] == -1

    df.loc[anomalies & (df["Price"] > df["Price"].quantile(0.98)), "Anomaly_Type"] = "price too high"
    df.loc[anomalies & (df["Price"] < df["Price"].quantile(0.02)), "Anomaly_Type"] = "price too low"

    df.loc[anomalies & (df["Distance"] < df["Distance"].quantile(0.02)), "Anomaly_Type"] = "distance too low"
    df.loc[anomalies & (df["Distance"] > df["Distance"].quantile(0.98)), "Anomaly_Type"] = "distance too high"

    df.loc[anomalies & (df["Age"] < df["Age"].quantile(0.02)), "Anomaly_Type"] = "age too new"
    df.loc[anomalies & (df["Age"] > df["Age"].quantile(0.98)), "Anomaly_Type"] = "age too old"

    return df
