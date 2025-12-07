from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split

def train_xgboost(df):
    X = df.drop("Price", axis=1)
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = XGBRFRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return model, preds, y_test