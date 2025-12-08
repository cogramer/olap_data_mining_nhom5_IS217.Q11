import matplotlib.pyplot as plt
import numpy as np
from src.depreciation import decay

def plot_xgboost(y_test, preds):
    errors = preds - y_test

    # Scatter colored by error
    plt.scatter(y_test, preds, c=errors, cmap="coolwarm", alpha=0.8)

    # Perfect prediction line (error = 0)
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label="Error = 0")

    plt.colorbar(label="Prediction Error")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("XGBoost Prediction Error Map")
    plt.legend()
    plt.show()

def plot_depreciation(df, params, brand):
    age = df["Age"].values
    price = df["Price"].values

    a, b = params

    age_smooth = np.linspace(age.min(), age.max(), 200)
    price_fit = decay(age_smooth, a, b)

    plt.scatter(age, price, label="Actual Data", alpha=0.6)
    plt.plot(age_smooth, price_fit, color="red", linewidth=2, label="Fitted Exponential Decay")

    plt.xlabel("Age")
    plt.ylabel("Price")
    plt.title(f"Exponential Depreciation Curve Fit — {brand}")
    plt.legend()
    plt.show()