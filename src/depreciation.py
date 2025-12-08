from scipy.optimize import curve_fit
import numpy as np

def decay(age, a, b):
    return a * np.exp(-b * age)

def fit_depreciation(df):
    age = df["Age"].values
    price = df["Price"].values

    if len(age) < 2:
        return None

    try:
        params, _ = curve_fit(
            decay,
            age,
            price,
            p0=[price.max(), 0.1],
            bounds=([0, 0], [np.inf, 1]),
            maxfev=5000
        )
        return params
    except Exception as e:
        print("Skipping due to curve_fit error:", e)
        return None