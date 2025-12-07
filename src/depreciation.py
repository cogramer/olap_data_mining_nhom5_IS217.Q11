from scipy.optimize import curve_fit
import numpy as np

def decay(age, a, b):
    return a * np.exp(-b * age)

def fit_depreciation(df):
    age = df["Age"].values
    price = df["Price"].values

    params, _ = curve_fit(decay, age, price)
    return params