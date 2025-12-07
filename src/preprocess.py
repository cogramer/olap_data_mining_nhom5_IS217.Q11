from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime

def preprocess(df):
    df = df.copy()

    df = df.dropna(subset=["Year", "Price", "Distance"])    # Removes missing values
    df = df[df["Price"] > 0]                                # Removes negative prices
    df = df[df["Distance"] >= 0]                            # Removes negative distances

    # Derived measures
    current_year = datetime.now().year
    df["Age"] = current_year - df["Year"]
    df = df[df["Age"] >= 0]

    df["Price_per_km"] = df["Price"] / (df["Distance"] + 1)

    # Categorial encoding
    cat_cols = ["BrandName", "DriveType", "FuelType", "Location", "OwnerCount", "VehicleType"]
    encoder = OrdinalEncoder()
    df[cat_cols] = encoder.fit_transform(df[cat_cols])

    # Final cleaning
    df = df.dropna()

    return df