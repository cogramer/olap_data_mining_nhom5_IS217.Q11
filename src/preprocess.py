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
    cat_cols = ["BrandName", "DriveType", "FuelType", "Location", "OwnerCount", "VehicleType", "CarName"]
    encoder = OrdinalEncoder()

    # Fit first to save BrandName mapping
    encoder.fit(df[cat_cols])

    # --- Build and save BrandName mapping ---
    brand_categories = encoder.categories_[cat_cols.index("BrandName")]
    brand_mapping = {brand: idx for idx, brand in enumerate(brand_categories)}

    print("BrandName Encoding Mapping:")
    print(brand_mapping)

    # Apply the transform
    df[cat_cols] = encoder.transform(df[cat_cols])

    df = df.dropna()

    return df, brand_mapping