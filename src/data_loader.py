#Loads fact and dimension tables and join them into a pandas dataframe
import pandas as pd

def join_dimensions(fact, dims):
    df = fact.copy()
    for key, dim in dims.items():
        df = df.merge(dim, on=key, how="left")
    return df

def load_data(path):
    df = pd.read_excel(path, sheet_name=None)

    fact_car_sale = df["Fact_Car_Sale"]

    dim_brand = df["Dim_Brand"]
    dim_drive = df["Dim_Drive"]
    dim_fuel = df["Dim_Fuel"]
    dim_location = df["Dim_Location"]
    dim_owner = df["Dim_Owner"]
    dim_type = df["Dim_Type"]
    dim_year = df["Dim_Year"]

    dims = {
        "BrandID": dim_brand,
        "DriveID": dim_drive,
        "FuelID": dim_fuel,
        "LocationID": dim_location,
        "OwnerID": dim_owner,
        "TypeID": dim_type,
        "YearID": dim_year
    }

    df = join_dimensions(fact_car_sale, dims)

    return df 

