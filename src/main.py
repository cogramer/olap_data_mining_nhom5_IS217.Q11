import pandas as pd
import numpy as np

df = pd.read_excel("data/car_24_database.xls", sheet_name=None)
df_DimDrive = df["Dim_Drive"]
df_DimFuel = df["Dim_Fuel"]
df_DimLocation = df["Dim_Location"]
df_DimOwner = df["Dim_Owner"]
df_DimType = df["Dim_Type"]
df_DimYear = df["Dim_Year"]
df_FactCarSale = df["Fact_Car_Sale"]

