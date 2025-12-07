import os
from datetime import datetime

def create_snapshot(df):
    os.makedirs("test", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test/snapshots/car24database_snapshot_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print("File saved to:", filename)