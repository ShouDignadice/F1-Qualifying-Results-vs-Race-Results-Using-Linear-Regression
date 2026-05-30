from statistics import correlation

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Load CSV files
constructors = pd.read_csv(DATA_DIR / "constructors.csv")
qualifying = pd.read_csv(DATA_DIR / "qualifying.csv")
results = pd.read_csv(DATA_DIR / "results.csv")
drivers = pd.read_csv(DATA_DIR / "drivers.csv")
races = pd.read_csv(DATA_DIR / "races.csv")

# Clean qualifying data
qualifying_clean = qualifying[[
    "raceId",
    "driverId",
    "constructorId",
    "position"
]].rename(columns={
    "position": "qualifying_position"
})

# Clean race results data
results_clean = results[[
    "raceId",
    "driverId",
    "constructorId",
    "grid",
    "positionOrder",
]].rename(columns={
    "grid": "starting_grid",
    "positionOrder": "race_finish_position",
})

# Merge qualifying and race results
f1_data = pd.merge(
    qualifying_clean,
    results_clean,
    on=["raceId", "driverId", "constructorId"],
    how="inner"
)

# Add race details
drivers["driver_name"] = drivers["forename"] + " " + drivers["surname"]

f1_data = pd.merge(
    f1_data,
    drivers[["driverId", "driver_name", "nationality"]],
    on="driverId",
    how="left"
)

# Add constructor/team details
f1_data = pd.merge(
    f1_data,
    constructors[["constructorId", "name"]],
    on="constructorId",
    how="left"
)

# Rename columns for clarity
f1_data = f1_data.rename(columns={
    "name": "constructor_name"
}
)

#========================
#Creating main variables
#========================

# Convert important columns to numeric
f1_data["qualifying_position"] = pd.to_numeric(
    f1_data["qualifying_position"],
    errors="coerce"
)

f1_data["starting_grid"] = pd.to_numeric(
    f1_data["starting_grid"],
    errors="coerce"
)

f1_data["race_finish_position"] = pd.to_numeric(
    f1_data["race_finish_position"],
    errors="coerce"
)

# Remove rows with missing main values
f1_data = f1_data.dropna(subset=[
    "qualifying_position",
    "race_finish_position"
])

# Remove invalid position values
f1_data = f1_data[
    (f1_data["qualifying_position"] > 0) &
    (f1_data["race_finish_position"] > 0)
]

# Create position change variable
f1_data["position_change"] = (
    f1_data["qualifying_position"] - f1_data["race_finish_position"]
)

# Linear regression variables
X = f1_data[["qualifying_position"]]
y = f1_data["race_finish_position"]

#=====================
# Check relationship
#=====================

corr1 = f1_data["qualifying_position"].corr(
    f1_data["race_finish_position"]
)

print("\nCorrelation between qualifying position and race finish position:")
print(corr1)

# # Preview merged dataset
# print("Merged F1 Dataset Preview:")
# print(f1_data.head())
#
# print("\nDataset Shape:")
# print(f1_data.shape)
#
# print("\nColumns:")
# print(f1_data.columns)
#
# # Check for missing values
# print("\nMissing Values:")
# print(f1_data.isnull().sum())
#
# # Save merged data set
# f1_data.to_csv("data/f1_merged_data.csv", index=False)
#
# print("\nMerged F1 dataset saved as data/f1_merged_data.csv")
