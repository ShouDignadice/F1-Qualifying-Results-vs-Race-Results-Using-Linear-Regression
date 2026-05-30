import pandas as pd

# Load CSV files
constructors = pd.read_csv("data/constructors.csv")
qualifying = pd.read_csv("data/qualifying.csv")
results = pd.read_csv("data/results.csv")
drivers = pd.read_csv("data/drivers.csv")
races = pd.read_csv("data/races.csv")

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
    "points",
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

# Preview merged dataset
print("Merged F1 Dataset Preview:")
print(f1_data.head())

print("\nDataset Shape:")
print(f1_data.shape)

print("\nColumns:")
print(f1_data.columns)


# Check for missing values
print("\nMissing Values:")
print(f1_data.isnull().sum())

# Save merged data set
f1_data.to_csv("data/f1_merged_data.csv", index=False)

print("\nMerged F1 dataset saved as data/f1_merged_data.csv")