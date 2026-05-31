import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

DATA_DIR = Path("data")

# Load CSV files
constructors = pd.read_csv(DATA_DIR / "constructors.csv")
qualifying = pd.read_csv(DATA_DIR / "qualifying.csv")
results = pd.read_csv(DATA_DIR / "results.csv")
drivers = pd.read_csv(DATA_DIR / "drivers.csv")

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

# Check correlation
corr1 = f1_data["qualifying_position"].corr(
    f1_data["race_finish_position"]
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test data
y_prediction = model.predict(X_test)

# Model evaluation 
train_r2 = model.score(X_train, y_train)
test_r2 = r2_score(y_test, y_prediction)
mae = mean_absolute_error(y_test, y_prediction)
rmse = np.sqrt(mean_squared_error(y_test, y_prediction))

print(
    f"Model Equation: race_finish_position = {model.intercept_:.4f} + "
    f"{model.coef_[0]:.4f} * qualifying_position"
)

print("Correlation:", corr1)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Training R^2:", train_r2)
print("Testing R^2:", test_r2)
print("MAE:", mae)
print("RMSE:", rmse)

#=============
# Actual vs Predicted Table
#=============
results_table = X_test.copy()
results_table["actual_finish"] = y_test
results_table["predicted_finish"] = y_prediction
results_table["error"] = results_table["actual_finish"] - results_table["predicted_finish"]

print("\nActual vs Predicted Results:")
print(results_table.head(10))

#=============
# Regression Line Plot
#=============
plt.scatter(
    f1_data["qualifying_position"],
    f1_data["race_finish_position"],
    label="Actual Data"
)

# Create sorted x-values so the regression line displays correctly
line_x = pd.DataFrame({
    "qualifying_position": sorted(f1_data["qualifying_position"].unique())
})

line_y = model.predict(line_x)

plt.plot(
    line_x["qualifying_position"],
    line_y,
    label="Regression Line"
)

plt.xlabel("Qualifying Position")
plt.ylabel("Race Finish Position")
plt.title("Linear Regression: Qualifying vs Race Finish Position")
plt.legend()
plt.tight_layout()
plt.show()

#=============
# Residual Plot
#=============
residuals = y_test - y_prediction

plt.scatter(y_prediction, residuals)
plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Race Finish Position")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.show()

results_table.to_csv("data/actual_vs_predicted_results.csv", index=False)
f1_data.to_csv("data/f1_merged_data.csv", index=False)