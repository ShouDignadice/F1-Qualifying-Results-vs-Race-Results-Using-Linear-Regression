from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Loading data from .csv
def load_data(data_dir: Path) -> dict:

    data = {
        "constructors": pd.read_csv(data_dir / "constructors.csv"),
        "qualifying": pd.read_csv(data_dir / "qualifying.csv"),
        "results": pd.read_csv(data_dir / "results.csv"),
        "drivers": pd.read_csv(data_dir / "drivers.csv"),
        "races": pd.read_csv(data_dir / "races.csv"),
    }

    return data

# prep
def prep_dataset(data: dict) -> pd.DataFrame:

    qualifying_clean = data["qualifying"][
        ["raceId", "driverId", "constructorId", "position"]
    ].rename(columns={
        "position": "qualifying_position"
    })

    results_clean = data["results"][
        ["raceId", "driverId", "constructorId", "grid", "positionOrder"]
    ].rename(columns={
        "grid": "starting_grid",
        "positionOrder": "race_finish_position"
    })

    races_clean = data["races"][
        ["raceId", "year", "name"]
    ].rename(columns={
        "name": "race_name"
    })

    constructors_clean =data["constructors"][
        ["constructorId", "name"]
    ].rename(columns={
        "name": "constructor_name"
    })

    drivers_clean = data["drivers"].copy()
    drivers_clean["driver_name"] = (
        drivers_clean["forename"] + " " + drivers_clean["surname"]
    )

    drivers_clean = drivers_clean[
        ["driverId", "driver_name", "nationality"]
    ]

    f1_data = qualifying_clean.merge(
        results_clean,
        on=["raceId", "driverId", "constructorId"],
        how="inner"
    )

    f1_data = f1_data.merge(
        races_clean,
        on=["raceId"],
        how="left"
    )

    f1_data = f1_data.merge(
        drivers_clean,
        on=["driverId"],
        how="left"
    )

    f1_data = f1_data.merge(
        constructors_clean,
        on=["constructorId"],
        how="left"
    )

    return f1_data

# clean
def clean_model_columns(f1_data: pd.DataFrame) -> pd.DataFrame:

    numeric_columns = [
        "qualifying_position",
        "starting_grid",
        "race_finish_position"
    ]

    for column in numeric_columns:
        f1_data[column] = pd.to_numeric(
            f1_data[column],
            errors="coerce"
        )

    f1_data = f1_data.dropna(subset=[
        "qualifying_position",
        "race_finish_position"
    ])

    f1_data = f1_data[
        (f1_data["qualifying_position"] > 0) &
        (f1_data["race_finish_position"] > 0)
    ]

    f1_data["position_change"] = (
        f1_data["qualifying_position"] - f1_data["race_finish_position"]
    )

    return f1_data

def train_model(f1_data: pd.DataFrame):

    X = f1_data[["qualifying_position"]]
    y = f1_data["race_finish_position"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_prediction = model.predict(X_test)

    return model, X_train, X_test, y_train, y_test, y_prediction

def evaluate_model(
        model,
        f1_data: pd.DataFrame,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        y_prediction: np.ndarray
) -> dict:

    correlation = f1_data["qualifying_position"].corr(
        f1_data["race_finish_position"]
    )

    metrics = {
        "correlation": correlation,
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "training_r2": model.score(X_train, y_train),
        "testing_r2": r2_score(y_test, y_prediction),
        "mae": mean_absolute_error(y_test, y_prediction),
        "rmse": np.sqrt(mean_squared_error(y_test, y_prediction)),
    }

    return metrics

def print_model_summary(metrics: dict) -> None:

    print("\nLinear Regression Model Summary")
    print("=" * 40)

    print(
        "Model Equation: "
        f"race_finish_position = {metrics['intercept']:.4f} + "
        f"{metrics['slope']:.4f} * qualifying_position"
    )

    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"Slope: {metrics['slope']:.4f}")
    print(f"Intercept: {metrics['intercept']:.4f}")
    print(f"Training R²: {metrics['training_r2']:.4f}")
    print(f"Testing R²: {metrics['testing_r2']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")

def create_results_table(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prediction: np.ndarray
) -> pd.DataFrame:

    results_table = X_test.copy()

    results_table["actual_finish"] = y_test.values
    results_table["predicted_finish"] = y_prediction
    results_table["error"] = (
        results_table["actual_finish"] -
        results_table["predicted_finish"]
    )

    return results_table

def plot_regression_line(
    f1_data: pd.DataFrame,
    model: LinearRegression,
    output_dir: Path
) -> None:

    plt.figure(figsize=(9, 6))

    plt.scatter(
        f1_data["qualifying_position"],
        f1_data["race_finish_position"],
        alpha=0.5,
        label="Actual Data"
    )

    line_x = pd.DataFrame({
        "qualifying_position": sorted(
            f1_data["qualifying_position"].unique()
        )
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

    plt.savefig(output_dir / "regression_plot.png", dpi=300)
    plt.show()

def plot_residuals(
    y_test: pd.Series,
    y_prediction: np.ndarray,
    output_dir: Path
) -> None:

    residuals = y_test - y_prediction

    plt.figure(figsize=(9, 6))

    plt.scatter(y_prediction, residuals, alpha=0.5)
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Race Finish Position")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()

    plt.savefig(output_dir / "residual_plot.png", dpi=300)
    plt.show()

def main() -> None:
    """Run the full Formula 1 linear regression project."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    data = load_data(DATA_DIR)

    f1_data = prep_dataset(data)
    f1_data = clean_model_columns(f1_data)

    model, X_train, X_test, y_train, y_test, y_prediction = train_model(f1_data)

    metrics = evaluate_model(
        model,
        f1_data,
        X_train,
        y_train,
        y_test,
        y_prediction
    )

    results_table = create_results_table(
        X_test,
        y_test,
        y_prediction
    )

    print_model_summary(metrics)

    print("\nActual vs Predicted Results")
    print("=" * 40)
    print(results_table.head(10))

    plot_regression_line(f1_data, model, OUTPUT_DIR)
    plot_residuals(y_test, y_prediction, OUTPUT_DIR)

    f1_data.to_csv(
        OUTPUT_DIR / "f1_merged_data.csv",
        index=False
    )

    results_table.to_csv(
        OUTPUT_DIR / "actual_vs_predicted_results.csv",
        index=False
    )

if __name__ == "__main__":
    main()

