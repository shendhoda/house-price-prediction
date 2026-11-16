
import json

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from config import (
    DATA_PATH,
    MODEL_PATH,
    METRICS_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
)


# -------------------------------------------------
# Build the preprocessing + ML pipeline
# -------------------------------------------------

def build_pipeline(regressor):

    # Numeric features:
    # Fill missing values with the median
    # and standardise their scales.
    numeric_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    # Categorical features:
    # Fill missing values and convert ZIP codes
    # into machine-readable categories.
    categorical_transformer = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),
    ])

    # Apply different transformations
    # to numerical and categorical data.
    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_transformer,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES
        ),
    ])

    # Combine preprocessing and regression model.
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", regressor),
    ])

    return pipeline


# -------------------------------------------------
# Define evaluation metrics
# -------------------------------------------------

def evaluate(model, data):

    X = data[FEATURE_COLUMNS]
    y_true = data[TARGET_COLUMN]

    y_pred = model.predict(X)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "R2": round(float(r2), 4),
    }


# -------------------------------------------------
# Main training function
# -------------------------------------------------

def main():

    # Load the housing dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Check that all required columns exist
    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN, "id"]
    )

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Remove rows without valid sale prices
    df = df.loc[
        df[TARGET_COLUMN].notna()
        & (df[TARGET_COLUMN] > 0)
    ].copy()

    # Treat ZIP codes as categorical strings
    df["zipcode"] = (
        df["zipcode"]
        .fillna("Unknown")
        .astype(str)
    )

    # -------------------------------------------------
    # Split data into development and testing sets
    # -------------------------------------------------

    # GroupShuffleSplit keeps sales from the same
    # property ID together, reducing leakage.
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    development_idx, test_idx = next(
        first_split.split(
            df,
            groups=df["id"]
        )
    )

    development = df.iloc[development_idx]
    test = df.iloc[test_idx]

    # Create separate training and validation sets
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE + 1
    )

    train_idx, valid_idx = next(
        second_split.split(
            development,
            groups=development["id"]
        )
    )

    train = development.iloc[train_idx]
    valid = development.iloc[valid_idx]

    print("Training rows:", len(train))
    print("Validation rows:", len(valid))
    print("Testing rows:", len(test))

    # -------------------------------------------------
    # Define the ML algorithms
    # -------------------------------------------------

    algorithms = {

        "Median baseline": DummyRegressor(
            strategy="median"
        ),

        "Ridge Regression": Ridge(
            alpha=100.0
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=20,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    # -------------------------------------------------
    # Train and compare all models
    # -------------------------------------------------

    validation_results = {}

    for name, estimator in algorithms.items():

        print(f"\nTraining {name}...")

        pipeline = build_pipeline(
            clone(estimator)
        )

        pipeline.fit(
            train[FEATURE_COLUMNS],
            train[TARGET_COLUMN]
        )

        results = evaluate(
            pipeline,
            valid
        )

        validation_results[name] = results

        print("Validation results:")
        print(results)

    # -------------------------------------------------
    # Select the model with lowest validation MAE
    # -------------------------------------------------

    best_name = min(
        validation_results,
        key=lambda name: validation_results[name]["MAE"]
    )

    print("\nBEST MODEL:", best_name)

    # Retrain on all development data
    best_model = build_pipeline(
        clone(algorithms[best_name])
    )

    best_model.fit(
        development[FEATURE_COLUMNS],
        development[TARGET_COLUMN]
    )

    # -------------------------------------------------
    # Evaluate on completely unseen test data
    # -------------------------------------------------

    test_results = evaluate(
        best_model,
        test
    )

    print("\nFINAL TEST RESULTS")
    print(test_results)

    # -------------------------------------------------
    # Save the model
    # -------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        best_model,
        MODEL_PATH
    )

    # -------------------------------------------------
    # Save performance and configuration
    # -------------------------------------------------

    metadata = {
        "best_model": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "validation_results": validation_results,
        "test_results": test_results,
        "n_training_and_validation": len(development),
        "n_test": len(test),
        "zipcodes": sorted(
            development["zipcode"]
            .unique()
            .tolist()
        ),
    }

    METRICS_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\nMODEL TRAINING COMPLETE!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()

