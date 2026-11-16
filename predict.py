
import pandas as pd
import joblib

from config import (
    MODEL_PATH,
    FEATURE_COLUMNS,
)


# Load the trained model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python train.py first."
    )

model = joblib.load(MODEL_PATH)


# Define a new house
new_house = {
    "bedrooms": 3,
    "bathrooms": 2.0,
    "sqft_living": 2000,
    "sqft_lot": 5000,
    "floors": 2.0,
    "waterfront": 0,
    "condition": 3,
    "grade": 7,
    "yr_built": 1995,
    "zipcode": "98052",
}


# Convert input to a DataFrame
house_df = pd.DataFrame(
    [new_house],
    columns=FEATURE_COLUMNS
)


# Predict the sale price
prediction = model.predict(house_df)

predicted_price = max(
    0.0,
    float(prediction[0])
)


# Print the result
print("\nHOUSE PRICE PREDICTION")
print("------------------------------")

for feature, value in new_house.items():
    print(f"{feature}: {value}")

print("------------------------------")

print(
    f"Estimated House Price: "
    f"${predicted_price:,.2f}"
)

