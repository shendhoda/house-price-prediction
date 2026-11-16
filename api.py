
import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

from config import (
    MODEL_PATH,
    FEATURE_COLUMNS,
)


# ---------------------------------------
# Initialise FastAPI
# ---------------------------------------

app = FastAPI(
    title="House Price Prediction API",
    description=(
        "Predict historical King County "
        "house sale prices using ML."
    ),
    version="1.0.0"
)


# ---------------------------------------
# Load the trained model
# ---------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(
        "Model not found. Run python train.py first."
    )

model = joblib.load(MODEL_PATH)


# ---------------------------------------
# Define the expected input
# ---------------------------------------

class HouseInput(BaseModel):

    bedrooms: int = Field(
        ge=0,
        le=20
    )

    bathrooms: float = Field(
        ge=0,
        le=15
    )

    sqft_living: float = Field(
        gt=0
    )

    sqft_lot: float = Field(
        gt=0
    )

    floors: float = Field(
        ge=1,
        le=10
    )

    waterfront: int = Field(
        ge=0,
        le=1
    )

    condition: int = Field(
        ge=1,
        le=5
    )

    grade: int = Field(
        ge=1,
        le=13
    )

    yr_built: int = Field(
        ge=1900,
        le=2015
    )

    zipcode: str = Field(
        pattern=r"^\d{5}$"
    )


# ---------------------------------------
# Health check endpoint
# ---------------------------------------

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ---------------------------------------
# Prediction endpoint
# ---------------------------------------

@app.post("/predict")
def predict_price(house: HouseInput):

    # Convert validated JSON input into a DataFrame
    house_data = house.model_dump()

    input_df = pd.DataFrame(
        [house_data],
        columns=FEATURE_COLUMNS
    )

    # Generate prediction
    prediction = model.predict(input_df)

    predicted_price = max(
        0.0,
        float(prediction[0])
    )

    # Return the prediction as JSON
    return {
        "predicted_price": round(
            predicted_price,
            2
        ),
        "currency": "USD",
        "model": "house_price_predictor",
        "note": "Historical educational estimate"
    }
