
import json

import pandas as pd
import joblib
import streamlit as st

from config import (
    MODEL_PATH,
    METRICS_PATH,
    FEATURE_COLUMNS,
)


# ---------------------------------------
# Configure the application
# ---------------------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("House Price Prediction")

st.write(
    "Estimate a house's historical sale price "
    "using a trained Machine Learning model."
)

st.warning(
    "This model uses King County property sales from 2014–2015. Predictions are educational estimates, not current market valuations."
)


# ---------------------------------------
# Check whether the model exists
# ---------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "The model hasn't been trained yet. "
        "Run 'python train.py' first."
    )
    st.stop()

if not METRICS_PATH.exists():
    st.error(
        "Model metadata missing. "
        "Run 'python train.py' first."
    )
    st.stop()


# ---------------------------------------
# Load model and metrics
# ---------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return json.loads(
        METRICS_PATH.read_text(
            encoding="utf-8"
        )
    )


model = load_model()
metadata = load_metadata()

zipcodes = metadata["zipcodes"]

default_zip_index = (
    zipcodes.index("98052")
    if "98052" in zipcodes
    else 0
)


# ---------------------------------------
# Show model information
# ---------------------------------------

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Selected Model",
        metadata["best_model"]
    )

with col2:
    st.metric(
        "Test MAE",
        f"${metadata['test_results']['MAE']:,.0f}"
    )

with col3:
    st.metric(
        "Test R²",
        f"{metadata['test_results']['R2']:.3f}"
    )

st.divider()


# ---------------------------------------
# Create the property input form
# ---------------------------------------

st.subheader("Enter Property Details")

with st.form("house_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=0,
            max_value=20,
            value=3,
            step=1
        )

        living_area = st.number_input(
            "Living Area (sq ft)",
            min_value=100,
            max_value=20000,
            value=2000,
            step=100
        )

        lot_area = st.number_input(
            "Lot Area (sq ft)",
            min_value=100,
            max_value=2000000,
            value=5000,
            step=100
        )

        floors = st.number_input(
            "Number of Floors",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5
        )

        zipcode = st.selectbox(
            "ZIP Code",
            options=zipcodes,
            index=default_zip_index
        )

    with col2:

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.0,
            max_value=15.0,
            value=2.0,
            step=0.25
        )

        waterfront = st.checkbox(
            "Waterfront Property"
        )

        condition = st.slider(
            "Property Condition",
            min_value=1,
            max_value=5,
            value=3
        )

        grade = st.slider(
            "Construction Grade",
            min_value=1,
            max_value=13,
            value=7
        )

        year_built = st.number_input(
            "Year Built",
            min_value=1900,
            max_value=2015,
            value=1995,
            step=1
        )

    submitted = st.form_submit_button(
        "Predict House Price",
        use_container_width=True
    )


# ---------------------------------------
# Generate the prediction
# ---------------------------------------

if submitted:

    house = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": living_area,
        "sqft_lot": lot_area,
        "floors": floors,
        "waterfront": int(waterfront),
        "condition": condition,
        "grade": grade,
        "yr_built": year_built,
        "zipcode": zipcode,
    }

    input_df = pd.DataFrame(
        [house],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(input_df)

    predicted_price = max(
        0.0,
        float(prediction[0])
    )

    st.success("Prediction completed!")

    st.metric(
        "Estimated House Price",
        f"${predicted_price:,.0f}"
    )

    st.caption(
        "This estimate is based on historical "
        "housing transactions and may differ "
        "substantially from an actual sale price."
    )

    with st.expander("View Model Input Data"):
        st.dataframe(input_df)


# ---------------------------------------
# Footer
# ---------------------------------------

st.divider()

st.caption(
    "House Price Prediction | "
    "Python + Scikit-learn + Streamlit"
)
