
from pathlib import Path

# Main project directory
BASE_DIR = Path(__file__).resolve().parent

# Input and output file locations
DATA_PATH = BASE_DIR / "data" / "kc_house_data.csv"
MODEL_PATH = BASE_DIR / "models" / "house_price_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
FIGURES_DIR = BASE_DIR / "figures"

# Features used to predict the house price
NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
    "grade",
    "yr_built",
]

CATEGORICAL_FEATURES = [
    "zipcode"
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Value that our ML model must predict
TARGET_COLUMN = "price"

RANDOM_STATE = 42

