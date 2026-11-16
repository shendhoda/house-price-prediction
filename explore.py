
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    DATA_PATH,
    FIGURES_DIR,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def main():
    # 1. Load the dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 2. Display basic information
    print("\nDATASET SHAPE")
    print(df.shape)

    print("\nFIRST 5 ROWS")
    print(df.head().to_string(index=False))

    print("\nCOLUMN INFORMATION")
    df.info()

    print("\nDESCRIPTIVE STATISTICS")
    print(df.describe().to_string())

    # 3. Check missing values
    print("\nMISSING VALUES")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    # 4. Check duplicate rows
    print("\nDUPLICATE ROWS")
    print(df.duplicated().sum())

    # 5. Explore the target variable
    print("\nHOUSE PRICE STATISTICS")
    print(df[TARGET_COLUMN].describe())

    print("\nMEDIAN HOUSE PRICE")
    print(f"${df[TARGET_COLUMN].median():,.2f}")

    # 6. Price distribution
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="price",
        bins=50,
        kde=True
    )

    plt.title("Distribution of House Prices")
    plt.xlabel("House Price (USD)")
    plt.ylabel("Number of Houses")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "price_distribution.png",
        dpi=150
    )
    plt.close()

    # 7. House size versus price
    sample = df.sample(
        n=min(5000, len(df)),
        random_state=42
    )

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=sample,
        x="sqft_living",
        y="price",
        alpha=0.4,
        s=25
    )

    plt.title("Living Area vs House Price")
    plt.xlabel("Living Area (sq ft)")
    plt.ylabel("Price (USD)")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "size_vs_price.png",
        dpi=150
    )
    plt.close()

    # 8. Average price by bedrooms
    plt.figure(figsize=(10, 6))

    bedroom_prices = (
        df.groupby("bedrooms")["price"]
        .median()
        .sort_index()
    )

    sns.barplot(
        x=bedroom_prices.index,
        y=bedroom_prices.values
    )

    plt.title("Median House Price by Bedrooms")
    plt.xlabel("Bedrooms")
    plt.ylabel("Median Price (USD)")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "bedrooms_vs_price.png",
        dpi=150
    )
    plt.close()

    # 9. Top ZIP codes by median sale price
    zip_prices = (
        df.groupby("zipcode")["price"]
        .median()
        .sort_values(ascending=False)
        .head(12)
    )

    plt.figure(figsize=(10, 7))

    sns.barplot(
        x=zip_prices.values,
        y=zip_prices.index.astype(str),
        orient="h"
    )

    plt.title("12 Highest-Priced ZIP Codes")
    plt.xlabel("Median Price (USD)")
    plt.ylabel("ZIP Code")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "zip_price_comparison.png",
        dpi=150
    )
    plt.close()

    # 10. Correlation analysis
    correlation_columns = NUMERIC_FEATURES + ["price"]

    correlations = df[correlation_columns].corr()

    print("\nCORRELATIONS WITH PRICE")
    print(
        correlations["price"]
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(11, 8))

    sns.heatmap(
        correlations,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "correlation_heatmap.png",
        dpi=150
    )
    plt.close()

    print("\nEDA complete!")
    print(f"Charts saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
