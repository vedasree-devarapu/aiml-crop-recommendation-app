"""Train and evaluate a crop recommendation classifier.

Expected CSV: Crop_recommendation.csv, with feature columns such as N, P, K,
temperature, humidity, ph, rainfall, and a target column named ``label``.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_FILE = Path(__file__).with_name("Crop_recommendation.csv")
MODEL_FILE = Path(__file__).with_name("crop_recommendation_model.joblib")
TARGET_COLUMN = "label"


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}. Place Crop_recommendation.csv "
            "in the same directory as this script."
        )

    data = pd.read_csv(DATA_FILE)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Dataset must contain a '{TARGET_COLUMN}' column.")
    if data.empty:
        raise ValueError("The dataset is empty.")

    data = data.dropna(subset=[TARGET_COLUMN])
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    if features.empty or not all(pd.api.types.is_numeric_dtype(features[c]) for c in features):
        raise ValueError("All feature columns must be numeric, and at least one is required.")

    stratify = target if target.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    predictions = model.predict(X_test)
    print(f"Training accuracy: {accuracy_score(y_train, train_predictions):.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save the trained model together with the feature order used during training.
    import joblib

    joblib.dump({"model": model, "features": list(features.columns)}, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    main()