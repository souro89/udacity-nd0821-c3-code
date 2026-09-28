# Train, evaluate, and save the Census salary model.

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    evaluate_slices,
    inference,
    train_model,
)


ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
LABEL = "salary"


def main():
    data = pd.read_csv(ROOT / "data" / "census.csv", skipinitialspace=True)
    for column in data.select_dtypes(include="object"):
        data[column] = data[column].str.strip()

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data[LABEL],
    )
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CATEGORICAL_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, predictions)
    slices = evaluate_slices(
        model, X_test, y_test, test, CATEGORICAL_FEATURES
    )

    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "encoder": encoder,
            "lb": lb,
            "categorical_features": CATEGORICAL_FEATURES,
            "label": LABEL,
        },
        model_dir / "artifacts.joblib",
    )

    report = {
        "evaluation_rows": len(test),
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "categorical_slices": slices,
    }
    (model_dir / "evaluation_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
