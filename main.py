"""
    Containe the API which ,
    Hosts the Endpoints for inference runs on the model

    Date : 28-09-2026
    Author : Sourodeep Banerjee
"""

from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field
from starter.ml.data import process_data
from starter.ml.model import inference


ROOT = Path(__file__).resolve().parent
ARTIFACTS = joblib.load(ROOT / "model" / "artifacts.joblib")

app = FastAPI(title="Census Salary Prediction API")


class CensusInput(BaseModel):
    """Validate one Census record submitted for prediction.

    Input:
        A JSON object with the Census feature names. Hyphenated JSON keys are
        mapped to Python-safe attributes with Pydantic aliases.

    Output:
        A validated CensusInput instance.
    """
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 171150,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Married-civ-spouse",
                "occupation": "Sales",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 99999,
                "capital-loss": 0,
                "hours-per-week": 60,
                "native-country": "United-States",
            }
        }
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


class PredictionResponse(BaseModel):
    """Represent the salary prediction returned by the API.

    Input:
        A predicted salary label as a string.

    Output:
        A JSON object with the `prediction` field.
    """
    prediction: str


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a welcome message.

    Input:
        None.

    Output:
        A dictionary containing the API welcome message.
    """
    return {"message": "Welcome to the Census Salary Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CensusInput) -> PredictionResponse:
    """Predict the salary class for one Census record.

    Input:
        payload: A validated CensusInput request body.

    Output:
        A PredictionResponse containing the predicted salary label.
    """
    row: dict[str: Any] = payload.model_dump(by_alias=True)
    features = pd.DataFrame([row])

    processed, _, _, _ = process_data(
        features,
        categorical_features=ARTIFACTS["categorical_features"],
        training=False,
        encoder=ARTIFACTS["encoder"],
        lb=ARTIFACTS["lb"],
    )

    predicted_class = inference(ARTIFACTS["model"], processed)
    salary = ARTIFACTS["lb"].inverse_transform(predicted_class)[0]

    return PredictionResponse(prediction=str(salary))
