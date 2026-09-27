"""
app/main.py

FastAPI app that wraps predict.py in an HTTP API.

Run from the project root:
    uvicorn app.main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger docs,
or POST to http://127.0.0.1:8000/predict with a JSON body matching
PredictionInput below.
"""

import os
import sys

# Make src/ importable regardless of where uvicorn is launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from predict import predict  # noqa: E402  (import after sys.path fix)

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Predicts Benign vs Malignant from tumor measurements.",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    """
    All 30 features the model was trained on. Field names with spaces
    in the original dataset (e.g. 'concave points_mean') use an alias
    so the JSON body can match the dataset column names exactly.
    """
    model_config = ConfigDict(populate_by_name=True)

    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float = Field(alias="concave points_mean")
    symmetry_mean: float
    fractal_dimension_mean: float

    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float = Field(alias="concave points_se")
    symmetry_se: float
    fractal_dimension_se: float

    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float = Field(alias="concave points_worst")
    symmetry_worst: float
    fractal_dimension_worst: float


class PredictionOutput(BaseModel):
    diagnosis: str
    label: int
    probability_malignant: float | None


@app.get("/")
def root():
    return {"status": "ok", "message": "Breast Cancer Classification API is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(payload: PredictionInput):
    """Predict Benign/Malignant from tumor measurements."""
    try:
        # by_alias=True so keys match the exact column names the model
        # (and its saved feature_names list) were trained on.
        input_dict = payload.model_dump(by_alias=True)
        result = predict(input_dict)
        return result
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model artifacts not found. Run `python src/train.py` first.",
        )
