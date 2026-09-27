# Breast Cancer Classification Using Machine Learning

An end-to-end ML project that predicts whether a breast tumor is **Benign** or **Malignant** from diagnostic measurements (the Wisconsin Breast Cancer dataset), served through a REST API with a Streamlit UI on top.

## Problem Statement

Given 30 numeric features describing cell nuclei from a digitized image of a breast mass (radius, texture, perimeter, area, smoothness, etc., each reported as a mean, standard error, and "worst" value), predict the diagnosis: Benign (0) or Malignant (1).

## Architecture

```
Raw data (CSV)
      │
      ▼
notebooks/  ──►  EDA, feature exploration, initial model trials
      │
      ▼
src/preprocessing.py  ──►  clean data, encode target, split, scale
      │
      ▼
src/train.py  ──►  train Logistic Regression + Random Forest,
                    evaluate (accuracy, precision/recall, ROC-AUC, CV),
                    save best model + scaler to models/
      │
      ▼
src/predict.py  ──►  load saved model/scaler, run inference on new input
      │
      ▼
app/main.py (FastAPI)  ──►  /predict REST endpoint, Swagger docs at /docs
      │
      ▼
ui/streamlit_app.py  ──►  interactive form → calls the API → shows result
```

## Project Structure

```
project/
├── data/                   # dataset (data.csv)
├── notebooks/              # EDA and model exploration notebook
├── src/
│   ├── preprocessing.py    # load, clean, split, scale data
│   ├── train.py            # train + evaluate + save best model
│   └── predict.py          # load model, run predictions
├── app/
│   └── main.py             # FastAPI app exposing /predict
├── ui/
│   └── streamlit_app.py    # Streamlit front-end for the API
├── models/                 # saved model.pkl, scaler.pkl, feature_names.pkl
└── requirements.txt
```

## Models & Evaluation

Two models were trained and compared on a stratified 80/20 train/test split (features standardized with `StandardScaler`):

| Model               | Metric focus                                   |
|---------------------|-------------------------------------------------|
| Logistic Regression | Accuracy, precision/recall, ROC-AUC             |
| Random Forest       | Accuracy, precision/recall, ROC-AUC, 5-fold CV  |

Because this is a diagnostic task, **recall on the malignant class** matters more than raw accuracy — a false negative (missing a malignant tumor) is far costlier than a false positive. The best-performing model on the test set is automatically saved to `models/model.pkl`.

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (creates models/model.pkl, scaler.pkl, feature_names.pkl)
```bash
python src/train.py
```

### 3. Start the API
```bash
python -m uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive Swagger docs.

### 4. Start the UI (in a separate terminal, with the API still running)
```bash
streamlit run ui/streamlit_app.py
```
Opens in your browser — enter measurements (or use the pre-filled defaults) and click **Predict**.

## Example API Request

```bash
POST /predict
Content-Type: application/json

{
  "radius_mean": 17.99,
  "texture_mean": 10.38,
  ...
}
```

Response:
```json
{
  "diagnosis": "Malignant",
  "label": 1,
  "probability_malignant": 0.94
}
```

## Tech Stack

- **Data/ML**: pandas, scikit-learn, joblib
- **API**: FastAPI, Uvicorn
- **UI**: Streamlit
- **EDA**: matplotlib, seaborn

## Possible Next Steps

- Containerize with Docker for consistent environments
- Deploy the API + UI (Render, Railway, Hugging Face Spaces, or Streamlit Cloud) for a public demo link
- Add automated tests (`tests/`) for preprocessing and prediction functions
- Add structured logging for inference requests
