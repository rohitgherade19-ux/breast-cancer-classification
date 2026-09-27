"""
ui/streamlit_app.py

A simple Streamlit front-end for the Breast Cancer Classification API.
It does NOT load the model directly - it calls the FastAPI /predict
endpoint over HTTP, so make sure the API is running first.

Run:
    1) In one terminal: uvicorn app.main:app --reload
    2) In another:      streamlit run ui/streamlit_app.py
"""

import requests
import streamlit as st

st.set_page_config(page_title="Breast Cancer Classifier", page_icon="🩺", layout="centered")

# Reasonable starting values (approx. dataset-wide means) so the form
# is usable immediately without the user having to know real values.
DEFAULTS = {
    "radius_mean": 14.13, "texture_mean": 19.29, "perimeter_mean": 91.97,
    "area_mean": 654.89, "smoothness_mean": 0.0964, "compactness_mean": 0.1043,
    "concave points_mean": 0.0489, "concavity_mean": 0.0888,
    "symmetry_mean": 0.1812, "fractal_dimension_mean": 0.0628,
    "radius_se": 0.4052, "texture_se": 1.2169, "perimeter_se": 2.8661,
    "area_se": 40.34, "smoothness_se": 0.0070, "compactness_se": 0.0254,
    "concavity_se": 0.0319, "concave points_se": 0.0118,
    "symmetry_se": 0.0205, "fractal_dimension_se": 0.0038,
    "radius_worst": 16.27, "texture_worst": 25.68, "perimeter_worst": 107.26,
    "area_worst": 880.58, "smoothness_worst": 0.1324, "compactness_worst": 0.2543,
    "concavity_worst": 0.2722, "concave points_worst": 0.1146,
    "symmetry_worst": 0.2901, "fractal_dimension_worst": 0.0839,
}

MEAN_FEATURES = [k for k in DEFAULTS if k.endswith("_mean")]
SE_FEATURES = [k for k in DEFAULTS if k.endswith("_se")]
WORST_FEATURES = [k for k in DEFAULTS if k.endswith("_worst")]


def number_inputs(feature_list, columns=2):
    """Render number_input widgets for a list of features in a grid,
    return a dict of feature_name -> value."""
    values = {}
    cols = st.columns(columns)
    for i, feature in enumerate(feature_list):
        with cols[i % columns]:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=float(DEFAULTS[feature]),
                format="%.5f",
                key=feature,
            )
    return values


st.title("🩺 Breast Cancer Classifier")
st.caption("Enter tumor measurements below and get a Benign / Malignant prediction from the trained model.")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API base URL", value="http://127.0.0.1:8000")
    st.caption("Make sure the FastAPI server is running (`uvicorn app.main:app --reload`).")

tab_mean, tab_se, tab_worst = st.tabs(["Mean values", "Standard Error", "Worst values"])

all_values = {}
with tab_mean:
    all_values.update(number_inputs(MEAN_FEATURES))
with tab_se:
    all_values.update(number_inputs(SE_FEATURES))
with tab_worst:
    all_values.update(number_inputs(WORST_FEATURES))

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    try:
        response = requests.post(f"{api_url}/predict", json=all_values, timeout=10)
        response.raise_for_status()
        result = response.json()

        diagnosis = result["diagnosis"]
        prob = result.get("probability_malignant")

        if diagnosis == "Malignant":
            st.error(f"### Prediction: {diagnosis}")
        else:
            st.success(f"### Prediction: {diagnosis}")

        if prob is not None:
            st.metric("Probability of Malignancy", f"{prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not reach the API. Make sure it's running:\n\n"
            "`uvicorn app.main:app --reload`"
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e.response.text}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
