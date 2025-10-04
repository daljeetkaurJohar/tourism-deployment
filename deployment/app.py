
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "Daljeetk/tourism-best-model"
MODEL_FILE = "model.joblib"

local_model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(local_model_path)

st.title("Tourism Package Purchase Prediction")
st.write("Predict whether a customer will purchase a tourism package.")

input_features = [
    "CustomerID", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

user_input = {}
for feature in input_features:
    user_input[feature] = st.number_input(feature, value=0)

def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df)
    return df

if st.button("Predict Purchase"):
    df_input = preprocess_input(user_input)
    prediction = model.predict(df_input)
    st.success(f"Predicted Purchase: {int(prediction[0])}")
