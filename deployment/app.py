
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

MODEL_REPO = "Daljeetk/tourism-best-model"
MODEL_FILE = "model.joblib"

# Load model from Hugging Face Hub
local_model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(local_model_path)

st.title("Tourism Package Purchase Prediction")
st.write("Predict whether a customer will purchase a tourism package.")

# Define input fields
user_input = {}
user_input["CustomerID"] = st.text_input("CustomerID", "")
user_input["Age"] = st.number_input("Age", min_value=0, max_value=120, value=30)
user_input["TypeofContact"] = st.text_input("Type of Contact", "")
user_input["CityTier"] = st.number_input("City Tier", min_value=1, max_value=3, value=1)
user_input["DurationOfPitch"] = st.number_input("Duration of Pitch", min_value=0, value=0)
user_input["Occupation"] = st.text_input("Occupation", "")
user_input["Gender"] = st.text_input("Gender", "")
user_input["NumberOfPersonVisiting"] = st.number_input("Number of Persons Visiting", min_value=0, value=1)
user_input["NumberOfFollowups"] = st.number_input("Number of Followups", min_value=0, value=0)
user_input["ProductPitched"] = st.text_input("Product Pitched", "")
user_input["PreferredPropertyStar"] = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3)
user_input["MaritalStatus"] = st.text_input("Marital Status", "")
user_input["NumberOfTrips"] = st.number_input("Number of Trips", min_value=0, value=1)
user_input["Passport"] = st.text_input("Passport", "")
user_input["PitchSatisfactionScore"] = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=5)
user_input["OwnCar"] = st.text_input("Own Car (Yes/No)", "")
user_input["NumberOfChildrenVisiting"] = st.number_input("Number of Children Visiting", min_value=0, value=0)
user_input["Designation"] = st.text_input("Designation", "")
user_input["MonthlyIncome"] = st.number_input("Monthly Income", min_value=0, value=30000)

# Preprocess input
def preprocess_input(data_dict):
    df = pd.DataFrame([data_dict])
    df = pd.get_dummies(df)
    return df

# Prediction
if st.button("Predict Purchase"):
    df_input = preprocess_input(user_input)
    prediction = model.predict(df_input)
    st.success(f"Predicted Purchase: {int(prediction[0])}")

