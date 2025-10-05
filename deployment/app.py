import os
import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
from typing import Optional

# =============================
# Streamlit Page Config
# =============================
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="👜",
    layout="centered"
)
st.title("👜 Tourism Package Purchase Prediction")

# =============================
# Helpers
# =============================
def get_secret(name: str, default=None):
    """Fetches a secret from Streamlit secrets or environment variables."""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

MODEL_REPO = get_secret("MODEL_REPO", "Daljeetk/tourism-best-model")
HF_TOKEN = get_secret("HF_TOKEN", None)  # only needed if private repo

@st.cache_resource(show_spinner=True)
def load_artifacts(repo_id: str, token: Optional[str]):
    """Loads the model and metadata from Hugging Face Hub."""
    cache_dir = os.environ.get("HF_HUB_CACHE", None)
    try:
        model_path = hf_hub_download(repo_id=repo_id, filename="model.joblib", token=token, cache_dir=cache_dir)
        meta_path = hf_hub_download(repo_id=repo_id, filename="metadata.json", token=token, cache_dir=cache_dir)
        model = joblib.load(model_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return model, meta
    except Exception as e:
        st.error(f"Failed to load model artifacts from {repo_id}. Details: {e}")
        st.stop()

try:
    model, meta = load_artifacts(MODEL_REPO, HF_TOKEN)
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

st.caption("Model metrics (from training)")
st.json(meta.get("metrics", {}))

# =============================
# Sidebar - Customer Input
# =============================
st.sidebar.header("Enter Customer Profile")

def i_num(label, value, minv=None, maxv=None, step=1):
    return st.sidebar.number_input(label, value=value, min_value=minv, max_value=maxv, step=step)

inputs = {}
inputs["CustomerID"] = st.sidebar.text_input("CustomerID", "200000")
inputs["Age"] = i_num("Age", 32, 18, 90)
inputs["TypeofContact"] = st.sidebar.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])

city_tier_mapping = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
selected_city_tier_str = st.sidebar.selectbox("CityTier", ["Tier 1", "Tier 2", "Tier 3"])
inputs["CityTier"] = city_tier_mapping[selected_city_tier_str]

inputs["Occupation"] = st.sidebar.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
inputs["Gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
inputs["NumberOfPersonVisiting"] = i_num("NumberOfPersonVisiting", 2, 0, 10)
inputs["PreferredPropertyStar"] = i_num("PreferredPropertyStar", 3, 1, 5)
inputs["MaritalStatus"] = st.sidebar.selectbox("MaritalStatus", ["Single", "Married", "Divorced", "Unmarried"])
inputs["NumberOfTrips"] = i_num("NumberOfTrips", 1, 0, 50)
inputs["Passport"] = st.sidebar.selectbox("Passport", [0, 1])
inputs["OwnCar"] = st.sidebar.selectbox("OwnCar", [0, 1])
inputs["NumberOfChildrenVisiting"] = i_num("NumberOfChildrenVisiting", 0, 0, 10)
inputs["Designation"] = st.sidebar.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
inputs["MonthlyIncome"] = i_num("MonthlyIncome", 70000, 0, 1_000_000, 1000)
inputs["PitchSatisfactionScore"] = i_num("PitchSatisfactionScore", 3, 1, 5)
inputs["ProductPitched"] = st.sidebar.selectbox("ProductPitched", ["Basic", "Deluxe", "Super Deluxe", "King", "Standard"])
inputs["NumberOfFollowups"] = i_num("NumberOfFollowups", 1, 0, 20)
inputs["DurationOfPitch"] = i_num("DurationOfPitch", 10, 0, 120)

# Create DataFrame from inputs
df_in = pd.DataFrame([inputs])

# Drop non-feature columns
for junk in ["Unnamed: 0", "index"]:
    if junk in df_in.columns:
        df_in = df_in.drop(columns=[junk])

# =============================
# Align features to training feature order (the fix)
# =============================
feature_order = meta.get("feature_order")
if feature_order:
    # One-hot encode the input DataFrame
    df_in = pd.get_dummies(df_in)
    
    # Add missing columns with 0
    for c in feature_order:
        if c not in df_in.columns:
            df_in[c] = 0
    
    # Remove unexpected columns
    extra_cols = set(df_in.columns) - set(feature_order)
    if extra_cols:
        df_in = df_in.drop(columns=list(extra_cols))
    
    # Reorder columns exactly
    df_in = df_in[feature_order]
