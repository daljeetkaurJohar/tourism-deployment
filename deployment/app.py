import os, json, joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

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
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)

MODEL_REPO = get_secret("MODEL_REPO", "Daljeetk/tourism-best-model")
HF_TOKEN   = get_secret("HF_TOKEN", None)  # only needed if private repo

@st.cache_resource(show_spinner=True)
def load_artifacts(repo_id: str, token: str | None):
    cache_dir = os.environ.get("HF_HUB_CACHE", "/data/.cache/huggingface/hub")
    model_path = hf_hub_download(repo_id=repo_id, filename="model.joblib", token=token, cache_dir=cache_dir)
    meta_path  = hf_hub_download(repo_id=repo_id, filename="metadata.json", token=token, cache_dir=cache_dir)
    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return model, meta

try:
    model, meta = load_artifacts(MODEL_REPO, HF_TOKEN)
except Exception as e:
    st.error(f"Failed to load model artifacts from {MODEL_REPO}. Details: {e}")
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
inputs["CustomerID"] = st.sidebar.text_input("CustomerID", "200000") # Handles numeric IDs like 200000 fine
inputs["Age"] = i_num("Age", 40, 18, 32)
inputs["TypeofContact"] = st.sidebar.selectbox("TypeofContact", ["Company Invited","Self Inquiry"])

# Explicitly map CityTier if it was treated numerically during training
city_tier_mapping = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
selected_city_tier_str = st.sidebar.selectbox("CityTier", ["Tier 1","Tier 2","Tier 3"])
inputs["CityTier"] = city_tier_mapping[selected_city_tier_str]

inputs["Occupation"] = st.sidebar.selectbox("Occupation", ["Salaried","Freelancer","Small Business","Large Business"])
inputs["Gender"] = st.sidebar.selectbox("Gender", ["Male","Female"])
inputs["NumberOfPersonVisiting"] = i_num("NumberOfPersonVisiting", 1, 2, 3,4 , 5)
inputs["PreferredPropertyStar"] = i_num("PreferredPropertyStar", 2, 3,4, 5)
inputs["MaritalStatus"] = st.sidebar.selectbox("MaritalStatus", ["Single","Married","Divorced","Unmarried"])
inputs["NumberOfTrips"] = i_num("NumberOfTrips", 1, 4, 5)
inputs["Passport"] = st.sidebar.selectbox("Passport", [0,1]) # Assumes 0 for No, 1 for Yes
inputs["OwnCar"] = st.sidebar.selectbox("OwnCar", [0,1])     # Assumes 0 for No, 1 for Yes
inputs["NumberOfChildrenVisiting"] = i_num("NumberOfChildrenVisiting", 0, 1, 2)
inputs["Designation"] = st.sidebar.selectbox("Designation", ["Executive","Manager","Senior Manager","AVP","VP"])
inputs["MonthlyIncome"] = i_num("MonthlyIncome", 300000, 400000, 15000, 10000)
inputs["PitchSatisfactionScore"] = i_num("PitchSatisfactionScore", 2, 3, 5)
inputs["ProductPitched"] = st.sidebar.selectbox("ProductPitched", ["Basic","Deluxe","Super Deluxe","King","Standard"])
inputs["NumberOfFollowups"] = i_num("NumberOfFollowups", 2, 1, 3)
inputs["DurationOfPitch"] = i_num("DurationOfPitch", 6,3,20)

df_in = pd.DataFrame([inputs])

# Drop junk columns if any (CustomerID is often treated as non-feature for prediction)
for junk in ["Unnamed: 0", "index", "CustomerID"]: # Added CustomerID here if it's not a feature
    if junk in df_in.columns:
        df_in = df_in.drop(columns=[junk])

# Respect training feature order
feature_order = meta.get("feature_order")
if feature_order:
    # Ensure all categorical columns are one-hot encoded
    df_in = pd.get_dummies(df_in)
    
    missing_cols = set(feature_order) - set(df_in.columns)
    for c in missing_cols:
        df_in[c] = 0 # Add missing one-hot encoded columns with 0

    # Align columns and fill any new columns (from new categories) with 0
    df_in = df_in.reindex(columns=feature_order, fill_value=0)
    
    # Check if there are any extra columns in df_in not present in feature_order
    extra_cols = set(df_in.columns) - set(feature_order)
    if extra_cols:
        st.warning(f"Input contains unexpected columns not in training data: {list(extra_cols)}. These will be ignored.")
        df_in = df_in.drop(columns=list(extra_cols))
    
    missing = [c for c in feature_order if c not in df_in.columns]
    if missing:
        st.warning(f"Input still missing expected columns after one-hot encoding: {missing}. This might impact prediction accuracy.")


# =============================
# Prediction
# =============================
threshold = float(meta.get("threshold", 0.5))
if st.button("Predict"):
    try:
        proba = float(model.predict_proba(df_in)[:, 1][0])
        pred = int(proba >= threshold)
        st.metric("Purchase Probability", f"{proba:.3f}")
        st.write("Prediction:", "Will Purchase (1)" if pred == 1 else "Will Not Purchase (0)")
        with st.expander("Input snapshot"):
            st.dataframe(df_in)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.dataframe(df_in) # Display dataframe on error for debugging

# =============================
# Optional: Model Performance Comparison (if results in metadata)
# =============================
if "results" in meta:
    st.subheader("📊 Model Performance Comparison")
    df_results = pd.DataFrame(meta["results"]).T
    df_results = df_results[['accuracy', 'precision', 'recall', 'f1_score']]
    st.bar_chart(df_results)

