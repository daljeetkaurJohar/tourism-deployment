# register_dataset.py
import os
from huggingface_hub import HfApi

# --- Hugging Face config ---
HF_USERNAME = "Daljeetk"
HF_DATASET_NAME = "tourism-customer-purchase-prediction"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID_DATA = f"{HF_USERNAME}/{HF_DATASET_NAME}"

# --- Local file ---
DATA_FOLDER = os.path.join(os.getcwd(), "data")
SOURCE_CSV_PATH = os.path.join(DATA_FOLDER, "tourism.csv")

# --- Create repo and upload dataset ---
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_REPO_ID_DATA, repo_type="dataset", exist_ok=True)

if os.path.exists(SOURCE_CSV_PATH):
    api.upload_file(
        path_or_fileobj=SOURCE_CSV_PATH,
        path_in_repo="tourism.csv",
        repo_id=HF_REPO_ID_DATA,
        repo_type="dataset",
        commit_message="Initial upload of tourism.csv"
    )
    print(f"✅ {SOURCE_CSV_PATH} uploaded to {HF_REPO_ID_DATA}")
else:
    print(f"❌ File not found: {SOURCE_CSV_PATH}")
