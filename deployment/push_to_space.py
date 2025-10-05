from huggingface_hub import HfApi, upload_folder
import os

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "Daljeetk"  # make sure this matches EXACT username
SPACE_NAME = "Tourism-Package-Prediction"
DEPLOY_DIR = "deployment"

if not os.path.isdir(DEPLOY_DIR):
    raise ValueError(f"Deployment folder does not exist: {DEPLOY_DIR}")

api = HfApi(token=HF_TOKEN)

#  Only create repo if it does not already exist
try:
    api.create_repo(
        repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
        repo_type="space",
        space_sdk="streamlit",   # lowercase
        exist_ok=True
    )
except Exception as e:
    print(f" Repo may already exist: {e}")

# Upload files
upload_folder(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    folder_path=DEPLOY_DIR,
    token=HF_TOKEN,
    commit_message="Deploy tourism purchase prediction app"
)

print(f" Deployment pushed: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")

