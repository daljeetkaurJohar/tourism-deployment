from huggingface_hub import HfApi, upload_folder
import os

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "daljeetkaurJohar"
SPACE_NAME = "Tourism-Package-Prediction"   # use the same space name as on HF
DEPLOY_DIR = "deployment"

if not os.path.isdir(DEPLOY_DIR):
    raise ValueError(f"Deployment folder does not exist: {DEPLOY_DIR}")

api = HfApi(token=HF_TOKEN)

#  SDK value
api.create_repo(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    space_sdk="streamlit",   # MUST be lowercase
    exist_ok=True
)

upload_folder(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    folder_path=DEPLOY_DIR,
    token=HF_TOKEN,
    commit_message="Deploy tourism purchase prediction app"
)

print(f"✅ Deployment pushed to Hugging Face Space: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
