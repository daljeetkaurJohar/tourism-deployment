
from huggingface_hub import HfApi, upload_folder
import os

# Read token from environment variable (do NOT hardcode)
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "Daljeetk"
SPACE_NAME = "tourism-deployment"
DEPLOY_DIR = "tourism_project/deployment"

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=f"{HF_USERNAME}/{SPACE_NAME}", repo_type="space", space_sdk="streamlit",exist_ok=True)

upload_folder(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    folder_path=DEPLOY_DIR,
    token=HF_TOKEN,
    commit_message="Deploy tourism purchase prediction app"
)

print(f"✅ Deployment pushed to Hugging Face Space: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
