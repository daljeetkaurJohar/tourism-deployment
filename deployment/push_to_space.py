from huggingface_hub import HfApi, upload_folder
import os

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "daljeetkaurJohar"   # <-- must match your HF username exactly
SPACE_NAME = "Tourism-Package-Prediction"  # <-- use the actual Space name
DEPLOY_DIR = "deployment"

if not os.path.isdir(DEPLOY_DIR):
    raise ValueError(f"Deployment folder does not exist: {DEPLOY_DIR}")

api = HfApi(token=HF_TOKEN)

# ✅ Create space with SDK
api.create_repo(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    space_sdk="streamlit",   # REQUIRED!
    exist_ok=True
)

# Upload code
upload_folder(
    repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
    repo_type="space",
    folder_path=DEPLOY_DIR,
    token=HF_TOKEN,
    commit_message="Deploy tourism purchase prediction app"
)

print(f"✅ Deployment pushed to Hugging Face Space: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")


