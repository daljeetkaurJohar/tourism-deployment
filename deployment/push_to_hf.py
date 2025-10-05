from huggingface_hub import HfApi, upload_folder
import os

# Read Hugging Face token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "Daljeetk"
SPACE_NAME = "Tourism-Package-Prediction"

# Deployment folder on Colab
DEPLOY_DIR = "/content/tourism_project/deployment"

# Make sure folder exists
if not os.path.isdir(DEPLOY_DIR):
    raise ValueError(f"Deployment folder does not exist: {DEPLOY_DIR}")

api = HfApi(token=HF_TOKEN)

# Check if the space exists
spaces = [space.id for space in api.list_spaces(author=HF_USERNAME)]
if f"{HF_USERNAME}/{SPACE_NAME}" not in spaces:
    print(f"⚠ Space '{SPACE_NAME}' not found under user '{HF_USERNAME}'.")
    print("Please create it manually at https://huggingface.co/spaces")
else:
    # Upload deployment folder
    upload_folder(
        repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
        repo_type="space",
        folder_path=DEPLOY_DIR,
        token=HF_TOKEN,
        commit_message="Deploy tourism purchase prediction app"
    )
    print(f"✅ Deployment pushed to Hugging Face Space: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
