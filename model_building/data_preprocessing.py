# data_preprocessing.py
import os
from huggingface_hub import HfApi
from datasets import load_dataset

HF_USERNAME = "Daljeetk"
HF_TOKEN = os.getenv("HF_TOKEN")

DATA_FOLDER = os.path.join(os.getcwd(), "data")
os.makedirs(DATA_FOLDER, exist_ok=True)

HF_RAW_REPO = f"{HF_USERNAME}/tourism-customer-purchase-prediction"
HF_TRAIN_REPO = f"{HF_USERNAME}/tourism-train-dataset"
HF_TEST_REPO = f"{HF_USERNAME}/tourism-test-dataset"

TRAIN_CSV_PATH = os.path.join(DATA_FOLDER, "train_dataset.csv")
TEST_CSV_PATH = os.path.join(DATA_FOLDER, "test_dataset.csv")

# Load raw dataset
dataset = load_dataset(HF_RAW_REPO, token=HF_TOKEN)

# Remove Unnamed columns
columns_to_remove = [c for c in dataset['train'].column_names if c.lower().startswith("unnamed")]
if columns_to_remove:
    dataset = dataset.remove_columns(columns_to_remove)

# Filter valid records
def clean_record(record):
    return record["CustomerID"] is not None and record["ProdTaken"] is not None

dataset = dataset.filter(clean_record)

# Split 80/20
split_dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# Save CSVs locally
train_dataset.to_csv(TRAIN_CSV_PATH, index=False)
test_dataset.to_csv(TEST_CSV_PATH, index=False)

# Upload to Hugging Face
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_TRAIN_REPO, repo_type="dataset", private=True, exist_ok=True)
api.create_repo(repo_id=HF_TEST_REPO, repo_type="dataset", private=True, exist_ok=True)

api.upload_file(
    path_or_fileobj=TRAIN_CSV_PATH,
    path_in_repo="train_dataset.csv",
    repo_id=HF_TRAIN_REPO,
    repo_type="dataset",
    commit_message="Upload train dataset"
)

api.upload_file(
    path_or_fileobj=TEST_CSV_PATH,
    path_in_repo="test_dataset.csv",
    repo_id=HF_TEST_REPO,
    repo_type="dataset",
    commit_message="Upload test dataset"
)

print("✅ Train and test datasets uploaded successfully")

