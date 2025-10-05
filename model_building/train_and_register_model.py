# train_and_register_model.py
import os
import joblib
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from huggingface_hub import HfApi, upload_folder

HF_USERNAME = "Daljeetk"
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-best-model"
TARGET = "ProdTaken"

# Load datasets
train_dataset = load_dataset(f"{HF_USERNAME}/tourism-train-dataset", split="train", token=HF_TOKEN)
test_dataset  = load_dataset(f"{HF_USERNAME}/tourism-test-dataset", split="train", token=HF_TOKEN)

train_df = pd.DataFrame(train_dataset)
test_df  = pd.DataFrame(test_dataset)

X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
X_test, y_test   = test_df.drop(columns=[TARGET]), test_df[TARGET]

# One-hot encode categorical
X_train = pd.get_dummies(X_train)
X_test  = pd.get_dummies(X_test).reindex(columns=X_train.columns, fill_value=0)

# Models & hyperparameters
models = {
    "DecisionTree": {"model": DecisionTreeClassifier(random_state=42), "params": {"max_depth": [None, 5, 10], "min_samples_split": [2,5,10]}},
    "Bagging": {"model": BaggingClassifier(random_state=42), "params": {"n_estimators": [10,50,100], "max_samples":[0.5,1.0]}},
    "RandomForest": {"model": RandomForestClassifier(random_state=42), "params":{"n_estimators":[50,100], "max_depth":[None,5,10]}},
    "AdaBoost": {"model": AdaBoostClassifier(random_state=42), "params":{"n_estimators":[50,100],"learning_rate":[0.5,1.0]}},
    "GradientBoosting": {"model": GradientBoostingClassifier(random_state=42), "params":{"n_estimators":[50,100],"learning_rate":[0.05,0.1],"max_depth":[3,5]}},
    "XGBoost": {"model": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42), "params":{"n_estimators":[50,100],"max_depth":[3,5],"learning_rate":[0.05,0.1]}}
}

results = {}
best_models = {}

for name, m in models.items():
    print(f"🔹 Training {name}...")
    grid = GridSearchCV(m["model"], m["params"], cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1, "best_params": grid.best_params_}
    best_models[name] = best_model
    print(classification_report(y_test, y_pred))
    print(f"Best params: {grid.best_params_}")

# Save results
eval_path = os.path.join("model_building", "evaluation_results.json")
os.makedirs(os.path.dirname(eval_path), exist_ok=True)
with open(eval_path, "w") as f:
    json.dump(results, f, indent=4)

# Select best model
best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = best_models[best_model_name]
model_dir = f"best_{best_model_name.lower()}_model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(best_model, os.path.join(model_dir, "model.joblib"))

# README
with open(os.path.join(model_dir, "README.md"), "w") as f:
    f.write(f"# {best_model_name} for Tourism Purchase Prediction\n"
            f"- Accuracy: {results[best_model_name]['accuracy']:.4f}\n"
            f"- Precision: {results[best_model_name]['precision']:.4f}\n"
            f"- Recall: {results[best_model_name]['recall']:.4f}\n"
            f"- F1 Score: {results[best_model_name]['f1_score']:.4f}\n"
            f"- Best Params: {results[best_model_name]['best_params']}")

# Upload to Hugging Face
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
upload_folder(repo_id=MODEL_REPO_ID, repo_type="model", folder_path=model_dir, token=HF_TOKEN, commit_message=f"Upload best model ({best_model_name})")

print(f"✅ Best model ({best_model_name}) uploaded to {MODEL_REPO_ID}")
