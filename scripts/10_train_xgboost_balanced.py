import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = BASE_DIR / "features" / "kmer_features_k5_balanced.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "xgboost_k5_balanced_model.pkl"
FEATURE_COLUMNS_FILE = MODEL_DIR / "k5_balanced_feature_columns.pkl"
RESULTS_FILE = MODEL_DIR / "xgboost_k5_balanced_results.txt"


df = pd.read_csv(FEATURE_FILE)

print("Dataset loaded:")
print(df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())

X = df.drop(columns=["accession", "label", "sequence_length"])
y = df["label"].astype(int)

joblib.dump(list(X.columns), FEATURE_COLUMNS_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("\nTraining balanced XGBoost model...")
model.fit(X_train, y_train, sample_weight=sample_weights)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

results = f"""
Balanced XGBoost Probiotic Genome Classifier
============================================

Feature file:
{FEATURE_FILE}

Total samples:
{len(df)}

Training samples:
{len(X_train)}

Testing samples:
{len(X_test)}

Label meaning:
1 = Probiotic
0 = Non-probiotic / pathogenic

Label counts:
{df["label"].value_counts().to_string()}

Confusion Matrix:
TN: {tn}
FP: {fp}
FN: {fn}
TP: {tp}

Accuracy:    {accuracy:.4f}
Precision:   {precision:.4f}
Recall:      {recall:.4f}
Specificity: {specificity:.4f}
F1 Score:    {f1:.4f}

Classification Report:
{classification_report(y_test, y_pred, zero_division=0)}
"""

print(results)

with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write(results)

joblib.dump(model, MODEL_FILE)

print("Model saved to:", MODEL_FILE)
print("Feature columns saved to:", FEATURE_COLUMNS_FILE)
print("Results saved to:", RESULTS_FILE)