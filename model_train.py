import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("patient_snapshots.csv")

# Labels (change if needed)
LABELS = ["diabetes", "ckd", "heart_failure", "hypertension", "copd"]

X = df.drop(columns=LABELS)
Y = df[LABELS]

# ==============================
# 2. Train/Test Split
# ==============================
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ==============================
# 3. Define Models
# ==============================
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )
}

best_model = None
best_score = -1

# ==============================
# 4. Train & Evaluate
# ==============================
for name, base_model in models.items():
    print(f"\n🔹 Training & Evaluating {name}...")
    model = MultiOutputClassifier(base_model, n_jobs=-1)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Track macro F1-score across all diseases
    f1_scores = []

    for i, label in enumerate(LABELS):
        y_true = Y_test[label]
        y_pred = Y_pred[:, i]

        # Classification Report
        print(f"\n📌 {name} - {label}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred, digits=3))
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_scores.append(f1)

    avg_f1 = np.mean(f1_scores)
    print(f"\n✅ {name} - Average Macro F1 across diseases: {avg_f1:.3f}")

    # Save best model
    if avg_f1 > best_score:
        best_score = avg_f1
        best_model = model
        best_name = name

# ==============================
# 5. Save Best Model
# ==============================
filename = f"best_model_{best_name}.joblib"
joblib.dump(best_model, filename)
print(f"\n🎉 Best model is {best_name} with Avg Macro F1={best_score:.3f}")
print(f"💾 Saved as {filename}")
