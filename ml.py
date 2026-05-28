import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

df = pd.read_csv("cleaned_ecommerce_data.csv")

drop_cols = [
    'customer_id',
    'session_id',
    'review_text'
]

existing_drop_cols = [
    col for col in drop_cols
    if col in df.columns
]

df.drop(columns=existing_drop_cols, inplace=True)

target = "purchased"

X = df.drop(columns=[target])

y = df[target]

X = X.select_dtypes(include=np.number)

if X.shape[1] > 100:
    X = X.iloc[:, :100]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=500,
        n_jobs=1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        random_state=42,
        n_jobs=1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
}

results = []

for name, model in models.items():

    print("=" * 60)
    print(f"TRAINING {name}")
    print("=" * 60)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    results.append({

        "Model": name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred
        ),

        "Recall": recall_score(
            y_test,
            y_pred
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_prob
        )
    })

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)

print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df)