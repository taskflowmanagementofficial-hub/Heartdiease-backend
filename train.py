"""
train.py — Train ML model on heart_2020_cleaned.csv
Usage: python backend/train.py
"""
import os, sys, joblib, json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

BASE  = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(BASE, "data", "heart_disease.csv")
MODEL = os.path.join(BASE, "model.pkl")

# ── column groups ────────────────────────────────────────────────────────────
NUMERIC_COLS = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

BINARY_COLS  = [
    "Smoking", "AlcoholDrinking", "Stroke",
    "DiffWalking", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer",
]

ORDERED_CATS = {
    "Sex":         ["Female", "Male"],
    "AgeCategory": ["18-24","25-29","30-34","35-39","40-44","45-49",
                    "50-54","55-59","60-64","65-69","70-74","75-79","80 or older"],
    "Race":        ["White","Black","Asian","American Indian/Alaskan Native","Other","Hispanic"],
    "Diabetic":    ["No","No, borderline diabetes","Yes (during pregnancy)","Yes"],
    "GenHealth":   ["Poor","Fair","Good","Very good","Excellent"],
}
TARGET = "HeartDisease"


def load(path=DATA):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.dropna(subset=[TARGET], inplace=True)
    # Normalise Yes/No target
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return df


def build_preprocessor():
    num_t  = StandardScaler()
    bin_t  = OrdinalEncoder(categories=[["No","Yes"]]*len(BINARY_COLS),
                            handle_unknown="use_encoded_value", unknown_value=-1)
    cat_t  = OrdinalEncoder(categories=list(ORDERED_CATS.values()),
                            handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(transformers=[
        ("num", num_t,  NUMERIC_COLS),
        ("bin", bin_t,  BINARY_COLS),
        ("cat", cat_t,  list(ORDERED_CATS.keys())),
    ], remainder="drop")


def train(data_path=DATA, model_path=MODEL):
    print(f"\n📂 Loading: {data_path}")
    df = load(data_path)
    print(f"   Rows: {len(df):,}  |  Heart Disease Yes: {df[TARGET].sum():,}")

    feat_cols = NUMERIC_COLS + BINARY_COLS + list(ORDERED_CATS.keys())
    X = df[feat_cols]
    y = df[TARGET]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    prep = build_preprocessor()

    candidates = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=5,
            subsample=0.8, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_split=4,
            class_weight="balanced", random_state=42, n_jobs=-1),
        "NaiveBayes": GaussianNB(),
    }

    best_name, best_pipe, best_auc = None, None, 0
    all_results = {}

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        prob   = pipe.predict_proba(X_te)[:, 1]
        acc    = round(float(accuracy_score(y_te, y_pred)), 4)
        f1     = round(float(f1_score(y_te, y_pred)), 4)
        auc    = round(float(roc_auc_score(y_te, prob)), 4)
        all_results[name] = {"accuracy": acc, "f1": f1, "auc": auc}
        print(f"\n── {name} ──\n   Acc={acc}  F1={f1}  AUC={auc}")
        print(classification_report(y_te, y_pred, target_names=["No HD","Has HD"]))
        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe

    clf_best = best_pipe.named_steps["clf"]
    fi = []
    if hasattr(clf_best, "feature_importances_"):
        names = NUMERIC_COLS + BINARY_COLS + list(ORDERED_CATS.keys())
        fi = sorted(zip(names, clf_best.feature_importances_.tolist()),
                    key=lambda x: x[1], reverse=True)

    payload = {
        "pipeline":            best_pipe,
        "model_name":          best_name,
        "accuracy":            all_results[best_name]["accuracy"],
        "auc":                 all_results[best_name]["auc"],
        "f1":                  all_results[best_name]["f1"],
        "feature_importances": fi,
        "numeric_cols":        NUMERIC_COLS,
        "binary_cols":         BINARY_COLS,
        "cat_cols":            list(ORDERED_CATS.keys()),
        "all_results":         all_results,
        "feature_cols":        feat_cols,
    }
    joblib.dump(payload, model_path)
    print(f"\n✅ Best: {best_name}  (AUC={best_auc})  → {model_path}")
    return payload


if __name__ == "__main__":
    custom = sys.argv[1] if len(sys.argv) > 1 else DATA
    train(custom)
