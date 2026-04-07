"""
main.py — HeartWise FastAPI Backend
Run: python backend/main.py
"""
import os, io, csv, logging, subprocess, sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("heartwise")

BASE       = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model.pkl")
DATA_PATH  = os.path.join(BASE, "data", "heart_disease.csv")

_model: dict = {}


# ── startup ──────────────────────────────────────────────────────────────────
def _load_or_train():
    global _model
    if os.path.exists(MODEL_PATH):
        log.info("Loading model …")
        _model = joblib.load(MODEL_PATH)
    else:
        log.info("No model found — training now (this may take a few minutes) …")
        from train import train
        _model = train(DATA_PATH, MODEL_PATH)
    log.info("Model ready: %s  acc=%.4f  auc=%.4f",
             _model["model_name"], _model["accuracy"], _model["auc"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_or_train()
    yield


app = FastAPI(title="HeartWise API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True,
)


# ── schemas ───────────────────────────────────────────────────────────────────
class PatientIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    BMI:              float = Field(..., ge=10,  le=100)
    Smoking:          str   = Field(..., pattern="^(Yes|No)$")
    AlcoholDrinking:  str   = Field(..., pattern="^(Yes|No)$")
    Stroke:           str   = Field(..., pattern="^(Yes|No)$")
    PhysicalHealth:   float = Field(..., ge=0,   le=30)
    MentalHealth:     float = Field(..., ge=0,   le=30)
    DiffWalking:      str   = Field(..., pattern="^(Yes|No)$")
    Sex:              str   = Field(..., pattern="^(Male|Female)$")
    AgeCategory:      str
    Race:             str
    Diabetic:         str
    PhysicalActivity: str   = Field(..., pattern="^(Yes|No)$")
    GenHealth:        str
    SleepTime:        float = Field(..., ge=0,   le=24)
    Asthma:           str   = Field(..., pattern="^(Yes|No)$")
    KidneyDisease:    str   = Field(..., pattern="^(Yes|No)$")
    SkinCancer:       str   = Field(..., pattern="^(Yes|No)$")


class PredictionOut(BaseModel):
    prediction:       str    # "Yes" | "No"
    probability:      float
    risk_level:       str    # Low / Moderate / High / Very High
    risk_score:       int    # 0-100
    top_factors:      list
    recommendations:  list[str]
    model_used:       str
    confidence:       str


# ── helpers ───────────────────────────────────────────────────────────────────
FEAT_COLS = [
    "BMI","PhysicalHealth","MentalHealth","SleepTime",
    "Smoking","AlcoholDrinking","Stroke","DiffWalking",
    "PhysicalActivity","Asthma","KidneyDisease","SkinCancer",
    "Sex","AgeCategory","Race","Diabetic","PhysicalActivity","GenHealth",
]


def _to_df(p: PatientIn) -> pd.DataFrame:
    return pd.DataFrame([{
        "BMI": p.BMI, "PhysicalHealth": p.PhysicalHealth,
        "MentalHealth": p.MentalHealth, "SleepTime": p.SleepTime,
        "Smoking": p.Smoking, "AlcoholDrinking": p.AlcoholDrinking,
        "Stroke": p.Stroke, "DiffWalking": p.DiffWalking,
        "PhysicalActivity": p.PhysicalActivity, "Asthma": p.Asthma,
        "KidneyDisease": p.KidneyDisease, "SkinCancer": p.SkinCancer,
        "Sex": p.Sex, "AgeCategory": p.AgeCategory,
        "Race": p.Race, "Diabetic": p.Diabetic, "GenHealth": p.GenHealth,
    }])


def _risk_label(prob: float) -> str:
    if prob < 0.20: return "Low"
    if prob < 0.40: return "Moderate"
    if prob < 0.65: return "High"
    return "Very High"


def _recommendations(p: PatientIn, prob: float) -> list[str]:
    r = []
    if p.Smoking == "Yes":
        r.append("🚭 Quit smoking — it's the single biggest reducible heart disease risk factor.")
    if p.PhysicalActivity == "No":
        r.append("🏃 Get at least 150 min/week of moderate exercise (brisk walking, cycling).")
    if p.BMI > 30:
        r.append(f"⚖️ BMI {p.BMI:.1f} indicates obesity. A 5-10% weight loss significantly lowers risk.")
    if p.AlcoholDrinking == "Yes":
        r.append("🍷 Limit alcohol to ≤ 1 drink/day (female) or ≤ 2/day (male).")
    if p.PhysicalHealth > 14:
        r.append("🩺 You reported poor physical health for many days — seek a doctor evaluation.")
    if p.MentalHealth > 14:
        r.append("🧠 High mental health burden. Consider counselling or stress-management practices.")
    if p.SleepTime < 6 or p.SleepTime > 9:
        r.append(f"😴 Sleep of {p.SleepTime}h is outside the healthy 7-9h range. Prioritise sleep hygiene.")
    if p.Diabetic in ("Yes", "Yes (during pregnancy)"):
        r.append("🩸 Manage blood glucose tightly — diabetes doubles cardiovascular risk.")
    if p.Stroke == "Yes":
        r.append("🏥 Prior stroke is a major risk factor — regular cardiology follow-up is essential.")
    if p.KidneyDisease == "Yes":
        r.append("💊 Kidney disease and heart disease are strongly linked — monitor both closely.")
    if p.GenHealth in ("Poor", "Fair"):
        r.append("📋 Self-reported poor health correlates with outcomes — schedule a full health check.")
    if prob >= 0.5:
        r.append("❤️ High predicted risk — schedule a comprehensive cardiac evaluation now.")
    if not r:
        r.append("✅ Great profile! Maintain your healthy habits and get regular check-ups.")
    return r


def _top_factors(p: PatientIn) -> list[dict]:
    fi = _model.get("feature_importances", [])
    val_map = {
        "BMI": p.BMI, "PhysicalHealth": p.PhysicalHealth,
        "MentalHealth": p.MentalHealth, "SleepTime": p.SleepTime,
        "Smoking": p.Smoking, "AlcoholDrinking": p.AlcoholDrinking,
        "Stroke": p.Stroke, "DiffWalking": p.DiffWalking,
        "PhysicalActivity": p.PhysicalActivity, "Asthma": p.Asthma,
        "KidneyDisease": p.KidneyDisease, "SkinCancer": p.SkinCancer,
        "Sex": p.Sex, "AgeCategory": p.AgeCategory,
        "Race": p.Race, "Diabetic": p.Diabetic, "GenHealth": p.GenHealth,
    }
    return [{"feature": f, "importance": round(v, 4), "value": val_map.get(f, "–")}
            for f, v in fi[:8]]


def _append_to_csv(p: PatientIn, prediction: str, probability: float):
    """Append a new prediction row to the dataset CSV."""
    row = {
        "HeartDisease":     prediction,
        "BMI":              p.BMI,
        "Smoking":          p.Smoking,
        "AlcoholDrinking":  p.AlcoholDrinking,
        "Stroke":           p.Stroke,
        "PhysicalHealth":   p.PhysicalHealth,
        "MentalHealth":     p.MentalHealth,
        "DiffWalking":      p.DiffWalking,
        "Sex":              p.Sex,
        "AgeCategory":      p.AgeCategory,
        "Race":             p.Race,
        "Diabetic":         p.Diabetic,
        "PhysicalActivity": p.PhysicalActivity,
        "GenHealth":        p.GenHealth,
        "SleepTime":        p.SleepTime,
        "Asthma":           p.Asthma,
        "KidneyDisease":    p.KidneyDisease,
        "SkinCancer":       p.SkinCancer,
    }
    file_exists = os.path.exists(DATA_PATH)
    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    log.info("Appended new record to dataset (prediction=%s, prob=%.3f)", prediction, probability)


# ── routes ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model": _model.get("model_name", "none")}


@app.get("/api/model/info")
def model_info():
    if not _model:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_name":          _model["model_name"],
        "accuracy":            _model["accuracy"],
        "auc":                 _model["auc"],
        "f1":                  _model["f1"],
        "feature_importances": _model["feature_importances"][:10],
        "all_results":         _model.get("all_results", {}),
        "dataset_rows":        _get_dataset_rows(),
    }


def _get_dataset_rows() -> int:
    try:
        df = pd.read_csv(DATA_PATH, usecols=["HeartDisease"])
        return len(df)
    except Exception:
        return 0


@app.post("/api/predict", response_model=PredictionOut)
def predict(patient: PatientIn, background_tasks: BackgroundTasks):
    if not _model:
        raise HTTPException(503, "Model not loaded")
    try:
        df   = _to_df(patient)
        pipe = _model["pipeline"]
        prob = float(pipe.predict_proba(df)[0, 1])
        pred = "Yes" if prob >= 0.5 else "No"

        # Save new record to CSV in background (non-blocking)
        background_tasks.add_task(_append_to_csv, patient, pred, prob)

        return PredictionOut(
            prediction=pred,
            probability=round(prob, 4),
            risk_level=_risk_label(prob),
            risk_score=min(int(prob * 100), 99),
            top_factors=_top_factors(patient),
            recommendations=_recommendations(patient, prob),
            model_used=_model["model_name"],
            confidence="High" if abs(prob - 0.5) > 0.25 else ("Medium" if abs(prob - 0.5) > 0.1 else "Low"),
        )
    except Exception as e:
        log.exception("Predict error")
        raise HTTPException(500, str(e))


@app.get("/api/dataset/analyze")
def analyze_dataset(page: int = 1, page_size: int = 1000):
    """
    Read backend CSV and return predictions for a specific page.
    This is a server-side batch analysis — no file upload needed.
    """
    if not _model:
        raise HTTPException(503, "Model not loaded")
    try:
        df   = pd.read_csv(DATA_PATH)
        df   = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        
        # Sort so newest predictions (bottom of CSV) are first if desired, 
        # or just reverse it for display. Let's just reverse the whole DF so page 1 is newest:
        df_rev = df.iloc[::-1].reset_index(drop=True)
        
        total_rows = len(df_rev)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * page_size
        end_idx   = start_idx + page_size
        
        sample = df_rev.iloc[start_idx:end_idx]
        
        pipe   = _model["pipeline"]
        feat   = _model["feature_cols"]
        probs  = pipe.predict_proba(sample[feat])[:, 1]

        results = []
        for i, (prob, row) in enumerate(zip(probs, sample.itertuples())):
            # The exact row number in the original dataset
            original_row_num = total_rows - (start_idx + i)
            results.append({
                "row":         original_row_num,
                "prediction":  "Yes" if prob >= 0.5 else "No",
                "actual":      getattr(row, "HeartDisease", "–"),
                "probability": round(float(prob), 3),
                "risk_level":  _risk_label(prob),
                "risk_score":  min(int(prob * 100), 99),
            })

        hd_count = int((df["HeartDisease"] == "Yes").sum())
        pred_hd  = sum(1 for r in results if r["prediction"] == "Yes")

        return {
            "total_dataset_rows": total_rows,
            "heart_disease_count": hd_count,
            "prevalence_pct": round(hd_count / total_rows * 100, 2) if total_rows else 0,
            "analyzed_rows": len(results),
            "predicted_hd_in_sample": pred_hd,
            "model_used": _model["model_name"],
            "page": page,
            "total_pages": total_pages,
            "results": results,
        }
    except Exception as e:
        log.exception("Dataset analyze error")
        raise HTTPException(500, str(e))


@app.get("/api/dataset/stats")
def dataset_stats():
    """Return high-level statistics about the stored dataset."""
    try:
        df = pd.read_csv(DATA_PATH)
        total     = len(df)
        hd_yes    = int((df["HeartDisease"] == "Yes").sum())
        hd_no     = int((df["HeartDisease"] == "No").sum())
        avg_bmi   = round(float(df["BMI"].mean()), 2)
        smokers   = int((df["Smoking"] == "Yes").sum())
        diabetics = int(df["Diabetic"].isin(["Yes", "Yes (during pregnancy)"]).sum())
        return {
            "total_rows":       total,
            "heart_disease_yes": hd_yes,
            "heart_disease_no":  hd_no,
            "prevalence_pct":   round(hd_yes / total * 100, 2) if total else 0,
            "avg_bmi":          avg_bmi,
            "smokers":          smokers,
            "diabetics":        diabetics,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/retrain")
def retrain():
    """Retrain the model on the latest dataset (runs synchronously — may take minutes)."""
    try:
        from train import train
        global _model
        _model = train(DATA_PATH, MODEL_PATH)
        return {"status": "ok", "model": _model["model_name"],
                "accuracy": _model["accuracy"], "auc": _model["auc"]}
    except Exception as e:
        log.exception("Retrain error")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
