import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
ART = BASE_DIR / "artifacts"
STATIC = BASE_DIR / "static"

app = Flask(__name__, static_folder=str(STATIC), static_url_path="/static")

# ---------- Load everything saved from the notebook ----------
sklearn_model = joblib.load(ART / "sklearn_model.joblib")
scaler = joblib.load(ART / "scaler.joblib")
custom = np.load(ART / "custom_model.npz")
custom_weights = custom["weights"]
custom_bias = float(custom["bias"])
with open(ART / "feature_columns.json") as f:
    FEATURE_COLUMNS = json.load(f)

# ---------- Allowed input values (same as the dataset) ----------
YES_NO = ["Yes", "No"]
ADDON = ["Yes", "No", "No internet service"]
ADDON_FIELDS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

CATEGORICAL = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ADDON,
    "OnlineBackup": ADDON,
    "DeviceProtection": ADDON,
    "TechSupport": ADDON,
    "StreamingTV": ADDON,
    "StreamingMovies": ADDON,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}


def validate(payload):
    """Return (clean_dict, list_of_errors, total_charges_was_estimated)."""
    payload = dict(payload)
    errors, clean = [], {}

    # Dataset rules: no internet -> add-ons are "No internet service"; no phone -> "No phone service"
    if payload.get("InternetService") == "No":
        for field in ADDON_FIELDS:
            payload[field] = "No internet service"
    if payload.get("PhoneService") == "No":
        payload["MultipleLines"] = "No phone service"

    for field, allowed in CATEGORICAL.items():
        value = payload.get(field)
        if value not in allowed:
            errors.append(f"{field} must be one of {allowed}")
        else:
            clean[field] = value

    if payload.get("InternetService") in ("DSL", "Fiber optic"):
        for field in ADDON_FIELDS:
            if payload.get(field) == "No internet service":
                errors.append(f"{field} cannot be 'No internet service' when the customer has internet")

    try:
        senior = int(payload.get("SeniorCitizen"))
        if senior not in (0, 1):
            raise ValueError
        clean["SeniorCitizen"] = senior
    except (TypeError, ValueError):
        errors.append("SeniorCitizen must be 0 or 1")

    try:
        tenure = int(payload.get("tenure"))
        if not 1 <= tenure <= 72:
            raise ValueError
        clean["tenure"] = tenure
    except (TypeError, ValueError):
        errors.append("tenure must be a whole number between 1 and 72 (months)")

    try:
        monthly = float(payload.get("MonthlyCharges"))
        if not 0 < monthly <= 200:
            raise ValueError
        clean["MonthlyCharges"] = monthly
    except (TypeError, ValueError):
        errors.append("MonthlyCharges must be a number between 0 and 200")

    estimated = False
    total = payload.get("TotalCharges")
    if total in (None, ""):
        if "tenure" in clean and "MonthlyCharges" in clean:
            clean["TotalCharges"] = round(clean["tenure"] * clean["MonthlyCharges"], 2)
            estimated = True
    else:
        try:
            total = float(total)
            if not 0 <= total <= 10000:
                raise ValueError
            clean["TotalCharges"] = total
        except (TypeError, ValueError):
            errors.append("TotalCharges must be a number between 0 and 10000")

    return clean, errors, estimated


def build_features(clean):
    """Turn one customer into the exact 30-column row used in training."""
    row = pd.DataFrame([clean])
    # drop_first=False on purpose: with ONE row, drop_first=True would delete the only
    # column that exists. reindex() keeps only the columns the model was trained on.
    encoded = pd.get_dummies(row, drop_first=False)
    encoded = encoded.reindex(columns=FEATURE_COLUMNS, fill_value=0).astype(float)
    return encoded


def risk_category(p):
    # Simple labelling convention, NOT learned by the model. We can tune it in Phase 2.
    if p < 0.30:
        return "Low"
    if p < 0.60:
        return "Medium"
    return "High"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def describe(feature, raw):
    """Plain-language label for a feature, given the customer's actual (unscaled) value."""
    if feature == "tenure":
        return f"Tenure: {int(raw)} months"
    if feature == "MonthlyCharges":
        return f"Monthly charges: ${raw:.2f}"
    if feature == "TotalCharges":
        return f"Total charges: ${raw:.2f}"
    if feature == "SeniorCitizen":
        return "Senior citizen" if raw == 1 else "Not a senior citizen"
    if "_" in feature:
        base, cat = feature.split("_", 1)
        return f"{base}: {cat}" if raw == 1 else f"{base}: not {cat}"
    return feature


@app.get("/")
def home():
    return send_from_directory(STATIC, "index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", n_features=len(FEATURE_COLUMNS),
                   models=["sklearn", "custom"])


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="Send a JSON object"), 400

    clean, errors, total_estimated = validate(payload)
    if errors:
        return jsonify(error="Invalid input", details=errors), 400

    which = payload.get("model", "sklearn")
    if which not in ("sklearn", "custom"):
        return jsonify(error="model must be 'sklearn' or 'custom'"), 400

    features = build_features(clean)
    x_scaled = scaler.transform(features)[0]
    raw_values = features.iloc[0]

    sk_w = sklearn_model.coef_[0]
    sk_b = float(sklearn_model.intercept_[0])
    prob_sklearn = float(sigmoid(np.dot(x_scaled, sk_w) + sk_b))
    prob_custom = float(sigmoid(np.dot(x_scaled, custom_weights) + custom_bias))

    weights = sk_w if which == "sklearn" else custom_weights
    prob = prob_sklearn if which == "sklearn" else prob_custom

    # Exact explanation for logistic regression: contribution = weight * scaled value (log-odds).
    # A scaled value of 0 means "average training customer", so these are shifts vs the average.
    contrib = weights * x_scaled
    order = np.argsort(-np.abs(contrib))[:8]
    top_factors = [{
        "feature": FEATURE_COLUMNS[i],
        "label": describe(FEATURE_COLUMNS[i], float(raw_values.iloc[i])),
        "contribution": round(float(contrib[i]), 4),
        "effect": "increases churn risk" if contrib[i] > 0 else "reduces churn risk",
    } for i in order]

    return jsonify(
        model_used=which,
        churn_prediction=int(prob >= 0.5),
        threshold=0.5,
        churn_probability=round(prob, 4),
        risk_category=risk_category(prob),
        probabilities_by_model={"sklearn": round(prob_sklearn, 4),
                                "custom": round(prob_custom, 4)},
        top_factors=top_factors,
        total_charges_estimated=total_estimated,
    )

@app.get("/dashboard")
def dashboard_page():
    return send_from_directory(STATIC, "dashboard.html")

@app.get("/insights")
def insights_page():
    return send_from_directory(STATIC, "insights.html")

@app.get("/performance")
def performance_page():
    return send_from_directory(STATIC, "performance.html")

@app.get("/settings")
def settings_page():
    return send_from_directory(STATIC, "settings.html")

@app.get("/api/stats")
def stats():
    path = ART / "stats.json"
    if not path.exists():
        return jsonify(error="stats.json not found. Run Cell 9 in the notebook."), 404
    with open(path) as f:
        return jsonify(json.load(f))

    
if __name__ == "__main__":
    app.run(debug=True, port=5000)