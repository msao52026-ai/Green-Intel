"""
GreenIntel - Production Ready Flask Application
Run with: gunicorn --bind 0.0.0.0:8000 app:app
For development: python app.py (will use debug if FLASK_DEBUG=true)
"""

import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, flash, jsonify
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pickle
from werkzeug.utils import secure_filename

# =========================
# CONFIGURATION (from environment)
# =========================
class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", os.urandom(24).hex())
    DEBUG = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    REQUIRED_COLUMNS = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]
    ALLOWED_EXTENSIONS = {"csv"}

# =========================
# INIT APP & EXTENSIONS
# =========================
app = Flask(__name__)
app.config.from_object(Config)

# CSRF Protection
csrf = CSRFProtect(app)

# Rate Limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s'
)
app.logger.setLevel(logging.INFO)

# =========================
# MODEL LOADING (robust)
# =========================
model = None
def load_model():
    global model
    try:
        with open(app.config["MODEL_PATH"], "rb") as f:
            model = pickle.load(f)
        app.logger.info("Model loaded successfully from %s", app.config["MODEL_PATH"])
    except Exception as e:
        app.logger.error("Failed to load model: %s", e)
        model = None

load_model()  # load at startup

# =========================
# HELPER FUNCTIONS
# =========================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def validate_range(field, value, lo, hi):
    if not (lo <= value <= hi):
        raise ValueError(f"{field} value {value} outside range {lo}-{hi}")

# =========================
# HEALTH CHECK
# =========================
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# ---------- MANUAL PREDICTION ----------
@app.route("/predict", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def predict():
    if request.method == "POST":
        is_ajax = (
            request.is_json or
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
        )

        # --- Input parsing ---
        try:
            if is_ajax:
                data = request.get_json(force=True)
                n = float(data["N"])
                p = float(data["P"])
                k = float(data["K"])
                temperature = float(data["temperature"])
                humidity = float(data["humidity"])
                ph = float(data["ph"])
                rainfall = float(data["rainfall"])
            else:
                n = float(request.form["N"])
                p = float(request.form["P"])
                k = float(request.form["K"])
                temperature = float(request.form["temperature"])
                humidity = float(request.form["humidity"])
                ph = float(request.form["ph"])
                rainfall = float(request.form["rainfall"])
        except (KeyError, TypeError, ValueError) as e:
            app.logger.warning("Bad input: %s", e)
            if is_ajax:
                return jsonify({"error": "Invalid input values"}), 400
            flash("❌ Invalid input values – check all fields.", "danger")
            return render_template("predict.html")

        # --- Range validation ---
        RANGES = {
            "N": (0, 140), "P": (0, 145), "K": (0, 205),
            "temperature": (0, 50), "humidity": (0, 100),
            "ph": (0, 14), "rainfall": (0, 3000)
        }
        values = {
            "N": n, "P": p, "K": k,
            "temperature": temperature, "humidity": humidity,
            "ph": ph, "rainfall": rainfall
        }
        try:
            for field, (lo, hi) in RANGES.items():
                validate_range(field, values[field], lo, hi)
        except ValueError as e:
            app.logger.warning("Range error: %s", e)
            if is_ajax:
                return jsonify({"error": str(e)}), 400
            flash(f"❌ {str(e)}", "danger")
            return render_template("predict.html")

        # --- Model guard ---
        if model is None:
            app.logger.error("Prediction attempted but model not loaded")
            if is_ajax:
                return jsonify({"error": "Model unavailable"}), 503
            flash("❌ Prediction model is currently unavailable.", "danger")
            return render_template("predict.html")

        # --- Inference ---
        try:
            features = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
            prediction = model.predict(features)[0]
            confidence = 0.0
            if hasattr(model, "predict_proba"):
                confidence = float(max(model.predict_proba(features)[0])) * 100
        except Exception as e:
            app.logger.error("Inference failed: %s", e)
            if is_ajax:
                return jsonify({"error": "Prediction failed"}), 500
            flash("❌ Prediction failed – please try again.", "danger")
            return render_template("predict.html")

        # --- Respond ---
        if is_ajax:
            return jsonify({
                "prediction": str(prediction),
                "confidence": round(confidence, 2)
            })
        # legacy full-page response
        return render_template(
            "predict.html",
            prediction=prediction,
            confidence=f"{confidence:.2f}",
            form_data=request.form
        )

    # GET request
    return render_template("predict.html")

# ---------- MODEL COMPARISON ----------
@app.route("/supervised")
def supervised():
    return render_template(
        "supervised.html",
        accuracy_rf=99.39,
        accuracy_svm=96.13,
        accuracy_knn=97.05,
        accuracy_dt=98.64,
        accuracy_lr=94.77
    )

# ---------- ANALYSIS ----------
@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

# ---------- CSV UPLOAD ----------
@app.route("/upload", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not allowed_file(file.filename):
            flash("Please upload a valid CSV file.", "warning")
            return render_template("upload.html")

        filename = secure_filename(file.filename)
        try:
            df = pd.read_csv(file, nrows=10000)
            if len(df) == 10000:
                flash("Only first 10,000 rows processed due to file size.", "info")

            df.columns = [c.lower().strip() for c in df.columns]

            missing = set(app.config["REQUIRED_COLUMNS"]) - set(df.columns)
            if missing:
                flash(f"Missing columns: {', '.join(missing)}", "danger")
                return render_template("upload.html")

            X = df[app.config["REQUIRED_COLUMNS"]]
            predictions = model.predict(X)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X).max(axis=1) * 100
            else:
                probabilities = [0] * len(X)

            df["Predicted Crop"] = predictions
            df["Confidence (%)"] = np.round(probabilities, 2)

            table_html = df.to_html(classes="table table-striped table-bordered", index=False)
            flash("File processed successfully!", "success")
            return render_template("upload.html", table_html=table_html)

        except pd.errors.EmptyDataError:
            flash("Uploaded file is empty.", "danger")
        except Exception as e:
            app.logger.error("CSV processing error: %s", e)
            flash("Error processing CSV file.", "danger")

    return render_template("upload.html")

# ---------- MSP DASHBOARD ----------
@app.route("/msp")
def msp():
    return render_template("msp.html")

# ---------- CUBE GALLERY ----------
@app.route("/cube")
def cube():
    return render_template("front.html")

# =========================
# ERROR HANDLERS
# =========================
@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(e):
    app.logger.error("Server error: %s", e)
    return render_template("500.html"), 500

# =========================
# RUN (only for development)
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=app.config["DEBUG"])