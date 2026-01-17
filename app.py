from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

# Use __name__ (double underscore)
app = Flask(__name__, template_folder="templates")

# Load trained model
model_path = os.path.join(os.getcwd(), "XGBoost_multi_disease_model.joblib")
model = joblib.load(model_path)

# Labels (must match your training labels)
LABELS = ["diabetes", "ckd", "heart_failure", "hypertension", "copd"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.get_json()
        input_df = pd.DataFrame([data])

        probs, preds = [], []
        for i, est in enumerate(model.estimators_):
            prob = float(est.predict_proba(input_df)[:, 1][0])  # Convert to float
            pred = int(prob >= 0.5)
            probs.append(prob)
            preds.append(pred)

        # Build result dictionary
        result = {
            label: {
                "prediction": int(p),
                "probability": round(float(pr), 3)
            }
            for label, p, pr in zip(LABELS, preds, probs)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

# Entry point
if __name__ == "__main__":
    app.run(debug=True)
