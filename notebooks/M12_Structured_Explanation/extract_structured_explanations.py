import joblib
import json
import numpy as np
from datetime import datetime

# -----------------------------
# Load artifacts
# -----------------------------
X_eval = joblib.load("models/X_eval.pkl")
shap_values = joblib.load("models/shap_values.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# -----------------------------
# Configuration
# -----------------------------
TOP_K = 10
OUTPUT_PATH = "models/M12_outputs/structured_explanations.json"

structured_explanations = []

# -----------------------------
# Generate structured explanations
# -----------------------------
for idx in range(len(X_eval)):
    shap_instance = shap_values[idx]

    # Sort by absolute contribution
    top_indices = np.argsort(np.abs(shap_instance))[::-1][:TOP_K]

    feature_list = []
    for i in top_indices:
        feature_list.append({
            "feature_name": feature_names[i],
            "contribution_score": float(shap_instance[i]),
            "impact": "positive" if shap_instance[i] > 0 else "negative"
        })

    explanation = {
        "instance_id": idx,
        "generated_at": datetime.utcnow().isoformat(),
        "top_features": feature_list
    }

    structured_explanations.append(explanation)

# -----------------------------
# Save output
# -----------------------------
with open(OUTPUT_PATH, "w") as f:
    json.dump(structured_explanations, f, indent=4)

print(f"✅ PHASE M12 completed. Output saved to {OUTPUT_PATH}")
