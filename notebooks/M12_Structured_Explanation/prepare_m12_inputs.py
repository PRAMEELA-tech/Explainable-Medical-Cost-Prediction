import joblib
import shap
import numpy as np

# -----------------------------
# Load trained model & data
# -----------------------------
model = joblib.load("models/random_forest_optimized.pkl")
X = joblib.load("models/X_processed.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# -----------------------------
# Extract feature names
# -----------------------------
cat_features = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out()
)

num_features = preprocessor.named_transformers_["num"].feature_names_in_

feature_names = list(num_features) + list(cat_features)

# -----------------------------
# Compute SHAP values
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# -----------------------------
# Save artifacts for M12
# -----------------------------
joblib.dump(X, "models/X_eval.pkl")
joblib.dump(shap_values, "models/shap_values.pkl")
joblib.dump(feature_names, "models/feature_names.pkl")

print("✅ M12 input artifacts prepared successfully")
