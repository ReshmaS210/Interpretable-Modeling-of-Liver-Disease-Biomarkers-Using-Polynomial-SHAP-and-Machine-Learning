import shap
import numpy as np
import pandas as pd
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*SHAP values output has changed.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

class SHAPAnalyzer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = None
        self.selected_features = None

    def compute_shap_values(self, X_test_poly):
        self.shap_values = self.explainer.shap_values(X_test_poly)
        if isinstance(self.shap_values, list):
            # Use SHAP values for the positive class (index 1)
            mean_abs_shap = np.abs(self.shap_values[1])
        else:
            mean_abs_shap = np.abs(self.shap_values)
        mean_shap_per_feature = np.mean(mean_abs_shap, axis=0)
        total_shap = np.sum(mean_shap_per_feature)
        shap_percentages = mean_shap_per_feature / total_shap
        shap_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_ABS_SHAP": mean_shap_per_feature,
            "Contribution": shap_percentages
        })
        self.shap_df = shap_df.sort_values("Contribution", ascending=False)

    def select_features(self, threshold=0.03):
        self.selected_features = self.shap_df[self.shap_df["Contribution"] >= threshold]["Feature"].values
        return self.selected_features

    def get_selected_indices(self):
        all_features = np.array(self.feature_names)
        selected_idx = [i for i, f in enumerate(all_features) if f in self.selected_features]
        return selected_idx