from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class FeatureEngineer:
    def __init__(self, degree=2, interaction_only=True, include_bias=False):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )
        self.feature_names = None

    def fit(self, X_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_poly = self.poly.fit_transform(X_train_scaled)
        self.feature_names = self.poly.get_feature_names_out(X_train.columns)
        return X_train_poly

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        X_poly = self.poly.transform(X_scaled)
        return X_poly