from imblearn.over_sampling import SMOTE

class ImbalanceHandler:
    def __init__(self, random_state=42):
        self.smote = SMOTE(random_state=random_state)

    def resample(self, X_train_poly, y_train):
        X_train_res, y_train_res = self.smote.fit_resample(X_train_poly, y_train)
        return X_train_res, y_train_res