import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def train(self, model, X_train, y_train, eval_set=None):
        if isinstance(model, lgb.LGBMClassifier) and eval_set is not None:
            model.fit(X_train, y_train, eval_set=eval_set)
        else:
            model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()