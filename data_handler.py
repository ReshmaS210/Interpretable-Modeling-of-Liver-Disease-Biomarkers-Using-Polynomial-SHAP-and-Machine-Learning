import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataHandler:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.le = LabelEncoder()
        self.means = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def load_dataset(self):
        column_names = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Proteins', 'Albumin',
            'Albumin_and_Globulin_Ratio', 'Selector'
        ]
        df = pd.read_csv(self.dataset_path, header=None, names=column_names)
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        return df

    def prepare_data(self, test_size=0.4, val_size=0.5, random_state=42):
        df = self.load_dataset()
        X = df.drop(columns=['Selector'])
        y = df['Selector']
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
        )
        self.means = X_train.mean()
        X_train.fillna(self.means, inplace=True)
        X_val.fillna(self.means, inplace=True)
        X_test.fillna(self.means, inplace=True)
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_test, self.y_test = X_test, y_test

    def load_test_csv(self, test_path):
        test_df = pd.read_csv(test_path, header=None, names=self.X_train.columns)
        test_df['Gender'] = test_df['Gender'].map({'Male': 1, 'Female': 0})
        test_df.fillna(self.means, inplace=True)
        return test_df