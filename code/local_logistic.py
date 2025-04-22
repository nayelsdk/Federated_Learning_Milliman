import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

class LocalLogisticModel:
    def __init__(self, df):
        self.df = df.dropna()
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.exposure_train = self.X_train["Exposure"]
        self.exposure_test = self.X_test["Exposure"]
        self.X_train = self.X_train.drop(columns=["Exposure"])
        self.X_test = self.X_test.drop(columns=["Exposure"])

        self.X_train = self.normalize(self.X_train, fit=True)
        self.X_test = self.normalize(self.X_test, fit=False)

        self.model = LogisticRegression(
            class_weight="balanced",
            fit_intercept=True,
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )

    def split_data(self, test_size=0.40):
        X = self.df.drop(columns=["Sinistre"])
        y = self.df["Sinistre"]
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def normalize(self, X, fit=False):
        if fit:
            return pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

    def train(self):
        self.model.fit(self.X_train, self.y_train, sample_weight=self.exposure_train)

    def predict_proba(self):
        probs = self.model.predict_proba(self.X_test)[:, 1]
        return probs * self.exposure_test.values

    def get_coefficients(self):
        coefs = dict(zip(self.X_train.columns, self.model.coef_[0]))
        coefs["Intercept"] = self.model.intercept_[0]
        return coefs

    def set_coefficients(self, coef_dict):
        self.model.coef_ = np.array([[coef_dict[k] for k in self.X_train.columns]])
        self.model.intercept_ = np.array([coef_dict["Intercept"]])
