from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import numpy as np
from .base_model import BaseModel

class LogisticModel_Stat(BaseModel):
    def __init__(self, alpha=0.0):
        print(f"[DEBUG LogisticModel_SKL] Initialisation avec alpha = {alpha}")
        self.alpha = alpha
        self.C = 1.0 / alpha if alpha > 0 else 1e12  # C élevé si pas de régularisation
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def train(self, X, y, sample_weight=None):
        print(f"[DEBUG train] alpha = {self.alpha} | C = {self.C}")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = LogisticRegression(
            penalty='l2',
            C=self.C,
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=True
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.fitted = True
        return self

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError("Le modèle doit être entraîné avant.")
        try:
            X_scaled = self.scaler.transform(X)
        except NotFittedError:
            raise NotFittedError("Le StandardScaler doit être entraîné avant.")
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_weights(self):
        if not self.fitted:
            raise ValueError("Modèle non entraîné.")
        intercept = self.model.intercept_
        coef = self.model.coef_.flatten()
        return np.concatenate([intercept, coef])

    def set_weights(self, weights):
        """
        Initialise un modèle non entraîné avec des poids. À utiliser uniquement après un premier fit().
        """
        intercept = weights[0]
        coef = weights[1:]

        self.model = LogisticRegression(
            penalty='l2',
            C=self.C,
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=True
        )
        # Important : on initialise un scaler identitaire si non fit
        if not hasattr(self.scaler, 'mean_'):
            self.scaler = StandardScaler()
            # Simule un scaler "neutre" : pas de centrage/échelle
            self.scaler.mean_ = np.zeros_like(coef)
            self.scaler.scale_ = np.ones_like(coef)
            self.scaler.var_ = np.ones_like(coef)
            self.scaler.n_features_in_ = len(coef)
            self.scaler.n_samples_seen_ = 1

        self.model.classes_ = np.array([0, 1])
        self.model.coef_ = np.array([coef])
        self.model.intercept_ = np.array([intercept])
        self.fitted = True
