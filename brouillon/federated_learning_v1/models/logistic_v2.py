from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import numpy as np
from .base_model import BaseModel

class LogisticModel_Stat(BaseModel):
    def __init__(self, alpha=0.0):
        print(f"[DEBUG LogisticModel_SKL] Initialisation avec alpha = {alpha}")
        self.alpha = alpha
        self.C = 1.0 / alpha if alpha > 0 else float('inf')  # Utiliser inf au lieu de 1e12
        self.model = None
        self.scaler = StandardScaler()
        self.fitted = False

    def train(self, X, y, sample_weight=None):
        print(f"[DEBUG train] alpha = {self.alpha} | C = {self.C}")
        
        # Mise à jour du modèle avec la régularisation actuelle
        self.model = LogisticRegression(
            penalty='l2',
            C=self.C,
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=True,
            warm_start=False  # On désactive warm_start pour forcer la réinitialisation
        )
        
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement avec la régularisation
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.fitted = True
        return self

    def predict_proba(self, X):
        if not self.fitted:
            raise NotFittedError("Le modèle doit être entraîné avant.")
        X_scaled = self.scaler.transform(X)
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
        Initialise un modèle avec des poids donnés et la régularisation courante
        """
        intercept = weights[0]
        coef = weights[1:]

        # Création d'un nouveau modèle avec la régularisation actuelle
        self.model = LogisticRegression(
            penalty='l2',
            C=self.C,
            solver='lbfgs',
            max_iter=1000,
            fit_intercept=True,
            warm_start=False
        )

        # Configuration du scaler si nécessaire
        if not hasattr(self.scaler, 'mean_'):
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.zeros_like(coef)
            self.scaler.scale_ = np.ones_like(coef)
            self.scaler.var_ = np.ones_like(coef)
            self.scaler.n_features_in_ = len(coef)
            self.scaler.n_samples_seen_ = 1

        # Configuration du modèle
        self.model.classes_ = np.array([0, 1])
        self.model.coef_ = np.array([coef])
        self.model.intercept_ = np.array([intercept])
        self.fitted = True

    def update_regularization(self, alpha):
        """
        Met à jour le paramètre de régularisation
        """
        self.alpha = alpha
        self.C = 1.0 / alpha if alpha > 0 else float('inf')
        print(f"[DEBUG update_regularization] Nouveau alpha = {self.alpha}, C = {self.C}")
        return self
