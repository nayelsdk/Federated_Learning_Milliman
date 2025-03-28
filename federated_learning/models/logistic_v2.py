import numpy as np
import statsmodels.api as sm
from .base_model import BaseModel

class LogisticModel_Stat(BaseModel):
    """
    Modèle de régression logistique avec statsmodels.
    Inclut un offset basé sur 'exposure' (log-transformé).
    Compatible avec l'approche federated averaging sans réutilisation des poids globaux.
    """

    def __init__(self):
        self.model = None
        self.results = None
        self.fitted = False

    def train(self, X, y, sample_weight=None):
        """
        Entraîne le modèle sur les données locales avec offset = log(exposure)
        """
        X_const = sm.add_constant(X, has_constant='add')
        offset = np.log(sample_weight) if sample_weight is not None else None

        self.model = sm.Logit(y, X_const, offset=offset)
        self.results = self.model.fit(disp=False)
        self.fitted = True
        return self

    def predict(self, X):
        """
        Prédit les classes binaires (0/1) en utilisant une probabilité seuil de 0.5
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions.")
        X_const = sm.add_constant(X, has_constant='add')
        probs = self.results.predict(X_const)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X):
        """
        Retourne les probabilités prédites de la classe positive
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions.")
        X_const = sm.add_constant(X, has_constant='add')
        return self.results.predict(X_const)

    def get_weights(self):
        """
        Retourne les coefficients du modèle sous forme de tableau numpy 1D
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant d'extraire les poids.")
        return self.results.params.values

    def set_weights(self, weights):
        """
        Hack : met à jour les poids du modèle via les params de statsmodels.
        ATTENTION : Ne met pas à jour les p-values ni les statistiques associées.
        à utiliser uniquement pour permettre l'agrégation FedAvg.
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné au moins une fois avant de modifier les poids.")

        if len(weights) != len(self.results.params):
            raise ValueError("Taille du vecteur de poids incompatible avec le modèle actuel.")

        self.results.params[:] = weights
