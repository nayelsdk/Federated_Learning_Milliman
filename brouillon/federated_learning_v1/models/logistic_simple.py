from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from .base_model import BaseModel

class LogisticModelSimple(BaseModel):
    """
    Implémentation simple d'un modèle de régression logistique pour l'apprentissage fédéré.
    Sans régularisation ni apprentissage sensible au coût, juste les fonctionnalités de base.
    """
    
    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            fit_intercept=True,
            warm_start=True  # Utiliser warm_start pour réutiliser les poids
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self._scaler_fitted = False
    
    def train(self, X, y, sample_weight=None):
        """
        Entraîne le modèle sur les données locales
        """
        # Gestion du scaler pour réutiliser les mêmes transformations
        if not self._scaler_fitted:
            X_scaled = self.scaler.fit_transform(X)
            self._scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X)
        
        # Entraînement du modèle en partant des poids actuels
        self.model.fit(X_scaled, y, sample_weight=sample_weight)
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Prédit les probabilités des classes pour les données d'entrée
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_weights(self):
        """
        Retourne les poids du modèle sous forme de tableau numpy 1D
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant d'extraire les poids")
        
        intercept = self.model.intercept_
        coef = self.model.coef_.flatten()
        return np.concatenate([intercept, coef])
    
    def set_weights(self, weights):
        """
        Met à jour les poids du modèle à partir d'un tableau numpy 1D
        """
        if len(weights) < 2:
            raise ValueError("Le vecteur de poids doit inclure l'intercept et les coefficients")
        
        intercept = weights[0]
        coef = weights[1:]
        
        # S'assurer que le modèle a les attributs nécessaires
        if not hasattr(self.model, 'classes_'):
            self.model.classes_ = np.array([0, 1])
        
        # Mise à jour des poids
        self.model.intercept_ = np.array([intercept])
        self.model.coef_ = np.array([coef])
        self.fitted = True
        
        # Afficher un message de debug
        print(f"[DEBUG set_weights] Nouveaux poids: intercept={intercept:.6f}, coef_shape={coef.shape}") 