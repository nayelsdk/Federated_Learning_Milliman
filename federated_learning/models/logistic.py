import numpy as np
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticModel(BaseModel):
    """
    Implémentation d'un modèle de régression logistique pour l'apprentissage fédéré
    """
    
    def __init__(self, max_iter=1000, random_state=42, class_weight='balanced'):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            warm_start=True
        )
        self.fitted = False
    
    def train(self, X, y, sample_weight=None):
        """
        Entraîne le modèle sur les données locales
        """
        self.model.fit(X, y, sample_weight=sample_weight)
        self.fitted = True
        return self
    
    def predict(self, X):
        """
        Prédit les classes pour les données d'entrée
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Prédit les probabilités des classes pour les données d'entrée
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        return self.model.predict_proba(X)
    
    def get_weights(self):
        """
        Retourne les poids du modèle sous forme de tableau numpy 1D
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant d'extraire les poids")
        
        return np.concatenate([self.model.intercept_, self.model.coef_[0]])
    
    def set_weights(self, weights):
        """
        Met à jour les poids du modèle à partir d'un tableau numpy 1D
        """
        if len(weights) < 2:
            raise ValueError("Le vecteur de poids doit inclure l'intercept et les coefficients")
        
        self.model.intercept_ = np.array([weights[0]])
        self.model.coef_ = weights[1:].reshape(1, -1)
        self.fitted = True