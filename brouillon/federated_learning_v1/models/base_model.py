from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Classe abstraite pour tous les modèles d'apprentissage fédéré
    """
    
    @abstractmethod
    def train(self, X, y, sample_weight=None):
        """Entraîne le modèle sur les données locales"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Prédit les classes pour les données d'entrée"""
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """Prédit les probabilités des classes pour les données d'entrée"""
        pass
    
    @abstractmethod
    def get_weights(self):
        """Retourne les poids du modèle sous forme de tableau numpy 1D"""
        pass
    
    @abstractmethod
    def set_weights(self, weights):
        """Met à jour les poids du modèle à partir d'un tableau numpy 1D"""
        pass