import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FederatedClient:
    """
    Classe représentant un client dans le système d'apprentissage fédéré
    """
    
    def __init__(self, name, data, features, target, model, test_size=0.2, random_state=42):
        """
        Initialise un client fédéré
        
        Args:
            name: Identifiant du client
            data: DataFrame contenant les données
            features: Liste des colonnes utilisées comme caractéristiques après le preprocessing
            target: Nom de la colonne cible
            model: Instance d'un modèle héritant de BaseModel
            test_size: Proportion des données pour le test
            random_state: Graine aléatoire pour la reproductibilité
        """
        self.name = name
        self.data = data
        self.features = features
        self.target = target
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.prepare_data()
    
    def prepare_data(self):
        """
        Prépare les données pour l'entraînement
        """
        if "Exposure" not in self.data.columns:
            raise ValueError(f"'Exposure' requise pour {self.name}")
        
        X = self.data[self.features]
        y = self.data[self.target]
        exposure = self.data["Exposure"]
        
        split = train_test_split(
            X, y, exposure, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        self.X_train, self.X_test = split[0], split[1]
        self.y_train, self.y_test = split[2], split[3]
        self.exposure_train, self.exposure_test = split[4], split[5]
    
    def train_local_model(self):
        """
        Entraîne le modèle local
        """
        self.model.train(self.X_train, self.y_train, sample_weight=self.exposure_train)
        weights = self.model.get_weights()
        
        print(f"\nPoids du modèle local pour {self.name}:")
        feature_names = ['Intercept'] + self.features
        for name, weight in zip(feature_names, weights):
            print(f"  {name}: {weight:.6f}")
        
        self.evaluate_model()
        return weights
    
    def update_weights(self, global_weights):
        """
        Met à jour les poids du modèle avec les poids globaux
        """
        self.model.set_weights(global_weights)
    
    def evaluate_model(self, threshold=0.3):
        """
        Évalue le modèle sur les données de test
        
        Args:
            threshold: Seuil de décision pour la classification binaire
        """
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0)
        }
        
        print(f"Performance du modèle pour {self.name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name.capitalize()}: {value:.4f}")
        
        return metrics