import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve
)

class FederatedClient:
    """
    Classe représentant un client dans le système d'apprentissage fédéré,
    utilisant une matrice de coût asymétrique.
    """

    def __init__(self, name, data, features, target, model, test_size=0.2, random_state=42, cost_matrix=None):
        self.name = name
        self.data = data
        self.features = features
        self.target = target
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.cost_matrix = cost_matrix or {(0, 1): 1.0, (1, 0): 5.0}
        self.prepare_data()


    def prepare_data(self):
        if "Exposure" not in self.data.columns:
            raise ValueError(f"'Exposure' requise pour {self.name}")

        X = self.data[self.features]
        y = self.data[self.target]
        exposure = self.data["Exposure"]

        X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
            X, y, exposure, test_size=self.test_size, random_state=self.random_state
        )

        self.X_train = X_train.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        self.exposure_train = exposure_train.reset_index(drop=True)
        self.exposure_test = exposure_test.reset_index(drop=True)


    def compute_cost_sensitive_weights(self):
        """
        Calcule un poids par observation basé sur la matrice de coût et l'exposition.
        """
        weights = []
        for i in range(len(self.y_train)):
            true_label = int(self.y_train[i])
            if true_label == 0:
                cost = self.cost_matrix.get((0, 1), 1.0)  # Faux positif
            else:
                cost = self.cost_matrix.get((1, 0), 1.0)  # Faux négatif
            weights.append(cost * self.exposure_train[i])
        return np.array(weights)


    def train_local_model(self):
        combined_weights = self.compute_cost_sensitive_weights()
        
        # Affiche les paramètres du modèle avant l'entraînement
        if hasattr(self.model, 'alpha'):
            print(f"[DEBUG train_local_model] Client {self.name} utilise alpha={self.model.alpha}")
        
        # Entraînement du modèle avec les poids déjà mis à jour
        self.model.train(self.X_train, self.y_train, sample_weight=combined_weights)
        
        weights = self.model.get_weights()
        print(f"\nPoids du modèle local pour {self.name}:")
        feature_names = ['Intercept'] + self.features
        for name, weight in zip(feature_names, weights):
            print(f" {name}: {weight:.6f}")
        
        self.evaluate_model()
        return weights


    def evaluate_model(self, threshold=None):
        try:
            y_proba = self.model.predict_proba(self.X_test)
            if isinstance(y_proba, np.ndarray):
                proba = y_proba
                if y_proba.ndim > 1:
                    proba = y_proba[:, 1]
            else:
                proba = y_proba.values

            fpr, tpr, thresholds = roc_curve(self.y_test, proba)
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            threshold = thresholds[best_idx]
            print(f"[INFO] Seuil optimal (indice de Youden) pour {self.name} : {threshold:.4f}")

            y_pred = (proba >= threshold).astype(int)
            print(f"[DEBUG] Prédictions positives : {np.sum(y_pred)} / {len(y_pred)}")
            print(f"[DEBUG] Répartition y_test : {np.bincount(self.y_test)}")

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
        except Exception as e:
            print(f"[Erreur] évaluation impossible pour {self.name}: {e}")
            return {}

    def update_weights(self, weights):
        """
        Met à jour les poids du modèle local avec les poids globaux
        """
        try:
            self.model.set_weights(weights)
            print(f"[DEBUG] Poids mis à jour pour le client {self.name}")
            # Affichage des paramètres de régularisation actuels
            if hasattr(self.model, 'alpha'):
                print(f"[DEBUG] Alpha actuel pour {self.name} = {self.model.alpha}")
        except Exception as e:
            print(f"[Erreur] Impossible de mettre à jour les poids pour {self.name}: {e}")
