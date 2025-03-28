import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FederatedClient:
    """
    Classe représentant un client dans le système d'apprentissage fédéré
    """

    def __init__(self, name, data, features, target, model, test_size=0.2, random_state=42):
        self.name = name
        self.data = data
        self.features = features
        self.target = target
        self.model = model
        self.test_size = test_size
        self.random_state = random_state
        self.prepare_data()

    def prepare_data(self):
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
        self.model.train(self.X_train, self.y_train, sample_weight=self.exposure_train)
        weights = self.model.get_weights()

        print(f"\nPoids du modèle local pour {self.name}:")
        feature_names = ['Intercept'] + self.features
        for name, weight in zip(feature_names, weights):
            print(f"  {name}: {weight:.6f}")

        self.evaluate_model()
        return weights

    def update_weights(self, global_weights):
        try:
            self.model.set_weights(global_weights)
        except NotImplementedError:
            print(f"[Info] Le modèle du client '{self.name}' ne supporte pas la mise à jour des poids.")

    def evaluate_model(self, threshold=0.11):
        try:
            y_proba = self.model.predict_proba(self.X_test)
            if isinstance(y_proba, np.ndarray):
                proba = y_proba  # sklearn ou torch : shape (n_samples,) ou (n_samples, 2)
                if y_proba.ndim > 1:
                    proba = y_proba[:, 1]  # sklearn style
            else:
                proba = y_proba.values  # pandas Series (statsmodels)

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
