import argparse
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import json
import warnings
from load_csv_data import load_csv_data  # Assurez-vous que cette fonction est définie correctement.

class SklearnFederatedClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        X_train, X_test, y_train, y_test, exposure_train, exposure_test = load_csv_data(
            client_id, include_exposure=True
        )

        self.scaler = StandardScaler()
        self.x_train = self.scaler.fit_transform(X_train)
        self.x_test = self.scaler.transform(X_test)

        self.y_train = y_train
        self.y_test = y_test
        self.exposure_train = exposure_train
        self.exposure_test = exposure_test

        self.model = LogisticRegression(
            class_weight="balanced",
            fit_intercept=True,
            max_iter=1,
            warm_start=True,
            random_state=42,
            solver="saga",
        )

        # Liste pour enregistrer les poids de chaque client et les scores à chaque itération
        self.coefs_history = []
        self.f1_history = []
        self.accuracy_history = []

        # Création d'un répertoire pour sauvegarder les poids des modèles si nécessaire
        if not os.path.exists(f"client_{self.client_id}_weights"):
            os.makedirs(f"client_{self.client_id}_weights")

    def get_parameters(self, config):
        if not hasattr(self.model, "coef_"):
            self.model.fit(
                self.x_train[:1], self.y_train[:1], sample_weight=self.exposure_train[:1]
            )
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(
            self.x_train, self.y_train, sample_weight=self.exposure_train
        )
        
        # Enregistrer les poids (coefficients) du modèle pour chaque client
        self.coefs_history.append(self.model.coef_)

        # Évaluation du modèle sur les données de test
        y_pred = self.model.predict(self.x_test)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        accuracy = accuracy_score(self.y_test, y_pred)

        # Enregistrer les scores de performance
        self.f1_history.append(f1)
        self.accuracy_history.append(accuracy)

        # Sauvegarder les poids dans un fichier local après chaque itération
        np.save(f"client_{self.client_id}_weights/weights_round_{config['round']}.npy", self.model.coef_)

        return self.get_parameters(config), len(self.x_train), {"f1_score": f1, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.x_test)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        accuracy = accuracy_score(self.y_test, y_pred)
        return 0.0, len(self.y_test), {"f1_score": f1, "accuracy": accuracy}

    def plot_performance(self):
        """Tracer la performance du modèle au fil du temps (F1 et accuracy)."""
        plt.figure(figsize=(12, 6))

        # Tracer le F1 score
        plt.subplot(1, 2, 1)
        plt.plot(self.f1_history, label="F1 Score", color='blue')
        plt.xlabel("Itérations")
        plt.ylabel("F1 Score")
        plt.title(f"Evolution du F1 Score pour le client {self.client_id}")
        plt.grid(True)
        plt.legend()

        # Tracer l'accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_history, label="Accuracy", color='green')
        plt.xlabel("Itérations")
        plt.ylabel("Accuracy")
        plt.title(f"Evolution de l'Accuracy pour le client {self.client_id}")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_performance(self):
        """Sauvegarder les scores F1 et Accuracy dans un fichier JSON."""
        performance_data = {
            "f1_score_history": self.f1_history,
            "accuracy_history": self.accuracy_history,
        }
        with open(f"client_{self.client_id}_performance.json", 'w') as f:
            json.dump(performance_data, f)

if __name__ == "__main__":
    from flwr.client import start_client

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    client = SklearnFederatedClient(client_id=args.client_id)
    start_client(server_address="127.0.0.1:9092", client=client.to_client())

    # Appeler plot_performance après l'entraînement pour afficher les performances
    client.plot_performance()
    client.save_performance()  # Sauvegarder les performances dans un fichier JSON
