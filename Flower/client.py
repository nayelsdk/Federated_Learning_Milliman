import argparse
import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from load_csv_data import load_csv_data

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

        # Liste pour enregistrer les poids de chaque client
        self.coefs_history = []

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

        return self.get_parameters(config), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        y_pred = self.model.predict(self.x_test)
        f1 = f1_score(self.y_test, y_pred, average="macro")
        return 0.0, len(self.y_test), {"f1_score": f1}

    def plot_weights(self):
        """Tracer les poids du modèle au fil du temps"""
        # Transformer la liste des coefficients en un tableau numpy pour une manipulation plus facile
        coefs_array = np.array(self.coefs_history)

        # Tracer les coefficients pour chaque itération
        plt.figure(figsize=(10, 6))
        for i in range(coefs_array.shape[1]):  # Pour chaque feature (chaque colonne)
            plt.plot(coefs_array[:, i], label=f"Feature {i + 1}")

        plt.xlabel("Itérations")
        plt.ylabel("Poids")
        plt.title(f"Evolution des poids du modèle pour le client {self.client_id}")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    from flwr.client import start_client

    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()

    client = SklearnFederatedClient(client_id=args.client_id)
    start_client(server_address="127.0.0.1:9092", client=client.to_client()) 

    # Appeler plot_weights après l'entraînement pour afficher les poids
    client.plot_weights()
