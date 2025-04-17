import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_curve, log_loss
from scipy.special import expit
import matplotlib.pyplot as plt
import os


def youden_index_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    specificity = 1 - fpr
    youden_index = tpr + specificity - 1
    best_index = youden_index.argmax()
    best_threshold = thresholds[best_index]
    return best_threshold, youden_index[best_index]


def get_coefficients(model, X_train):
    coefficients = dict(zip(X_train.columns, model.coef_[0]))
    coefficients['Intercept'] = model.intercept_[0]
    return coefficients


def apply_coefficients(X, coefs):
    intercept = coefs.get("Intercept", 0)
    coef_values = [coefs.get(col, 0) for col in X.columns]
    logits = np.dot(X.values, coef_values) + intercept
    return expit(logits)


class FederatedLogisticRegression:
    def __init__(self, df: pd.DataFrame, local_epochs: int = 5):
        self.df = df.dropna()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.exposure_train = self.X_train['Exposure']
        self.exposure_test = self.X_test['Exposure']
        self.X_train = self.X_train.drop('Exposure', axis=1)
        self.X_test = self.X_test.drop('Exposure', axis=1)
        self.local_epochs = local_epochs
        self.scaler = StandardScaler()
        self.model = SGDClassifier(random_state=42, loss="log_loss", penalty='l2',
                                   fit_intercept=True, max_iter=1, warm_start=True,
                                   learning_rate='adaptive', eta0=0.01)
        self.model.classes_ = np.array([0, 1])
        self.best_threshold = None
        self.loss_history = []

    def split_data(self, test_size=0.3):
        X = self.df.drop('Sinistre', axis=1)
        y = self.df['Sinistre']
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    def normalize(self):
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.X_train.columns, index=self.X_train.index)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.X_test.columns, index=self.X_test.index)

    def train(self):
        self.normalize()
        classes = np.unique(self.y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=self.y_train)
        sample_weights = self.y_train.map(dict(zip(classes, class_weights)))

        for epoch in range(self.local_epochs):
            if hasattr(self.model, 'coef_'):
                print("\n[SGD Epoch {}/{}] Coefficients before SGD: {}".format(epoch + 1, self.local_epochs, self.model.coef_))
            self.model.partial_fit(self.X_train, self.y_train, classes=classes, sample_weight=sample_weights)
            y_pred_proba = self.model.predict_proba(self.X_train)[:, 1]
            loss = log_loss(self.y_train, y_pred_proba, sample_weight=sample_weights)
            self.loss_history.append(loss)
            print("[SGD Epoch {}/{}] Coefficients after SGD:  {}".format(epoch + 1, self.local_epochs, self.model.coef_))
            print("[SGD Epoch {}/{}] Log-loss: {:.5f}".format(epoch + 1, self.local_epochs, loss))

        y_proba = self.model.predict_proba(self.X_test)[:, 1] * self.exposure_test
        self.best_threshold, _ = youden_index_threshold(self.y_test, y_proba)
        y_pred = (y_proba >= self.best_threshold).astype(int)
        coefs = get_coefficients(self.model, self.X_train)
        return y_proba, y_pred, coefs

    def update_model_with_coefs(self, coefs: dict):
        intercept = coefs.get("Intercept", 0)
        coef_array = np.array([coefs.get(col, 0) for col in self.X_train.columns])
        self.model.coef_ = coef_array.reshape(1, -1)
        self.model.intercept_ = np.array([intercept])
        self.model.classes_ = np.array([0, 1])

    def predict_with_global(self, coefs: dict):
        y_proba = apply_coefficients(self.X_test, coefs) * self.exposure_test
        y_pred = (y_proba >= self.best_threshold).astype(int)
        return y_proba, y_pred

    def plot_loss_history(self):
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o')
        plt.title("Évolution de la log-loss par époque")
        plt.xlabel("Époque")
        plt.ylabel("Log-loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def federated_averaging(client_dicts):
    total_samples = sum(len(client['lr'].X_train) for client in client_dicts)
    weights = [len(client['lr'].X_train) / total_samples for client in client_dicts]

    aggregated = {}
    keys = client_dicts[0]['coefs'].keys()
    for key in keys:
        weighted_sum = sum(client['coefs'][key] * weights[i] for i, client in enumerate(client_dicts))
        aggregated[key] = weighted_sum
    return aggregated


def simulate_federated_learning(clients_data, rounds=5, local_epochs=5):
    clients = []

    for df in clients_data:
        client = {
            'lr': FederatedLogisticRegression(df, local_epochs=local_epochs),
            'X_test': None,
            'y_test': None,
            'y_proba_local': None,
            'y_pred_local': None,
            'y_proba_global': None,
            'y_pred_global': None,
            'coefs': None
        }
        clients.append(client)

    clients_snapshots = [[] for _ in clients]

    for r in range(rounds):
        print(f"Round {r + 1}/{rounds}")
        for i, client in enumerate(clients):
            y_proba, y_pred, coefs = client['lr'].train()
            client['X_test'] = client['lr'].X_test
            client['y_test'] = client['lr'].y_test
            client['y_proba_local'] = y_proba
            client['y_pred_local'] = y_pred
            client['coefs'] = coefs
            clients_snapshots[i].append(coefs.copy())

        global_coefs = federated_averaging(clients)

        for client in clients:
            client['lr'].update_model_with_coefs(global_coefs)
            y_proba, y_pred = client['lr'].predict_with_global(global_coefs)
            client['y_proba_global'] = y_proba
            client['y_pred_global'] = y_pred

    return clients, clients_snapshots


def plot_all_weights(clients_snapshots, export_path="weights_tracking", save_plots=True, save_csv=True):
    os.makedirs(export_path, exist_ok=True)

    n_rounds = len(clients_snapshots[0])
    all_features = list(clients_snapshots[0][0].keys())

    for feature in all_features:
        plt.figure(figsize=(10, 4))
        data_dict = {}

        for client_id, client_rounds in enumerate(clients_snapshots):
            weights = [round_coefs.get(feature, 0) for round_coefs in client_rounds]
            label = f'Client {client_id + 1}'
            data_dict[label] = weights
            plt.plot(range(n_rounds), weights, marker='o', label=label)

        for r in range(n_rounds):
            plt.axvline(x=r, color='grey', linestyle='--', linewidth=0.5)

        plt.title(f"\u00c9volution du poids pour '{feature}'")
        plt.xlabel("Round FL")
        plt.ylabel("Poids")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_plots:
            plt.savefig(os.path.join(export_path, f"{feature}_weight_plot.png"))

        plt.show()

        if save_csv:
            df_weights = pd.DataFrame(data_dict)
            df_weights.index.name = "Round"
            df_weights.to_csv(os.path.join(export_path, f"{feature}_weights.csv"))
            


def plot_all_clients_losses(clients):
    plt.figure(figsize=(12, 6))
    for i, client in enumerate(clients):
        if hasattr(client['lr'], 'loss_history'):
            plt.plot(client['lr'].loss_history, label=f'Client {i+1}')
    plt.title("Évolution de la log-loss par client (époques locales)")
    plt.xlabel("Époque locale")
    plt.ylabel("Log-loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    df_fr = pd.read_csv("data/french_data.csv")
    df_be = pd.read_csv("data/belgium_data.csv")

    clients, snap = simulate_federated_learning([df_fr, df_be], rounds=10, local_epochs=5)
    plot_all_clients_losses(clients)

    print("Simulation terminée.")
if __name__ == "__main__":
    main()
