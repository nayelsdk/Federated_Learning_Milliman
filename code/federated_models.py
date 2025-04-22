import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from federated_methods import federated_averaging
from plots import plot_coeff_evolution


class FederatedLogisticRegression:
    def __init__(self, df, local_epochs=1):
        self.df = df.dropna()
        self.local_epochs = local_epochs
        self.scaler = StandardScaler()

        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()
        self.exposure_train = self.X_train['Exposure']
        self.exposure_test = self.X_test['Exposure']
        self.X_train = self.X_train.drop('Exposure', axis=1)
        self.X_test = self.X_test.drop('Exposure', axis=1)

        self.X_train = self.normalize(self.X_train, fit=True)
        self.X_test = self.normalize(self.X_test, fit=False)

        # Ne pas setter max_iter ici, il sera contrôlé manuellement dans `train_and_log_coeffs`
        self.model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            penalty='l2',
            fit_intercept=True,
            warm_start=True,
            solver='lbfgs'
        )
        
    def tune_hyperparams(self):
        param_grid = {
            'C': np.logspace(-4, 4, 10),
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }

        lr = LogisticRegression(
            class_weight="balanced",
            fit_intercept=True,
            max_iter=100,
            random_state=42
        )

        grid = GridSearchCV(lr, param_grid, scoring='roc_auc', cv=3)
        grid.fit(self.X_train, self.y_train, sample_weight=self.exposure_train)

        best_params = grid.best_params_
        print(f"🔍 Meilleurs hyperparams pour ce client : {best_params}")

        # Mémorise les meilleurs params
        self.best_params = best_params

    def choose_model(self):
        if not hasattr(self, 'best_params'):
            self.tune_hyperparams()

            self.model = LogisticRegression(
                **self.best_params,
                class_weight="balanced",
                fit_intercept=True,
                max_iter=1,  # <== 1 itération à chaque epoch (voir train_and_log_coeffs)
                warm_start=True,
                random_state=42
            )

    def split_data(self, test_size=0.40):
        X = self.df.drop('Sinistre', axis=1)
        y = self.df['Sinistre']
        return train_test_split(X, y, test_size=test_size, random_state=42)

    def normalize(self, X, fit=False):
        if fit:
            return pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        else:
            return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

    def train(self):
        self.model.fit(self.X_train, self.y_train, sample_weight=self.exposure_train)

    def predict(self):
        probs = self.model.predict_proba(self.X_test)[:, 1]
        probs = probs * self.exposure_test.values  # application de f_i
        return probs

    def evaluate(self):
        y_scores = self.predict()
        return roc_auc_score(self.y_test, y_scores, sample_weight=self.exposure_test)

    def get_coefficients(self):
        coef_dict = dict(zip(self.X_train.columns, self.model.coef_[0]))
        coef_dict['Intercept'] = self.model.intercept_[0]
        return coef_dict

    def set_coefficients(self, coef_dict):
        self.model.coef_ = np.array([[coef_dict[k] for k in self.X_train.columns]])
        self.model.intercept_ = np.array([coef_dict['Intercept']])

    def train_and_log_coeffs(self, n_epochs: int):
        history = []
        self.choose_model()  # configure le modèle avec les bons hyperparams

        for epoch in range(n_epochs):
            self.model.fit(self.X_train, self.y_train, sample_weight=self.exposure_train)
            history.append(self.get_coefficients())
        return history

def federated_training(client_dicts, n_rounds=5, local_epochs=5):

    clients_history = [[] for _ in client_dicts]  # Historique des poids locaux
    global_history = []  # Liste de log des poids globaux

    for round_idx in range(n_rounds):
        print(f"\n--- Round fédéré {round_idx + 1}/{n_rounds} ---")

        # 1. Entraînement local
        for i, client in enumerate(client_dicts):
            print(f"--> {client['name']} - Entraînement local")

            local_weights = client["lr"].train_and_log_coeffs(local_epochs)
            clients_history[i].extend(local_weights)

            client["coefs"] = local_weights[-1]

        # 2. Agrégation FedAvg
        avg_coefs = federated_averaging(client_dicts)
        global_history.append(avg_coefs)

        for i, client in enumerate(client_dicts):
            client["lr"].set_coefficients(avg_coefs)
            client["coefs"] = avg_coefs

            clients_history[i].append(avg_coefs.copy())

    return clients_history, global_history






def main():
    features = ["Power","DriverAge","Fuel_type","Density","Sex", "Intercept"]
    local_epochs = 10
    rounds = 10
    df_fr = pd.read_csv("data/french_data.csv")
    df_be = pd.read_csv("data/belgium_data.csv")
    df_eu = pd.read_csv("data/european_data.csv")
    df_split = [df_fr, df_be,df_eu]

    client_dicts = []
    for i, df in enumerate(df_split):
        client_dicts.append({
            "name": f"Client {i+1}",
            "lr": FederatedLogisticRegression(df, local_epochs=local_epochs),
            "coefs": None
        })
if __name__ == "__main__":
    main()
