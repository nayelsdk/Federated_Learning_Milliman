import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score

class FederatedLearning:
    """
    Coordonnateur de l'apprentissage fédéré
    
    Si le modèle utilisé est une régression logistique (type statsmodels),
    un seul cycle d'entraînement suffit. Pour les modèles de type réseau de neurones,
    l'entraînement peut être réalisé sur plusieurs itérations via l'argument num_rounds.
    """

    def __init__(self, data_dict, features, target, model_class, server=None, **model_params):
        from .server import FederatedServer

        self.data_dict = data_dict
        self.features = features
        self.target = target
        self.model_class = model_class
        self.model_params = model_params
        self.server = server if server else FederatedServer()
        self.history_weights = []
        self.history_metrics = []

    def setup(self):
        from .client import FederatedClient
        from imblearn.over_sampling import SMOTE

        for client_name, data in self.data_dict.items():
            try:
                print(f"Configuration du client '{client_name}' avec {len(data)} observations")

                model = self.model_class(**self.model_params)

                client = FederatedClient(
                    name=client_name,
                    data=data,
                    features=self.features,
                    target=self.target,
                    model=model
                )

                self.server.add_client(client)
                print(f"Client '{client_name}' ajouté avec succès")

            except Exception as e:
                print(f"Erreur lors de la configuration du client '{client_name}': {str(e)}")


    def train(self, num_rounds=1):
        print(f"\nDémarrage de l'entraînement fédéré ({num_rounds} cycle(s))...")
        for _ in range(num_rounds):
            global_weights = self.server.train_global_model(1)
            self.history_weights.append(global_weights.copy())

            round_metrics = {}
            for client in self.server.clients:
                metrics = client.evaluate_model()
                round_metrics[client.name] = metrics
            self.history_metrics.append(round_metrics)

        return self.history_weights[-1]

    def visualize_diagnostics(self):
        self.plot_weights_evolution()
        self.plot_client_metrics('f1')
        self.summarize_class_balance()
        self.plot_proba_distribution()
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.display_statistical_tests()

    def display_statistical_tests(self):
        print("\nTest statistique des coefficients du premier client :")
        try:
            first_model = self.server.clients[0].model
            feature_names = ['Intercept'] + self.features

            if hasattr(first_model, 'results'):
                res = first_model.results
                for i, name in enumerate(feature_names):
                    coef = res.params[i]
                    pval = res.pvalues[i]
                    stderr = res.bse[i]
                    ci = res.conf_int().iloc[i]
                    print(f"{name:15}: coef = {coef:.4f}, p = {pval:.4g}, SE = {stderr:.4f}, 95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                print("  Résultats statsmodels non disponibles pour ce modèle.")
        except Exception as e:
            print(f"[Erreur test stat] : {e}")

    def plot_weights_evolution(self):
        if not self.history_weights:
            print("Aucun poids enregistré. Lancez d'abord l'entraînement.")
            return

        weight_matrix = np.vstack(self.history_weights)
        feature_names = ['Intercept'] + self.features

        plt.figure(figsize=(10, 6))
        for i, name in enumerate(feature_names):
            plt.plot(weight_matrix[:, i], label=name)

        plt.title("Évolution des poids globaux par cycle")
        plt.xlabel("Cycle fédéré")
        plt.ylabel("Poids")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_client_metrics(self, metric_name='f1'):
        if not self.history_metrics:
            print("Aucune métrique enregistrée. Lancez d'abord l'entraînement.")
            return

        client_names = list(self.history_metrics[0].keys())
        rounds = list(range(1, len(self.history_metrics) + 1))

        plt.figure(figsize=(10, 6))
        for client in client_names:
            values = [round_metrics[client].get(metric_name, 0.0) for round_metrics in self.history_metrics]
            plt.plot(rounds, values, label=client)

        plt.title(f"Évolution de la métrique '{metric_name}' par client")
        plt.xlabel("Cycle fédéré")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summarize_class_balance(self):
        print("\nDistribution des classes cibles par client :")
        for name, df in self.data_dict.items():
            if self.target in df.columns:
                counts = df[self.target].value_counts().sort_index()
                print(f"\n{name} :")
                for cls, count in counts.items():
                    print(f"  Classe {cls} : {count} exemples")

    def plot_proba_distribution(self):
        print("\nDistribution des probabilités prédites par client :")
        for client in self.server.clients:
            try:
                proba = client.model.predict_proba(client.X_test)
                if proba.ndim == 2:
                    proba = proba[:, 1]
                plt.figure(figsize=(8, 4))
                plt.hist(proba, bins=50, alpha=0.7, edgecolor='k')
                plt.title(f"Distribution des probabilités - {client.name}")
                plt.xlabel("Probabilité prédite")
                plt.ylabel("Nombre d'observations")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"[Erreur] Client {client.name} : {str(e)}")

    def plot_roc_curves(self):
        print("\nCourbes ROC par client :")
        for client in self.server.clients:
            try:
                y_true = client.y_test
                y_score = client.model.predict_proba(client.X_test)
                if y_score.ndim == 2:
                    y_score = y_score[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc_score = roc_auc_score(y_true, y_score)
                plt.plot(fpr, tpr, label=f"{client.name} (AUC = {auc_score:.3f})")
            except Exception as e:
                print(f"[Erreur ROC] {client.name}: {str(e)}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("Taux de faux positifs")
        plt.ylabel("Taux de vrais positifs")
        plt.title("Courbes ROC des clients")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_precision_recall_curves(self):
        print("\nCourbes Precision-Recall par client :")
        for client in self.server.clients:
            try:
                y_true = client.y_test
                y_score = client.model.predict_proba(client.X_test)
                if y_score.ndim == 2:
                    y_score = y_score[:, 1]
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                plt.plot(recall, precision, label=f"{client.name} (AUC = {pr_auc:.3f})")
            except Exception as e:
                print(f"[Erreur PR] {client.name}: {str(e)}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Courbes Precision-Recall des clients")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def find_best_thresholds(self, metric='f1'):
        print("\nSeuils optimaux par client selon la métrique :", metric)
        for client in self.server.clients:
            try:
                y_true = client.y_test
                y_score = client.model.predict_proba(client.X_test)
                if y_score.ndim == 2:
                    y_score = y_score[:, 1]
                thresholds = np.linspace(0.01, 0.5, 100)
                best_score = -1
                best_threshold = 0.0
                for t in thresholds:
                    y_pred = (y_score >= t).astype(int)
                    score = f1_score(y_true, y_pred, zero_division=0) if metric == 'f1' else 0
                    if score > best_score:
                        best_score = score
                        best_threshold = t
                print(f"  {client.name} : meilleur seuil = {best_threshold:.3f}, {metric} = {best_score:.4f}")
            except Exception as e:
                print(f"[Erreur seuil] {client.name}: {str(e)}")
    
    def plot_model_coefficients(self):
        try:
            first_model = self.server.clients[0].model
            feature_names = ['Intercept'] + self.features

            if not hasattr(first_model, 'results'):
                print("[Erreur] Les résultats statsmodels ne sont pas disponibles.")
                return

            res = first_model.results
            coef = res.params.values
            pvals = res.pvalues.values

            sorted_idx = np.argsort(np.abs(coef))[::-1]
            coef_sorted = coef[sorted_idx]
            pval_sorted = pvals[sorted_idx]
            features_sorted = [feature_names[i] for i in sorted_idx]

            colors = []
            for p, c in zip(pval_sorted, coef_sorted):
                if p < 0.05:
                    colors.append('#2ecc71' if c >= 0 else '#e74c3c')  # vert ou rouge si significatif
                else:
                    colors.append('lightgray')  # pas significatif

            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(coef_sorted)), coef_sorted, color=colors, edgecolor='black')
            for i, v in enumerate(coef_sorted):
                plt.text(v + 0.01 if v >= 0 else v - 0.1, i, f"{v:.4f}", va='center')

            plt.yticks(range(len(features_sorted)), features_sorted)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            plt.title("Visualisation des coefficients du modèle et leur significativité")
            plt.xlabel("Valeur du coefficient")
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Erreur plot coeffs] : {e}")

