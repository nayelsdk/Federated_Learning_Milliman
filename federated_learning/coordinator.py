import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from .client import FederatedClient

class FederatedLearning:
    """
    Coordonnateur de l'apprentissage fédéré.
    Fournit les méthodes de configuration des clients, d'entraînement global
    et de visualisation des résultats, y compris les courbes ROC/PR, la distribution
    des probabilités, les poids et les coefficients.
    """

    def __init__(self, data_dict, features, target, model_class, server=None, **model_params):
        from .server import FederatedServer
        self.data_dict = data_dict
        self.features = features
        self.target = target
        self.model_class = model_class
        self.model_params = model_params
        print(f"[DEBUG FederatedLearning.__init__] Paramètres reçus: {model_params}")
        self.server = server if server else FederatedServer()
        self.history_weights = []
        self.history_metrics = []

    def setup(self, class_weights=None):
        for client_name, data in self.data_dict.items():
            try:
                print(f"Configuration du client '{client_name}' avec {len(data)} observations")
                # Nettoyage des données
                print(f"Nettoyage des données pour {client_name} (NaN ou inf)...")
                X = data[self.features].replace([np.inf, -np.inf], np.nan)
                combined = pd.concat([X, data[[self.target, "Exposure"]]], axis=1)
                cleaned = combined.dropna()
                cleaned_data = cleaned.copy()
                cleaned_data[self.features] = cleaned_data[self.features].astype(float)
                print(f" => {len(cleaned_data)} lignes restantes après nettoyage")

                # Définir les poids des classes si demandé
                # Création du modèle avec les bons paramètres
                print(f"[DEBUG setup] Création du modèle avec paramètres: {self.model_params}")
                if isinstance(class_weights, dict):
                    model = self.model_class(class_weight=class_weights, **self.model_params)
                else:
                    model = self.model_class(**self.model_params)
                
                client = FederatedClient(
                    name=client_name,
                    data=cleaned_data,
                    features=self.features,
                    target=self.target,
                    model=model,
                    cost_matrix={ (0, 1): 1.0, (1, 0): 5.0 } 
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
        return self.history_weights[-1]

    def visualize_diagnostics(self):
        self.plot_weights_evolution()
        self.plot_proba_distribution()
        self.plot_model_coefficients()
        self.compare_local_vs_global_roc()
        self.compare_local_vs_global_pr()

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

    def plot_model_coefficients(self):
        try:
            first_model = self.server.clients[0].model
            feature_names = ['Intercept'] + self.features

            if not hasattr(first_model, 'results'):
                print("[Erreur] Les résultats statsmodels ne sont pas disponibles.")
                return

            res = first_model.results
            coef = res.params.values
            sorted_idx = np.argsort(np.abs(coef))[::-1]
            coef_sorted = coef[sorted_idx]
            features_sorted = [feature_names[i] for i in sorted_idx]

            # Vérifie la présence des p-values
            has_pvalues = hasattr(res, 'pvalues') and res.pvalues is not None

            if has_pvalues:
                pval_sorted = res.pvalues.values[sorted_idx]
                colors = [
                    '#2ecc71' if p < 0.05 and c >= 0 else
                    '#e74c3c' if p < 0.05 and c < 0 else
                    'lightgray'
                    for p, c in zip(pval_sorted, coef_sorted)
                ]
            else:
                print("[Info] Pas de p-values disponibles : affichage simple des coefficients.")
                colors = ['#3498db' if c >= 0 else '#e67e22' for c in coef_sorted]  # bleu / orange

            # Affichage
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(coef_sorted)), coef_sorted, color=colors, edgecolor='black')
            for i, v in enumerate(coef_sorted):
                plt.text(v + 0.01 if v >= 0 else v - 0.1, i, f"{v:.4f}", va='center')

            plt.yticks(range(len(features_sorted)), features_sorted)
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            plt.title("Visualisation des coefficients du modèle")
            plt.xlabel("Valeur du coefficient")
            plt.grid(axis='x', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"[Erreur plot coeffs] : {e}")

    def compare_local_vs_global_roc(self):
        print("\nComparaison des courbes ROC : modèle local vs modèle global")
        global_weights = self.history_weights[-1] if self.history_weights else None
        
        for client in self.server.clients:
            try:
                y_true = client.y_test
                y_local_score = client.model.predict_proba(client.X_test)
                if y_local_score.ndim == 2:
                    y_local_score = y_local_score[:, 1]
                auc_local = roc_auc_score(y_true, y_local_score)
                
                # Utilisez les mêmes paramètres pour le modèle global
                model_global = self.model_class(**self.model_params)
                model_global.train(client.X_train, client.y_train, sample_weight=client.exposure_train)
                model_global.set_weights(global_weights)
                
                y_global_score = model_global.predict_proba(client.X_test)
                if y_global_score.ndim == 2:
                    y_global_score = y_global_score[:, 1]
                auc_global = roc_auc_score(y_true, y_global_score)
                
                # Création des courbes ROC
                fpr_local, tpr_local, _ = roc_curve(y_true, y_local_score)
                fpr_global, tpr_global, _ = roc_curve(y_true, y_global_score)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr_local, tpr_local, label=f"Local (AUC = {auc_local:.3f})")
                plt.plot(fpr_global, tpr_global, label=f"Global (AUC = {auc_global:.3f})")
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('Taux de faux positifs')
                plt.ylabel('Taux de vrais positifs')
                plt.title(f'Courbe ROC - {client.name}')
                plt.legend(loc="lower right")
                plt.show()
            except Exception as e:
                print(f"[Erreur] Client {client.name} : {str(e)}")


    def compare_local_vs_global_pr(self):
        print("\nComparaison des courbes Precision-Recall : modèle local vs modèle global")
        global_weights = self.history_weights[-1] if self.history_weights else None

        for client in self.server.clients:
            try:
                y_true = client.y_test
                y_local_score = client.model.predict_proba(client.X_test)
                if y_local_score.ndim == 2:
                    y_local_score = y_local_score[:, 1]
                precision_local, recall_local, _ = precision_recall_curve(y_true, y_local_score)
                auc_local = auc(recall_local, precision_local)

                model_global = self.model_class(**self.model_params)
                model_global.train(client.X_train, client.y_train, sample_weight=client.exposure_train)
                model_global.set_weights(global_weights)
                y_global_score = model_global.predict_proba(client.X_test)
                if y_global_score.ndim == 2:
                    y_global_score = y_global_score[:, 1]
                precision_global, recall_global, _ = precision_recall_curve(y_true, y_global_score)
                auc_global = auc(recall_global, precision_global)

                plt.figure(figsize=(8, 6))
                plt.plot(recall_local, precision_local, label=f"Local (AUC = {auc_local:.3f})")
                plt.plot(recall_global, precision_global, label=f"Global (AUC = {auc_global:.3f})")
                plt.title(f"Courbe Precision-Recall - {client.name}")
                plt.xlabel("Recall")
                plt.ylabel("Precision")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.show()

            except Exception as e:
                print(f"[Erreur comparaison PR] {client.name} : {e}")
    
    def compare_cross_client_performance(self):
        """
        Compare les performances des modèles locaux sur toutes les bases + du modèle global sur chaque base.
        """
        print("\nComparaison croisée des performances entre clients (modèles locaux vs modèle global)...")

        clients = self.server.clients
        n = len(clients)
        client_names = [c.name for c in clients]
        global_weights = self.history_weights[-1] if self.history_weights else None

        records = []

        for source_client in clients:
            source_name = source_client.name
            local_weights = source_client.model.get_weights()

            for target_client in clients:
                target_name = target_client.name

                # Tester les poids locaux du client source
                model_local = self.model_class(**self.model_params)
                model_local.train(target_client.X_train, target_client.y_train, sample_weight=target_client.exposure_train)
                model_local.set_weights(local_weights)
                y_true = target_client.y_test
                y_score = model_local.predict_proba(target_client.X_test)
                if y_score.ndim == 2:
                    y_score = y_score[:, 1]
                auc_local = roc_auc_score(y_true, y_score)
                records.append((f"Local_{source_name}", target_name, auc_local))

        # Ajouter le modèle global (même poids pour tous)
        for target_client in clients:
            model_global = self.model_class(**self.model_params)
            model_global.train(target_client.X_train, target_client.y_train, sample_weight=target_client.exposure_train)
            model_global.set_weights(global_weights)
            y_true = target_client.y_test
            y_score = model_global.predict_proba(target_client.X_test)
            if y_score.ndim == 2:
                y_score = y_score[:, 1]
            auc_global = roc_auc_score(y_true, y_score)
            records.append(("Federated", target_client.name, auc_global))

        # Construction DataFrame
        df = pd.DataFrame(records, columns=["Source", "Cible", "AUC"])
        pivot = df.pivot(index="Source", columns="Cible", values="AUC")

        print("\nMatrice AUC croisés (modèles locaux + global) :\n")
        print(pivot.round(4))

        # Affichage heatmap avec valeur dans chaque case
        plt.figure(figsize=(10, 6))
        plt.title("Performance croisée des modèles (Local & Federated) - AUC")
        im = plt.imshow(pivot.values, cmap="coolwarm", vmin=0.5, vmax=1)
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.colorbar(im, label="AUC")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                plt.text(j, i, f"{val:.2f}", ha='center', va='center', color='black')

        plt.tight_layout()
        plt.show()
