import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import train_test_split
import time

# Définition de la fonction pour calculer l'indice de Youden
def calculate_youden_index(y_true, y_pred_proba):
    """
    Calcule l'indice de Youden (sensibilité + spécificité - 1) pour différents seuils
    et retourne le seuil optimal avec son indice correspondant.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculer l'indice de Youden pour chaque seuil
    youden_indices = tpr - fpr  # équivalent à: sensibilité + spécificité - 1
    
    # Trouver le seuil optimal (celui qui maximise l'indice de Youden)
    optimal_idx = np.argmax(youden_indices)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden_indices[optimal_idx]
    
    # Sensibilité et spécificité au seuil optimal
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    
    return {
        'threshold': optimal_threshold,
        'youden_index': optimal_youden,
        'sensitivity': optimal_sensitivity,
        'specificity': optimal_specificity,
        'all_thresholds': thresholds,
        'all_youden_indices': youden_indices
    }

# Classes clients
class BaseClient:
    def __init__(self, data_batches, client_id, local_epochs=3):
        """Initialise un client d'apprentissage fédéré"""
        self.client_id = client_id
        self.batches = data_batches
        self.n_rounds = len(data_batches)
        self.local_epochs = local_epochs
        self.n_samples = sum(len(batch[0]) for batch in data_batches)
        
        #  le modèle contrôle de l'intercept
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.01,  # Régularisation L2
            learning_rate="constant",
            eta0=0.01,  
            class_weight='balanced',  
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        
        # Initialisation des coeffs.
        n_features = data_batches[0][0].shape[1]
        self.model.classes_ = np.array([0, 1])
        self.model.coef_ = np.zeros((1, n_features))
        self.model.intercept_ = np.zeros(1)
        
        # Historique  poids
        self.weight_history = {'coef': [], 'intercept': []}
        
        # Historique loss
        self.loss_history = []
        
        # Imputer pour gérer les valeurs manquantes
        self.imputer = SimpleImputer(strategy='mean')

class FedAvgClient(BaseClient):
    def train_local_model(self, global_model, round_idx):
        """Entraîne le modèle local avec plusieurs époques sur le batch du round actuel"""
        # Copier les poids du modèle global
        if global_model is not None:
            self.model.coef_ = np.copy(global_model.coef_)
            self.model.intercept_ = np.copy(global_model.intercept_)
        
        # Récupérer le batch pour ce round
        X_batch, y_batch, _ = self.batches[round_idx]
        
        # Vérifier les classes présentes
        unique_classes = np.unique(y_batch)
        if len(unique_classes) < 2:
            print(f"⚠️ Batch {round_idx} du client {self.client_id} contient uniquement la classe {unique_classes[0]}")
            return self.model
        
        # Imputations si nécessaire
        if np.isnan(X_batch).any():
            X_batch = self.imputer.fit_transform(X_batch)
        
        # Entraînement local avec E époques
        for _ in range(self.local_epochs):
            self.model.partial_fit(X_batch, y_batch, classes=[0, 1])
        
        # Calculer la perte finale sur ce batch
        if np.isnan(X_batch).any():
            X_batch_clean = self.imputer.fit_transform(X_batch)
        else:
            X_batch_clean = X_batch

        y_pred = self.model.predict_proba(X_batch_clean)
        batch_loss = log_loss(y_batch, y_pred)
        self.loss_history.append(batch_loss)
        
        # Enregistrer les poids après entraînement
        self.weight_history['coef'].append(np.copy(self.model.coef_[0]))
        self.weight_history['intercept'].append(np.copy(self.model.intercept_[0]))
        
        return self.model

class FedProxClient(BaseClient):
    def __init__(self, data_batches, client_id, local_epochs=3, mu=0.01):
        super().__init__(data_batches, client_id, local_epochs)
        self.mu = mu  # Coefficient de régularisation proximal
    
    def train_local_model(self, global_model, round_idx):
        """Entraîne le modèle local avec régularisation proximale"""
        # Copier les poids du modèle global
        if global_model is not None:
            self.model.coef_ = np.copy(global_model.coef_)
            self.model.intercept_ = np.copy(global_model.intercept_)
            
            # Sauvegarder une copie des poids globaux pour la régularisation proximale
            global_weights = np.copy(global_model.coef_[0])
            global_intercept = np.copy(global_model.intercept_[0])
        else:
            global_weights = np.zeros_like(self.model.coef_[0])
            global_intercept = 0.0
        
        # Récupérer le batch pour ce round
        X_batch, y_batch, _ = self.batches[round_idx]
        
        # Vérifier les classes présentes
        unique_classes = np.unique(y_batch)
        if len(unique_classes) < 2:
            print(f"⚠️ Batch {round_idx} du client {self.client_id} contient uniquement la classe {unique_classes[0]}")
            return self.model
        
        # Imputations si nécessaire
        if np.isnan(X_batch).any():
            X_batch = self.imputer.fit_transform(X_batch)
        
        # Entraînement local avec terme proximal
        for _ in range(self.local_epochs):
            # Mise à jour standard avec SGD
            self.model.partial_fit(X_batch, y_batch, classes=[0, 1])
            
            # Appliquer la régularisation proximale manuellement
            prox_coef = self.model.coef_[0] - self.mu * (self.model.coef_[0] - global_weights)
            prox_intercept = self.model.intercept_[0] - self.mu * (self.model.intercept_[0] - global_intercept)
            
            self.model.coef_[0] = prox_coef
            self.model.intercept_[0] = prox_intercept
        
        # Calculer la perte finale sur ce batch
        if np.isnan(X_batch).any():
            X_batch_clean = self.imputer.fit_transform(X_batch)
        else:
            X_batch_clean = X_batch

        y_pred = self.model.predict_proba(X_batch_clean)
        batch_loss = log_loss(y_batch, y_pred)
        self.loss_history.append(batch_loss)
        
        # Enregistrer les poids après entraînement
        self.weight_history['coef'].append(np.copy(self.model.coef_[0]))
        self.weight_history['intercept'].append(np.copy(self.model.intercept_[0]))
        
        return self.model

class FedOptClient(BaseClient):
    def train_local_model(self, global_model, round_idx):
        """Entraîne le modèle local avec plusieurs époques sur le batch du round actuel"""
        # Copier les poids du modèle global
        if global_model is not None:
            self.model.coef_ = np.copy(global_model.coef_)
            self.model.intercept_ = np.copy(global_model.intercept_)
        
        # Récupérer le batch pour ce round
        X_batch, y_batch, _ = self.batches[round_idx]
        
        # Vérifier les classes présentes
        unique_classes = np.unique(y_batch)
        if len(unique_classes) < 2:
            print(f"⚠️ Batch {round_idx} du client {self.client_id} contient uniquement la classe {unique_classes[0]}")
            return self.model
        
        # Imputations si nécessaire
        if np.isnan(X_batch).any():
            X_batch = self.imputer.fit_transform(X_batch)
        
        # Entraînement local avec E époques
        for _ in range(self.local_epochs):
            self.model.partial_fit(X_batch, y_batch, classes=[0, 1])
        
        # Calculer la perte finale sur ce batch
        if np.isnan(X_batch).any():
            X_batch_clean = self.imputer.fit_transform(X_batch)
        else:
            X_batch_clean = X_batch

        y_pred = self.model.predict_proba(X_batch_clean)
        batch_loss = log_loss(y_batch, y_pred)
        self.loss_history.append(batch_loss)
        
        # Enregistrer les poids après entraînement
        self.weight_history['coef'].append(np.copy(self.model.coef_[0]))
        self.weight_history['intercept'].append(np.copy(self.model.intercept_[0]))
        
        return self.model

# Classes serveurs
class BaseServer:
    def __init__(self, clients, feature_names, clip_intercept=True):
        """Initialise le serveur d'agrégation"""
        self.clients = clients
        self.n_rounds = clients[0].n_rounds
        self.feature_names = feature_names
        self.clip_intercept = clip_intercept
        
        # Modèle global
        self.global_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.01,
            class_weight='balanced',
            max_iter=1,
            warm_start=True,
            random_state=42
        )
        
        # Initialisation
        n_features = clients[0].batches[0][0].shape[1]
        self.global_model.classes_ = np.array([0, 1])
        self.global_model.coef_ = np.zeros((1, n_features))
        self.global_model.intercept_ = np.zeros(1)
        
        # Historique
        self.weight_history = {'coef': [], 'intercept': []}
        self.auc_history = {client.client_id: [] for client in clients}
        self.auc_history['Global'] = []
        
        # Historique des indices de Youden
        self.youden_history = {client.client_id: [] for client in clients}
        self.youden_history['Global'] = []
    
    def aggregate_models(self, models, weights):
        """Agrège les modèles locaux en modèle global"""
        avg_coef = np.zeros_like(models[0].coef_)
        avg_intercept = np.zeros_like(models[0].intercept_)
        
        total_weight = sum(weights)
        for model, weight in zip(models, weights):
            avg_coef += (weight / total_weight) * model.coef_
            avg_intercept += (weight / total_weight) * model.intercept_
        
        self.global_model.coef_ = avg_coef
        
        # Contrôler l'intercept
        if self.clip_intercept:
            self.global_model.intercept_ = np.clip(avg_intercept, -5, 5)
        else:
            self.global_model.intercept_ = avg_intercept
        
        return self.global_model
    
    def train(self, X_test_dict, y_test_dict, exposure_test_dict, progress_bar=None, status_text=None):
        """Exécute l'entraînement fédéré et imprime les résultats par round"""
        imputer = SimpleImputer(strategy='mean')
        
        for round_idx in range(self.n_rounds):
            if status_text:
                status_text.text(f"Round {round_idx+1}/{self.n_rounds}")
            
            # Entraînement local
            local_models = []
            weights = []
            
            for client in self.clients:
                local_model = client.train_local_model(self.global_model, round_idx)
                local_models.append(local_model)
                weights.append(len(client.batches[round_idx][0]))
            
            # Agrégation
            self.global_model = self.aggregate_models(local_models, weights)
            
            # Enregistrer les poids globaux
            self.weight_history['coef'].append(np.copy(self.global_model.coef_[0]))
            self.weight_history['intercept'].append(np.copy(self.global_model.intercept_[0]))
            
            # Évaluation par client
            all_y_true = []
            all_y_pred = []
            
            for client in self.clients:
                X_test = X_test_dict[client.client_id]
                y_test = y_test_dict[client.client_id]
                exposure_test = exposure_test_dict[client.client_id]
                
                # Imputation si nécessaire
                if np.isnan(X_test).any():
                    X_test_clean = imputer.fit_transform(X_test)
                else:
                    X_test_clean = X_test
                
                # Prédiction avec le modèle global, ajustée par l'exposition
                y_pred = self.global_model.predict_proba(X_test_clean)[:, 1] * exposure_test
                
                # Calcul de l'AUC
                auc = roc_auc_score(y_test, y_pred)
                self.auc_history[client.client_id].append(auc)
                
                # Calcul de l'indice de Youden
                youden_results = calculate_youden_index(y_test, y_pred)
                self.youden_history[client.client_id].append(youden_results)
                
                # Pour AUC global et Youden global
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
            
            # AUC global
            global_auc = roc_auc_score(all_y_true, all_y_pred)
            self.auc_history['Global'].append(global_auc)
            
            # Youden global
            global_youden_results = calculate_youden_index(all_y_true, all_y_pred)
            self.youden_history['Global'].append(global_youden_results)
            
            if progress_bar:
                progress_bar.progress((round_idx + 1) / self.n_rounds)
        
        return self.global_model

class FedAvgServer(BaseServer):
    pass  # Hérite directement de BaseServer sans modifications

class FedProxServer(BaseServer):
    pass  # Hérite directement de BaseServer sans modifications

class FedOptServer(BaseServer):
    def __init__(self, clients, feature_names, clip_intercept=True, 
                 server_lr=0.1, beta1=0.9, beta2=0.99, tau=1e-3, optimizer="adam"):
        super().__init__(clients, feature_names, clip_intercept)
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.optimizer = optimizer.lower()
        
        # Initialiser les moments pour l'optimisation adaptative
        n_features = clients[0].batches[0][0].shape[1]
        self.m = np.zeros((1, n_features))  # Premier moment
        self.v = np.zeros((1, n_features))  # Second moment
        self.m_intercept = np.zeros(1)
        self.v_intercept = np.zeros(1)
    
    def aggregate_models(self, models, weights):
        """Agrège les modèles locaux avec optimisation adaptative"""
        # Calculer la moyenne pondérée des modèles locaux (comme FedAvg)
        avg_coef = np.zeros_like(models[0].coef_)
        avg_intercept = np.zeros_like(models[0].intercept_)
        
        total_weight = sum(weights)
        for model, weight in zip(models, weights):
            avg_coef += (weight / total_weight) * model.coef_
            avg_intercept += (weight / total_weight) * model.intercept_
        
        # Calculer la différence (delta) entre la moyenne et le modèle global actuel
        delta_coef = avg_coef - self.global_model.coef_
        delta_intercept = avg_intercept - self.global_model.intercept_
        
        # Mettre à jour les moments selon l'optimiseur choisi
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta_coef
        self.m_intercept = self.beta1 * self.m_intercept + (1 - self.beta1) * delta_intercept
        
        if self.optimizer == "adam":
            # Mise à jour Adam
            self.v = self.beta2 * self.v + (1 - self.beta2) * (delta_coef ** 2)
            self.v_intercept = self.beta2 * self.v_intercept + (1 - self.beta2) * (delta_intercept ** 2)
        elif self.optimizer == "yogi":
            # Mise à jour Yogi
            self.v = self.v - (1 - self.beta2) * np.sign(self.v - (delta_coef ** 2)) * (delta_coef ** 2)
            self.v_intercept = self.v_intercept - (1 - self.beta2) * np.sign(self.v_intercept - (delta_intercept ** 2)) * (delta_intercept ** 2)
        elif self.optimizer == "adagrad":
            # Mise à jour Adagrad
            self.v += delta_coef ** 2
            self.v_intercept += delta_intercept ** 2
        
        # Appliquer la mise à jour avec le taux d'apprentissage du serveur
        self.global_model.coef_ = self.global_model.coef_ + self.server_lr * self.m / (np.sqrt(self.v) + self.tau)
        self.global_model.intercept_ = self.global_model.intercept_ + self.server_lr * self.m_intercept / (np.sqrt(self.v_intercept) + self.tau)
        
        # Contrôler l'intercept si nécessaire
        if self.clip_intercept:
            self.global_model.intercept_ = np.clip(self.global_model.intercept_, -5, 5)
        
        return self.global_model

# Fonctions utilitaires
def load_datasets(data_paths):
    """Charge les datasets depuis les chemins spécifiés"""
    datasets = {}
    for name, path in data_paths.items():
        datasets[name] = pd.read_csv(path).dropna()
    return datasets

def prepare_client_data(
    df, 
    n_rounds, 
    features=["Power", "DriverAge", "Density", "Homme", "Diesel"], 
    target="Sinistre", 
    test_size=0.4
):
    """
    Prépare les données client pour l'apprentissage fédéré.
    Retourne :
        - batches : liste de tuples (X_batch, y_batch, exposure_batch) pour chaque round
        - (X_test, y_test, exposure_test) : données de test
        - (X_train, y_train, exposure_train) : données d'entraînement
        - local_model : modèle logistique entraîné sur les données locales (pour évaluation croisée)
    """
    X = df[features].values
    y = df[target].values
    exposure = df["Exposure"].values

    # Split stratifié
    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
        X, y, exposure, test_size=test_size, stratify=y, random_state=42
    )

    # Création de mini-batches stratifiés
    indices_class0 = np.where(y_train == 0)[0]
    indices_class1 = np.where(y_train == 1)[0]

    np.random.seed(42)
    np.random.shuffle(indices_class0)
    np.random.shuffle(indices_class1)

    batch_size_0 = len(indices_class0) // n_rounds
    batch_size_1 = max(1, len(indices_class1) // n_rounds)

    batches = []
    for i in range(n_rounds):
        start0, end0 = i * batch_size_0, min((i + 1) * batch_size_0, len(indices_class0))
        start1, end1 = i * batch_size_1, min((i + 1) * batch_size_1, len(indices_class1))
        batch_indices = np.concatenate([
            indices_class0[start0:end0],
            indices_class1[start1:end1]
        ])
        np.random.shuffle(batch_indices)
        batches.append((
            X_train[batch_indices], 
            y_train[batch_indices], 
            exposure_train[batch_indices]
        ))

    # Modèle local pour la comparaison croisée (important pour l'évaluation FL)
    local_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    local_model.fit(X_train, y_train)

    return (
        batches, 
        (X_test, y_test, exposure_test), 
        (X_train, y_train, exposure_train), 
        local_model
    )

def generate_synthetic_data(n_samples=1000, n_features=5, random_state=42):
    """Génère des données synthétiques pour la simulation"""
    np.random.seed(random_state)
    
    # Caractéristiques
    X = np.random.randn(n_samples, n_features)
    
    # Coefficients
    true_coef = np.random.uniform(-1, 1, n_features)
    
    # Probabilités
    logits = X.dot(true_coef) + np.random.randn(n_samples) * 0.5
    probs = 1 / (1 + np.exp(-logits))
    
    # Labels
    y = (np.random.random(n_samples) < probs).astype(int)
    
    # Exposition (pour simuler une assurance)
    exposure = np.random.uniform(0.1, 1.0, n_samples)
    
    # Créer un DataFrame
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
    df["Sinistre"] = y
    df["Exposure"] = exposure
    
    return df

def plot_results(server, clients, feature_names, placeholder):
    """Génère et affiche les graphiques pour un algorithme"""
    # 1. Évolution des poids
    for i, feature in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Poids globaux
        global_weights = [coef[i] for coef in server.weight_history['coef']]
        ax.plot(range(1, len(global_weights)+1), global_weights, 'r-o', linewidth=2, label='Global')
        
        # Poids locaux
        for client in clients:
            client_weights = [coef[i] for coef in client.weight_history['coef']]
            ax.plot(range(1, len(client_weights)+1), client_weights, '-o', linewidth=2, label=client.client_id)
        
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Valeur du poids')
        ax.set_title(f'Évolution du poids pour {feature}')
        ax.legend()
        ax.grid(True)
        placeholder.pyplot(fig)
        plt.close(fig)
    
    # 2. Évolution de l'intercept
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(server.weight_history['intercept'])+1), 
            server.weight_history['intercept'], 'r-o', linewidth=2, label='Global')
    
    for client in clients:
        ax.plot(range(1, len(client.weight_history['intercept'])+1), 
                client.weight_history['intercept'], '-o', linewidth=2, label=client.client_id)
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Valeur de l\'intercept')
    ax.set_title('Évolution de l\'intercept')
    ax.legend()
    ax.grid(True)
    placeholder.pyplot(fig)
    plt.close(fig)
    
    # 3. Évolution de l'AUC
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(server.auc_history['Global'])+1), 
            server.auc_history['Global'], 'k-o', linewidth=2, label='Global')
    
    for client_id, aucs in server.auc_history.items():
        if client_id != 'Global':
            ax.plot(range(1, len(aucs)+1), aucs, '-o', linewidth=2, label=client_id)
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel('AUC Score')
    ax.set_title('Évolution de l\'AUC')
    ax.legend()
    ax.grid(True)
    placeholder.pyplot(fig)
    plt.close(fig)
    
    # 4. Évolution de l'indice de Youden
    fig, ax = plt.subplots(figsize=(10, 5))
    youden_values_global = [data['youden_index'] for data in server.youden_history['Global']]
    ax.plot(range(1, len(youden_values_global)+1), 
            youden_values_global, 'k-o', linewidth=2, label='Global')
    
    for client_id in server.youden_history:
        if client_id != 'Global':
            youden_values = [data['youden_index'] for data in server.youden_history[client_id]]
            ax.plot(range(1, len(youden_values)+1), youden_values, '-o', linewidth=2, label=client_id)
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Indice de Youden')
    ax.set_title('Évolution de l\'indice de Youden')
    ax.legend()
    ax.grid(True)
    placeholder.pyplot(fig)
    plt.close(fig)
    
    # 5. Évolution de la loss par client
    fig, ax = plt.subplots(figsize=(10, 5))
    for client in clients:
        ax.plot(range(1, len(client.loss_history)+1), client.loss_history, '-o', linewidth=2, label=client.client_id)

    ax.set_xlabel('Rounds')
    ax.set_ylabel('Loss')
    ax.set_title('Évolution de la fonction de perte')
    ax.legend()
    ax.grid(True)
    placeholder.pyplot(fig)
    plt.close(fig)

def compare_algorithms(fedavg_server, fedprox_server, fedopt_server, placeholder):
    """Compare les performances des différents algorithmes"""
    # 1. Comparaison des AUC globaux
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(fedavg_server.auc_history['Global'])+1), 
            fedavg_server.auc_history['Global'], 'r-o', linewidth=2, label='FedAvg')
    ax.plot(range(1, len(fedprox_server.auc_history['Global'])+1), 
            fedprox_server.auc_history['Global'], 'g-s', linewidth=2, label='FedProx')
    ax.plot(range(1, len(fedopt_server.auc_history['Global'])+1), 
            fedopt_server.auc_history['Global'], 'b-^', linewidth=2, label='FedOpt')
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel('AUC Score Global')
    ax.set_title('Comparaison des AUC globaux')
    ax.legend()
    ax.grid(True)
    placeholder.pyplot(fig)
    plt.close(fig)
    
    # 2. Comparaison des indices de Youden globaux
    fig, ax = plt.subplots(figsize=(10, 5))
    youden_values_fedavg = [data['youden_index'] for data in fedavg_server.youden_history['Global']]
    youden_values_fedprox = [data['youden_index'] for data in fedprox_server.youden_history['Global']]
    youden_values_fedopt = [data['youden_index'] for data in fedopt_server.youden_history['Global']]
    
    ax.plot(range(1, len(youden_values_fedavg)+1), youden_values_fedavg, 'r-o', linewidth=2, label='FedAvg')
    ax.plot(range(1, len(youden_values_fedprox)+1), youden_values_fedprox, 'g-s', linewidth=2, label='FedProx')
    ax.plot(range(1, len(youden_values_fedopt)+1), youden_values_fedopt, 'b-^', linewidth=2, label='FedOpt')
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Indice de Youden Global')
    ax.set_title('Comparaison des indices de Youden globaux')
    ax.legend()
    ax.grid(True)
    placeholder.pyplot(fig)
    plt.close(fig)
    
    # 3. Tableau comparatif des performances finales
    final_metrics = {
        'FedAvg': {
            'AUC': fedavg_server.auc_history['Global'][-1],
            'Youden': fedavg_server.youden_history['Global'][-1]['youden_index'],
            'Sensibilité': fedavg_server.youden_history['Global'][-1]['sensitivity'],
            'Spécificité': fedavg_server.youden_history['Global'][-1]['specificity'],
            'Seuil optimal': fedavg_server.youden_history['Global'][-1]['threshold']
        },
        'FedProx': {
            'AUC': fedprox_server.auc_history['Global'][-1],
            'Youden': fedprox_server.youden_history['Global'][-1]['youden_index'],
            'Sensibilité': fedprox_server.youden_history['Global'][-1]['sensitivity'],
            'Spécificité': fedprox_server.youden_history['Global'][-1]['specificity'],
            'Seuil optimal': fedprox_server.youden_history['Global'][-1]['threshold']
        },
        'FedOpt': {
            'AUC': fedopt_server.auc_history['Global'][-1],
            'Youden': fedopt_server.youden_history['Global'][-1]['youden_index'],
            'Sensibilité': fedopt_server.youden_history['Global'][-1]['sensitivity'],
            'Spécificité': fedopt_server.youden_history['Global'][-1]['specificity'],
            'Seuil optimal': fedopt_server.youden_history['Global'][-1]['threshold']
        }
    }
    
    df_metrics = pd.DataFrame(final_metrics).T
    placeholder.table(df_metrics)

def run_federated_learning(data, n_clients=3, n_rounds=10, features=None, 
                          fedopt_params=None, fedprox_mu=0.01):
    """Exécute l'apprentissage fédéré avec les trois algorithmes"""
    if features is None:
        features = data.columns.tolist()
        features.remove('Sinistre')
        features.remove('Exposure')
    
    # Diviser les données entre les clients
    client_data = {}
    n_samples = len(data)
    samples_per_client = n_samples // n_clients
    
    for i in range(n_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < n_clients - 1 else n_samples
        client_data[f'Client_{i+1}'] = data.iloc[start_idx:end_idx].copy()
    
    # Préparer les données pour chaque client
    client_batches = {}
    client_test_data = {}
    client_train_data = {}
    local_models = {}
    
    for client_id, df in client_data.items():
        batches, test_data, train_data, local_model = prepare_client_data(
            df, n_rounds=n_rounds, features=features
        )
        client_batches[client_id] = batches
        client_test_data[client_id] = test_data
        client_train_data[client_id] = train_data
        local_models[client_id] = local_model
    
    # Extraire les données de test pour l'évaluation
    X_test_dict = {client_id: data[0] for client_id, data in client_test_data.items()}
    y_test_dict = {client_id: data[1] for client_id, data in client_test_data.items()}
    exposure_test_dict = {client_id: data[2] for client_id, data in client_test_data.items()}
    
    # Créer les clients FedAvg
    fedavg_clients = [
        FedAvgClient(client_batches[client_id], client_id)
        for client_id in client_batches
    ]
    
    # Créer les clients FedProx
    fedprox_clients = [
        FedProxClient(client_batches[client_id], client_id, mu=fedprox_mu)
        for client_id in client_batches
    ]
    
    # Créer les clients FedOpt
    fedopt_clients = [
        FedOptClient(client_batches[client_id], client_id)
        for client_id in client_batches
    ]
    
    # Créer les serveurs
    fedavg_server = FedAvgServer(fedavg_clients, features)
    fedprox_server = FedProxServer(fedprox_clients, features)
    
    # Paramètres par défaut pour FedOpt
    if fedopt_params is None:
        fedopt_params = {
            'server_lr': 0.1,
            'beta1': 0.9,
            'beta2': 0.99,
            'tau': 1e-3,
            'optimizer': 'adam'
        }
    
    fedopt_server = FedOptServer(
        fedopt_clients, 
        features, 
        server_lr=fedopt_params['server_lr'],
        beta1=fedopt_params['beta1'],
        beta2=fedopt_params['beta2'],
        tau=fedopt_params['tau'],
        optimizer=fedopt_params['optimizer']
    )
    
    return fedavg_server, fedprox_server, fedopt_server, X_test_dict, y_test_dict, exposure_test_dict

# Application Streamlit
def main():
    st.set_page_config(page_title="Simulation d'Apprentissage Fédéré", layout="wide")
    
    st.title("Simulation d'Apprentissage Fédéré")
    st.write("""
    Cette application simule trois algorithmes d'apprentissage fédéré : FedAvg, FedProx et FedOpt.
    Vous pouvez utiliser des données synthétiques ou télécharger vos propres données.
    """)
    
    # Paramètres de simulation
    st.sidebar.header("Paramètres de simulation")
    data_source = st.sidebar.radio("Source des données", ["Données synthétiques", "Télécharger des données"])
    
    n_clients = st.sidebar.slider("Nombre de clients", 2, 10, 3)
    n_rounds = st.sidebar.slider("Nombre de rounds", 5, 30, 10)
    
    # Paramètres spécifiques aux algorithmes
    st.sidebar.header("Paramètres des algorithmes")
    
    # FedProx
    fedprox_mu = st.sidebar.slider("FedProx - Coefficient μ", 0.001, 0.1, 0.01, format="%.3f")
    
    # FedOpt
    st.sidebar.subheader("Paramètres FedOpt")
    optimizer_choice = st.sidebar.selectbox("Optimiseur", ["adam", "adagrad", "yogi"])
    server_lr = st.sidebar.slider("Taux d'apprentissage", 0.01, 1.0, 0.1, format="%.2f")
    beta1 = st.sidebar.slider("Beta1", 0.8, 0.99, 0.9, format="%.2f")
    beta2 = st.sidebar.slider("Beta2", 0.9, 0.999, 0.99, format="%.3f")
    tau = st.sidebar.slider("Tau", 1e-5, 1e-1, 1e-3, format="%.5f")
    
    fedopt_params = {
        'server_lr': server_lr,
        'beta1': beta1,
        'beta2': beta2,
        'tau': tau,
        'optimizer': optimizer_choice
    }
    
    if data_source == "Données synthétiques":
        n_samples = st.sidebar.slider("Nombre d'échantillons", 500, 5000, 1000)
        n_features = st.sidebar.slider("Nombre de caractéristiques", 2, 10, 5)
        data = generate_synthetic_data(n_samples, n_features)
        features = [f"Feature_{i+1}" for i in range(n_features)]
    else:
        uploaded_file = st.sidebar.file_uploader("Télécharger un fichier CSV", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.sidebar.write("Aperçu des données :")
            st.sidebar.dataframe(data.head())
            
            # Sélection des caractéristiques
            all_columns = data.columns.tolist()
            target_col = st.sidebar.selectbox("Colonne cible (sinistre)", all_columns)
            exposure_col = st.sidebar.selectbox("Colonne d'exposition", all_columns)
            
            # Filtrer les colonnes déjà sélectionnées
            remaining_cols = [col for col in all_columns if col not in [target_col, exposure_col]]
            features = st.sidebar.multiselect("Caractéristiques", remaining_cols, remaining_cols[:5])
            
            # Renommer les colonnes pour correspondre à notre modèle
            data = data.rename(columns={target_col: "Sinistre", exposure_col: "Exposure"})
        else:
            st.warning("Veuillez télécharger un fichier CSV ou utiliser des données synthétiques.")
            return
    
    # Lancer la simulation
    if st.button("Lancer la simulation"):
        st.write("Préparation des données et initialisation des modèles...")
        
        # Créer les onglets pour les différents algorithmes
        tab1, tab2, tab3, tab4 = st.tabs(["FedAvg", "FedProx", "FedOpt", "Comparaison"])
        
        # Exécuter l'apprentissage fédéré
        fedavg_server, fedprox_server, fedopt_server, X_test_dict, y_test_dict, exposure_test_dict = run_federated_learning(
            data, n_clients, n_rounds, features, fedopt_params, fedprox_mu
        )
        
        # Entraîner FedAvg
        with tab1:
            st.header("Apprentissage Fédéré avec FedAvg")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            fedavg_server.train(X_test_dict, y_test_dict, exposure_test_dict, progress_bar, status_text)
            
            st.success("Entraînement FedAvg terminé!")
            st.subheader("Résultats FedAvg")
            
            # Afficher les graphiques
            plot_results(fedavg_server, fedavg_server.clients, features, st)
        
        # Entraîner FedProx
        with tab2:
            st.header(f"Apprentissage Fédéré avec FedProx (μ={fedprox_mu})")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            fedprox_server.train(X_test_dict, y_test_dict, exposure_test_dict, progress_bar, status_text)
            
            st.success("Entraînement FedProx terminé!")
            st.subheader("Résultats FedProx")
            
            # Afficher les graphiques
            plot_results(fedprox_server, fedprox_server.clients, features, st)
        
        # Entraîner FedOpt
        with tab3:
            st.header(f"Apprentissage Fédéré avec FedOpt ({optimizer_choice.upper()})")
            st.write(f"Paramètres: lr={server_lr}, beta1={beta1}, beta2={beta2}, tau={tau}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            fedopt_server.train(X_test_dict, y_test_dict, exposure_test_dict, progress_bar, status_text)
            
            st.success("Entraînement FedOpt terminé!")
            st.subheader("Résultats FedOpt")
            
            # Afficher les graphiques
            plot_results(fedopt_server, fedopt_server.clients, features, st)
        
        # Comparer les algorithmes
        with tab4:
            st.header("Comparaison des Algorithmes")
            compare_algorithms(fedavg_server, fedprox_server, fedopt_server, st)

if __name__ == "__main__":
    main()
