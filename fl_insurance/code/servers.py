import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
import matplotlib.pyplot as plt

def calculate_youden_index(y_true, y_pred_proba):
    """
    Calcule l'indice de Youden (sensibilité + spécificité - 1) pour différents seuils
    et retourne le seuil optimal avec son indice correspondant.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # l'indice de Youden pour chaque seuil
    youden_indices = tpr - fpr  # équivalent à: sensibilité + spécificité - 1
    
    # Trouver le seuil optimal (celui qui maximise l'indice de Youden)
    optimal_idx = np.argmax(youden_indices)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden_indices[optimal_idx]
    
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

class BaseServer:
    """Initialise le serveur d'agrégation"""
    def __init__(self, clients, feature_names, clip_intercept=False):
        self.clients = clients
        self.n_rounds = clients[0].n_rounds
        self.feature_names = feature_names
        self.clip_intercept = clip_intercept
        
        # Modèle global
        self.global_model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.01,
            class_weight='balanced', # pour gérer le déséquilibre des classes
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
        
        #  intercept
        if self.clip_intercept:
            self.global_model.intercept_ = np.clip(avg_intercept, -5, 5)
        else:
            self.global_model.intercept_ = avg_intercept
        
        return self.global_model
    
    def train(self, X_test_dict, y_test_dict, exposure_test_dict):
        """Exécute l'entraînement fédéré et imprime les résultats par round"""
        imputer = SimpleImputer(strategy='mean')
        
        for round_idx in range(self.n_rounds):
            print(f"\n --> Round {round_idx+1}/{self.n_rounds} <--")
            
            # Entraînement local
            local_models = []
            weights = []
            
            for client in self.clients:
                local_model = client.train_local_model(self.global_model, round_idx)
                local_models.append(local_model)
                weights.append(len(client.batches[round_idx][0]))
            
            # Agrégation et enregistrement des poids
            self.global_model = self.aggregate_models(local_models, weights)
            
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
                
                auc = roc_auc_score(y_test, y_pred)
                self.auc_history[client.client_id].append(auc)
                
                youden_results = calculate_youden_index(y_test, y_pred)
                self.youden_history[client.client_id].append(youden_results)
                
                print(f"   🔹 AUC {client.client_id}: {auc:.4f}, Youden: {youden_results['youden_index']:.4f} (seuil: {youden_results['threshold']:.4f})")
                
                # Pour AUC global et Youden global
                all_y_true.extend(y_test)
                all_y_pred.extend(y_pred)
            
            # AUC global
            global_auc = roc_auc_score(all_y_true, all_y_pred)
            self.auc_history['Global'].append(global_auc)
            
            # Youden global
            global_youden_results = calculate_youden_index(all_y_true, all_y_pred)
            self.youden_history['Global'].append(global_youden_results)
            
            print(f"   🔶 AUC Global: {global_auc:.4f}, Youden: {global_youden_results['youden_index']:.4f} (seuil: {global_youden_results['threshold']:.4f})")
            
            # Afficher les coefficients
            for i, feature in enumerate(self.feature_names):
                print(f"   📈 {feature}: {self.global_model.coef_[0][i]:.4f}")
            print(f"   📉 Intercept: {self.global_model.intercept_[0]:.4f}")
        
        return self.global_model

class FedAvgServer(BaseServer):
    pass  # Hérite directement de BaseServer sans modifications

class FedOptServer(BaseServer):
    """On modifie la méthode d'agrégation pour cette méthode avec Adam, Adagrad et Yogi
    """
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
        
        # Update des moments selon l'optimiseur choisi
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta_coef
        self.m_intercept = self.beta1 * self.m_intercept + (1 - self.beta1) * delta_intercept
        
        if self.optimizer == "adam":
            self.v = self.beta2 * self.v + (1 - self.beta2) * (delta_coef ** 2)
            self.v_intercept = self.beta2 * self.v_intercept + (1 - self.beta2) * (delta_intercept ** 2)
        elif self.optimizer == "yogi":
            self.v = self.v - (1 - self.beta2) * np.sign(self.v - (delta_coef ** 2)) * (delta_coef ** 2)
            self.v_intercept = self.v_intercept - (1 - self.beta2) * np.sign(self.v_intercept - (delta_intercept ** 2)) * (delta_intercept ** 2)
        elif self.optimizer == "adagrad":
            self.v += delta_coef ** 2
            self.v_intercept += delta_intercept ** 2
        
        # Appliquer la mise à jour avec le taux d'apprentissage du serveur tau
        self.global_model.coef_ = self.global_model.coef_ + self.server_lr * self.m / (np.sqrt(self.v) + self.tau)
        self.global_model.intercept_ = self.global_model.intercept_ + self.server_lr * self.m_intercept / (np.sqrt(self.v_intercept) + self.tau)
        
        # Contrôler l'intercept si nécessaire
        if self.clip_intercept:
            self.global_model.intercept_ = np.clip(self.global_model.intercept_, -5, 5)
        
        return self.global_model
