import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.impute import SimpleImputer

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
            
            # Stabilisation de l'intercept
            #if abs(self.model.intercept_[0]) > 5:
            #    self.model.intercept_[0] = np.clip(self.model.intercept_[0], -5, 5)
        
        # Calculer la perte finale sur ce batch
        from sklearn.metrics import log_loss
        if np.isnan(X_batch).any():
            X_batch_clean = self.imputer.fit_transform(X_batch)
        else:
            X_batch_clean = X_batch

        y_pred = self.model.predict_proba(X_batch_clean)
        batch_loss = log_loss(y_batch, y_pred)
        self.loss_history.append(batch_loss)
        print(f"   📉 Loss {self.client_id} (Round {round_idx+1}): {batch_loss:.4f}")
        
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
            
            # Stabilisation de l'intercept
            #if abs(self.model.intercept_[0]) > 5:
            #    self.model.intercept_[0] = np.clip(self.model.intercept_[0], -5, 5)
        
        # Calculer la perte finale sur ce batch
        from sklearn.metrics import log_loss
        if np.isnan(X_batch).any():
            X_batch_clean = self.imputer.fit_transform(X_batch)
        else:
            X_batch_clean = X_batch

        y_pred = self.model.predict_proba(X_batch_clean)
        batch_loss = log_loss(y_batch, y_pred)
        self.loss_history.append(batch_loss)
        print(f"   📉 Loss {self.client_id} (Round {round_idx+1}): {batch_loss:.4f}")
        
        # Enregistrer les poids après entraînement
        self.weight_history['coef'].append(np.copy(self.model.coef_[0]))
        self.weight_history['intercept'].append(np.copy(self.model.intercept_[0]))
        
        return self.model
