import numpy as np

class FederatedServer:
    """
    Serveur central pour l'agrégation des modèles
    """
    
    def __init__(self):
        self.clients = []
        self.global_weights = None
    
    def add_client(self, client):
        """
        Ajoute un client au système
        """
        self.clients.append(client)
    
    def federated_averaging(self, weights_list, sample_sizes):
        """
        Implémente l'algorithme FedAvg pour agréger les poids
        """
        total_samples = sum(sample_sizes)
        weights_array = np.array(weights_list)
        weights = np.zeros_like(weights_array[0])
        
        for i, client_weights in enumerate(weights_array):
            pk = sample_sizes[i] / total_samples
            weights += pk * client_weights
        
        return weights
    
    def train_global_model(self, num_rounds=5):
        """
        Entraîne le modèle global sur plusieurs cycles
        """
        print(f"\nDémarrage de l'apprentissage fédéré avec {len(self.clients)} clients")

        if self.global_weights is None and self.clients:
            self.global_weights = self.clients[0].train_local_model()

        for round_num in range(num_rounds):
            print(f"\n--- Cycle d'apprentissage fédéré {round_num + 1}/{num_rounds} ---")

            if round_num > 0:
                for client in self.clients:
                    try:
                        client.update_weights(self.global_weights)
                    except NotImplementedError:
                        print(f"[Info] Le modèle du client '{client.name}' ne supporte pas la mise à jour des poids.")

            local_weights = []
            sample_sizes = []

            for client in self.clients:
                weights = client.train_local_model()
                local_weights.append(weights)
                sample_sizes.append(len(client.X_train))

            try:
                self.global_weights = self.federated_averaging(local_weights, sample_sizes)
            except Exception as e:
                print(f"[Warning] Échec de l'agrégation fédérée: {str(e)}")
                break

            if self.clients:
                feature_names = ['Intercept'] + self.clients[0].features
                print(f"\nPoids globaux après le cycle {round_num + 1}:")
                for name, weight in zip(feature_names, self.global_weights):
                    print(f"  {name}: {weight:.6f}")

        return self.global_weights
