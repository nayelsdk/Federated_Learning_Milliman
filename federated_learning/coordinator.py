class FederatedLearning:
    """
    Coordonnateur de l'apprentissage fédéré
    """
    
    def __init__(self, data_dict, features, target, model_class, server=None, **model_params):
        """
        Initialise le système d'apprentissage fédéré
        
        Args:
            data_dict: Dictionnaire {nom_client: dataframe}
            features: Liste des caractéristiques
            target: Nom de la variable cible
            model_class: Classe du modèle à utiliser
            server: Instance de FederatedServer ou None
            **model_params: Paramètres à passer au constructeur du modèle
        """
        from .server import FederatedServer
        
        self.data_dict = data_dict
        self.features = features
        self.target = target
        self.model_class = model_class
        self.model_params = model_params
        self.server = server if server else FederatedServer()
    
    def setup(self):
        """
        Configure le système en créant les clients
        """
        from .client import FederatedClient
        
        for client_name, data in self.data_dict.items():
            try:
                print(f"Configuration du client '{client_name}' avec {len(data)} observations")
                
                # Créer une nouvelle instance du modèle pour chaque client
                model = self.model_class(**self.model_params)
                
                # Créer le client
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
    
    def train(self, num_rounds=5):
        """
        Lance l'entraînement fédéré
        """
        return self.server.train_global_model(num_rounds)