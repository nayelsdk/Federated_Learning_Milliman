import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common.logger import log
import logging

# Configurer les logs pour voir ce qui se passe
log(logging.INFO, "Démarrage du serveur fédéré avec FedAvg...")

# Définir la stratégie FedAvg (sans eval_fn)
strategy = FedAvg(
    fraction_fit=1.0,        # Tous les clients participent à l'entraînement à chaque round
    min_fit_clients=2,       # Nombre minimal de clients pour entraîner
    min_available_clients=2, # Nombre minimal de clients connectés pour lancer un round
)

# Lancer le serveur Flower
fl.server.start_server(
    server_address="127.0.0.1:8080",  # Adresse locale
    config=fl.server.ServerConfig(num_rounds=10),  # Nombre de rounds
    strategy=strategy,
)
