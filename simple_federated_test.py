import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from federated_learning.coordinator import FederatedLearning
from federated_learning.models.logistic_simple import LogisticModelSimple

# Définir les chemins vers vos fichiers de données
file_paths = {
    'Client BE': 'data/european_data.csv',
    'Client EU': 'data/european_data.csv'
}

# Variables à utiliser pour le modèle
features = ['Power', 'DriverAge', 'Fuel_type', 'Density', 'Sex']
target = 'Sinistre'

# Charger les données
data_dict = {}
for client_name, path in file_paths.items():
    df = pd.read_csv(path)
    print(f"Données chargées pour {client_name}: {len(df)} lignes")
    data_dict[client_name] = df

# Initialiser le système d'apprentissage fédéré
print("\nInitialisation du système d'apprentissage fédéré...")
fl = FederatedLearning(
    data_dict=data_dict,
    features=features,
    target=target,
    model_class=LogisticModelSimple
)

# Configuration
print("\nConfiguration des clients...")
fl.setup()

# Entraînement fédéré
print("\nDémarrage de l'entraînement fédéré...")
global_weights = fl.train(num_rounds=5)

# Affichage des résultats
feature_names = ['Intercept'] + features
print("\nPoids finaux du modèle global:")
for name, weight in zip(feature_names, global_weights):
    print(f"  {name}: {weight:.6f}")

# Visualisations
print("\nGénération des visualisations...")
fl.compare_local_vs_global_roc()
fl.compare_local_vs_global_pr()
fl.compare_cross_client_performance()

print("\nTest terminé avec succès!") 