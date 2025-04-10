import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from federated_learning.coordinator import FederatedLearning
from federated_learning.models.logistic_v2 import LogisticModel_Stat
from federated_learning.server import FederatedServer

# Chemins des fichiers de données
file_paths = {
    'Client BE': 'data/european_data.csv',  # Adaptez ces chemins à vos fichiers
    'Client EU': 'data/european_data.csv'   # Adaptez ces chemins à vos fichiers
}

features = ['Power', 'DriverAge', 'Fuel_type', 'Density', 'Sex']
target = 'Sinistre'

# Valeurs de alpha à tester
alphas = [0.0, 0.01, 0.1, 1.0, 10.0]
results = {}

# Test de chaque valeur d'alpha séparément
for alpha in alphas:
    print(f"\n\n========== Test avec alpha = {alpha} ==========")
    
    # Force la création d'un nouveau serveur pour chaque test
    server = FederatedServer()
    
    # Initialisation du système fédéré avec cette valeur d'alpha
    fl = FederatedLearning(
        data_dict={},  # Sera rempli plus tard
        features=features,
        target=target,
        model_class=LogisticModel_Stat,
        server=server,
        alpha=alpha
    )
    
    # Chargement des données à chaque fois pour assurer la fraîcheur
    data_dict = {}
    for client_name, path in file_paths.items():
        df = pd.read_csv(path)
        print(f"Données chargées pour {client_name}: {len(df)} lignes")
        data_dict[client_name] = df
    
    # Mise à jour des données
    fl.data_dict = data_dict
    
    # Configuration avec le nouvel alpha
    fl.setup()
    
    # Entraînement avec un seul round
    global_weights = fl.train(num_rounds=1)
    
    # Récupération et affichage des poids
    results[alpha] = global_weights
    
    feature_names = ['Intercept'] + features
    print(f"\nPoids finaux pour alpha = {alpha}:")
    for name, weight in zip(feature_names, global_weights):
        print(f"  {name}: {weight:.6f}")

# Visualisation de l'évolution des poids avec alpha
plt.figure(figsize=(12, 6))
feature_names = ['Intercept'] + features

for feature_idx, feature_name in enumerate(feature_names):
    weights = [results[alpha][feature_idx] for alpha in alphas]
    plt.plot(alphas, weights, marker='o', label=feature_name)

plt.xlabel('Alpha (force de régularisation)')
plt.ylabel('Valeur du coefficient')
plt.title('Évolution des coefficients en fonction de alpha')
plt.legend()
plt.grid(True)
plt.xscale('log')  # Échelle logarithmique pour alpha
plt.savefig('regularization_test.png')
plt.show()

print("\nTest terminé. Les résultats sont disponibles dans regularization_test.png") 