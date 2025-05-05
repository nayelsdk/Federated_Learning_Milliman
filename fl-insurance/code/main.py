import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve

from data import load_datasets, prepare_client_data
from clients import FedAvgClient, FedProxClient
from servers import FedAvgServer, FedOptServer
from evaluation import evaluate_cross_performance, plot_results

def run_algorithm(algorithm, n_rounds, local_epochs, results_dir, **kwargs):
    """Exécute un algorithme d'apprentissage fédéré spécifique et sauvegarde les résultats"""
    print(f"\n🚀 Exécution de {algorithm}...")
    
    # Charger les données
    data_paths = {
        "France": "./data/french_data.csv",
        "Belgique": "./data/belgium_data.csv",
        "Europe": "./data/european_data.csv"
    }
    datasets = load_datasets(data_paths)
    
    feature_names = ["Power", "DriverAge", "Fuel_type", "Density", "Sex"]
    
    # Préparer les mini-batches et obtenir les modèles locaux
    fr_batches, (X_fr_test, y_fr_test, exposure_fr_test), _, fr_local_model = prepare_client_data(datasets["France"], n_rounds, feature_names)
    be_batches, (X_be_test, y_be_test, exposure_be_test), _, be_local_model = prepare_client_data(datasets["Belgique"], n_rounds, feature_names)
    eu_batches, (X_eu_test, y_eu_test, exposure_eu_test), _, eu_local_model = prepare_client_data(datasets["Europe"], n_rounds, feature_names)
    
    # Stocker les modèles locaux
    local_models = {
        "France": fr_local_model,
        "Belgique": be_local_model,
        "Europe": eu_local_model
    }
    
    # Initialiser les clients selon l'algorithme
    if algorithm == "FedAvg":
        clients = [
            FedAvgClient(fr_batches, "France", local_epochs),
            FedAvgClient(be_batches, "Belgique", local_epochs),
            FedAvgClient(eu_batches, "Europe", local_epochs)
        ]
        server = FedAvgServer(clients, feature_names)
    elif algorithm == "FedProx":
        mu = kwargs.get('mu', 0.01)
        clients = [
            FedProxClient(fr_batches, "France", local_epochs, mu),
            FedProxClient(be_batches, "Belgique", local_epochs, mu),
            FedProxClient(eu_batches, "Europe", local_epochs, mu)
        ]
        server = FedAvgServer(clients, feature_names)  # FedProx utilise l'agrégation FedAvg
    elif algorithm == "FedOpt":
        clients = [
            FedAvgClient(fr_batches, "France", local_epochs),
            FedAvgClient(be_batches, "Belgique", local_epochs),
            FedAvgClient(eu_batches, "Europe", local_epochs)
        ]
        server = FedOptServer(
            clients, 
            feature_names,
            server_lr=kwargs.get('server_lr', 0.1),
            beta1=kwargs.get('beta1', 0.9),
            beta2=kwargs.get('beta2', 0.99),
            tau=kwargs.get('tau', 1e-3),
            optimizer=kwargs.get('optimizer', 'adam')
        )
    else:
        raise ValueError(f"Algorithme {algorithm} non reconnu")
    
    # Préparer les dictionnaires de test
    X_test_dict = {
        "France": X_fr_test,
        "Belgique": X_be_test,
        "Europe": X_eu_test
    }
    y_test_dict = {
        "France": y_fr_test,
        "Belgique": y_be_test,
        "Europe": y_eu_test
    }
    exposure_test_dict = {
        "France": exposure_fr_test,
        "Belgique": exposure_be_test,
        "Europe": exposure_eu_test
    }
    
    # Entraîner le modèle
    global_model = server.train(X_test_dict, y_test_dict, exposure_test_dict)
    
    # Créer un dossier pour cet algorithme
    algo_dir = os.path.join(results_dir, algorithm)
    os.makedirs(algo_dir, exist_ok=True)
    
    # Sauvegarder les résultats
    np.save(f"{algo_dir}/weights.npy", {
        'coef': server.weight_history['coef'],
        'intercept': server.weight_history['intercept']
    })
    np.save(f"{algo_dir}/auc.npy", server.auc_history)
    
    # Sauvegarder les indices de Youden dans un fichier texte
    with open(f"{algo_dir}/youden_indices.txt", "w") as f:
        f.write(f"Indices de Youden pour {algorithm}\n")
        f.write("=" * 50 + "\n\n")
        
        # Écrire les indices de Youden pour chaque round et chaque client
        for round_idx in range(n_rounds):
            f.write(f"Round {round_idx+1}/{n_rounds}\n")
            f.write("-" * 30 + "\n")
            
            for client_id in server.youden_history:
                if round_idx < len(server.youden_history[client_id]):
                    youden_data = server.youden_history[client_id][round_idx]
                    f.write(f"{client_id}:\n")
                    f.write(f"  Indice de Youden: {youden_data['youden_index']:.4f}\n")
                    f.write(f"  Seuil optimal: {youden_data['threshold']:.4f}\n")
                    f.write(f"  Sensibilité: {youden_data['sensitivity']:.4f}\n")
                    f.write(f"  Spécificité: {youden_data['specificity']:.4f}\n\n")
            
            f.write("\n")
    
    # Générer et sauvegarder les graphiques
    plot_results(server, clients, feature_names, algo_dir)
    
    # Évaluation croisée des performances
    evaluate_cross_performance(local_models, global_model, X_test_dict, y_test_dict, exposure_test_dict, algo_dir)
    
    return global_model, server, local_models

def run_all_algorithms(n_rounds=10, local_epochs=3):
    """Exécute tous les algorithmes d'apprentissage fédéré et sauvegarde les résultats"""
    # Créer un dossier pour les résultats
    results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Exécuter FedAvg
    fedavg_model, fedavg_server, fedavg_local_models = run_algorithm("FedAvg", n_rounds, local_epochs, results_dir)
    
    # Exécuter FedProx avec différentes valeurs de mu
    fedprox_model, fedprox_server, fedprox_local_models = run_algorithm("FedProx", n_rounds, local_epochs, results_dir, mu=0.01)
    
    # Exécuter FedOpt avec Adam
    fedopt_model, fedopt_server, fedopt_local_models = run_algorithm(
        "FedOpt", 
        n_rounds, 
        local_epochs, 
        results_dir,
        server_lr=0.1,
        optimizer="adam"
    )
    
    # Créer un graphique comparatif des AUC globales
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, n_rounds+1), fedavg_server.auc_history['Global'], 'r-o', linewidth=2, label='FedAvg')
    plt.plot(range(1, n_rounds+1), fedprox_server.auc_history['Global'], 'g-o', linewidth=2, label='FedProx')
    plt.plot(range(1, n_rounds+1), fedopt_server.auc_history['Global'], 'b-o', linewidth=2, label='FedOpt')
    
    plt.xlabel('Rounds')
    plt.ylabel('AUC Global')
    plt.title('Comparaison des AUC globales entre les algorithmes')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/comparison_auc.png")
    plt.close()
    
    # Créer un graphique comparatif des indices de Youden globaux
    plt.figure(figsize=(12, 6))
    youden_values_fedavg = [data['youden_index'] for data in fedavg_server.youden_history['Global']]
    youden_values_fedprox = [data['youden_index'] for data in fedprox_server.youden_history['Global']]
    youden_values_fedopt = [data['youden_index'] for data in fedopt_server.youden_history['Global']]
    
    plt.plot(range(1, n_rounds+1), youden_values_fedavg, 'r-o', linewidth=2, label='FedAvg')
    plt.plot(range(1, n_rounds+1), youden_values_fedprox, 'g-o', linewidth=2, label='FedProx')
    plt.plot(range(1, n_rounds+1), youden_values_fedopt, 'b-o', linewidth=2, label='FedOpt')
    
    plt.xlabel('Rounds')
    plt.ylabel('Indice de Youden Global')
    plt.title('Comparaison des indices de Youden globaux entre les algorithmes')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_dir}/comparison_youden.png")
    plt.close()
    
    # Sauvegarder les paramètres de l'expérience
    with open(f"{results_dir}/parameters.txt", "w") as f:
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nombre de rounds: {n_rounds}\n")
        f.write(f"Époques locales: {local_epochs}\n")
        f.write(f"FedProx mu: 0.01\n")
        f.write(f"FedOpt optimizer: adam\n")
        f.write(f"FedOpt server_lr: 0.1\n")
    
    # Sauvegarder une comparaison des indices de Youden finaux
    with open(f"{results_dir}/final_youden_comparison.txt", "w") as f:
        f.write("Comparaison des indices de Youden finaux\n")
        f.write("=" * 50 + "\n\n")
        
        # Pour chaque client et global
        for client_id in fedavg_server.youden_history:
            f.write(f"{client_id}:\n")
            f.write("-" * 30 + "\n")
            
            # Récupérer les derniers indices de Youden pour chaque algorithme
            if fedavg_server.youden_history[client_id]:
                youden_fedavg = fedavg_server.youden_history[client_id][-1]
                f.write(f"FedAvg:\n")
                f.write(f"  Indice de Youden: {youden_fedavg['youden_index']:.4f}\n")
                f.write(f"  Seuil optimal: {youden_fedavg['threshold']:.4f}\n")
                f.write(f"  Sensibilité: {youden_fedavg['sensitivity']:.4f}\n")
                f.write(f"  Spécificité: {youden_fedavg['specificity']:.4f}\n\n")
            
            if fedprox_server.youden_history[client_id]:
                youden_fedprox = fedprox_server.youden_history[client_id][-1]
                f.write(f"FedProx:\n")
                f.write(f"  Indice de Youden: {youden_fedprox['youden_index']:.4f}\n")
                f.write(f"  Seuil optimal: {youden_fedprox['threshold']:.4f}\n")
                f.write(f"  Sensibilité: {youden_fedprox['sensitivity']:.4f}\n")
                f.write(f"  Spécificité: {youden_fedprox['specificity']:.4f}\n\n")
            
            if fedopt_server.youden_history[client_id]:
                youden_fedopt = fedopt_server.youden_history[client_id][-1]
                f.write(f"FedOpt:\n")
                f.write(f"  Indice de Youden: {youden_fedopt['youden_index']:.4f}\n")
                f.write(f"  Seuil optimal: {youden_fedopt['threshold']:.4f}\n")
                f.write(f"  Sensibilité: {youden_fedopt['sensitivity']:.4f}\n")
                f.write(f"  Spécificité: {youden_fedopt['specificity']:.4f}\n\n")
            
            f.write("\n")
    
    return results_dir

if __name__ == "__main__":
    results_dir = run_all_algorithms(n_rounds=20, local_epochs=20)
    print(f"Résultats disponibles dans: {results_dir}")