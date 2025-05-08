import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
import os

def evaluate_cross_performance(local_models, global_model, X_test_dict, y_test_dict, exposure_test_dict, results_dir):
    """Évalue les performances croisées des modèles locaux vs. modèle global"""
    
    # Créer un dossier pour les résultats de l'évaluation croisée
    cross_eval_dir = os.path.join(results_dir, "cross_evaluation")
    os.makedirs(cross_eval_dir, exist_ok=True)
    
    # Pour chaque paire (modèle source, données cible)
    results = {}
    
    for source_id, source_model in local_models.items():
        results[source_id] = {}
        
        # Évaluer sur chaque client cible
        for target_id, X_test in X_test_dict.items():
            y_test = y_test_dict[target_id]
            exposure_test = exposure_test_dict[target_id]
            
            # Prédiction avec le modèle local source
            y_pred_local = source_model.predict_proba(X_test)[:, 1] * exposure_test
            local_auc = roc_auc_score(y_test, y_pred_local)
            
            # Prédiction avec le modèle global fédéré
            y_pred_global = global_model.predict_proba(X_test)[:, 1] * exposure_test
            global_auc = roc_auc_score(y_test, y_pred_global)
            
            results[source_id][target_id] = {
                'local_auc': local_auc,
                'global_auc': global_auc
            }
            
            # Générer la courbe ROC comparative
            plt.figure(figsize=(10, 8))
            
            # Courbe ROC du modèle local
            fpr_local, tpr_local, _ = roc_curve(y_test, y_pred_local)
            plt.plot(fpr_local, tpr_local, 'r-', lw=2, 
                    label=f'Modèle {source_id} (AUC = {local_auc:.4f})')
            
            # Courbe ROC du modèle global
            fpr_global, tpr_global, _ = roc_curve(y_test, y_pred_global)
            plt.plot(fpr_global, tpr_global, 'b-', lw=2, 
                    label=f'Modèle Fédéré (AUC = {global_auc:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taux de faux positifs')
            plt.ylabel('Taux de vrais positifs')
            plt.title(f'Comparaison: Modèle {source_id} vs Fédéré sur données {target_id}')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f"{cross_eval_dir}/{source_id}_on_{target_id}.png")
            plt.close()
    
    # Créer un tableau récapitulatif
    with open(f"{cross_eval_dir}/cross_evaluation_summary.txt", "w") as f:
        f.write("Résumé de l'évaluation croisée\n")
        f.write("============================\n\n")
        
        for source_id in results:
            f.write(f"Modèle source: {source_id}\n")
            f.write("-" * 50 + "\n")
            
            for target_id, metrics in results[source_id].items():
                f.write(f"  Sur données {target_id}:\n")
                f.write(f"    AUC modèle local: {metrics['local_auc']:.4f}\n")
                f.write(f"    AUC modèle fédéré: {metrics['global_auc']:.4f}\n")
                f.write(f"    Amélioration: {(metrics['global_auc'] - metrics['local_auc']) * 100:.2f}%\n\n")
    
    return results

def plot_results(server, clients, feature_names, algo_dir):
    """Génère et sauvegarde les graphiques pour un algorithme"""
    # 1. Évolution des poids
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(12, 6))
        
        # Poids globaux
        global_weights = [coef[i] for coef in server.weight_history['coef']]
        plt.plot(range(1, len(global_weights)+1), global_weights, 'r-o', linewidth=2, label='Global')
        
        # Poids locaux
        for client in clients:
            client_weights = [coef[i] for coef in client.weight_history['coef']]
            plt.plot(range(1, len(client_weights)+1), client_weights, '-o', linewidth=2, label=client.client_id)
        
        plt.xlabel('Rounds')
        plt.ylabel('Valeur du poids')
        plt.title(f'Évolution du poids pour {feature}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{algo_dir}/weight_{feature}.png")
        plt.close()
    
    # 2. Évolution de l'intercept
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(server.weight_history['intercept'])+1), 
            server.weight_history['intercept'], 'r-o', linewidth=2, label='Global')
    
    for client in clients:
        plt.plot(range(1, len(client.weight_history['intercept'])+1), 
                client.weight_history['intercept'], '-o', linewidth=2, label=client.client_id)
    
    plt.xlabel('Rounds')
    plt.ylabel('Valeur de l\'intercept')
    plt.title('Évolution de l\'intercept')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{algo_dir}/intercept.png")
    plt.close()
    
    # 3. Évolution de l'AUC
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(server.auc_history['Global'])+1), 
            server.auc_history['Global'], 'k-o', linewidth=2, label='Global')
    
    for client_id, aucs in server.auc_history.items():
        if client_id != 'Global':
            plt.plot(range(1, len(aucs)+1), aucs, '-o', linewidth=2, label=client_id)
    
    plt.xlabel('Rounds')
    plt.ylabel('AUC Score')
    plt.title('Évolution de l\'AUC')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{algo_dir}/auc.png")
    plt.close()
    
    # 4. Évolution de l'indice de Youden
    plt.figure(figsize=(12, 6))
    youden_values_global = [data['youden_index'] for data in server.youden_history['Global']]
    plt.plot(range(1, len(youden_values_global)+1), 
            youden_values_global, 'k-o', linewidth=2, label='Global')
    
    for client_id in server.youden_history:
        if client_id != 'Global':
            youden_values = [data['youden_index'] for data in server.youden_history[client_id]]
            plt.plot(range(1, len(youden_values)+1), youden_values, '-o', linewidth=2, label=client_id)
    
    plt.xlabel('Rounds')
    plt.ylabel('Indice de Youden')
    plt.title('Évolution de l\'indice de Youden')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{algo_dir}/youden_index.png")
    plt.close()
    
    # 5. Évolution de la loss par client
    plt.figure(figsize=(12, 6))
    for client in clients:
        plt.plot(range(1, len(client.loss_history)+1), client.loss_history, '-o', linewidth=2, label=client.client_id)

    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Évolution de la fonction de perte')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{algo_dir}/client_loss.png")
    plt.close()
