import matplotlib.pyplot as plt
import numpy as np

def visualize_model_weights(weights, feature_names, save_path=None):
    """
    Visualise les poids
    
    Args:
        weights: Liste ou array des poids du modèle (avec intercept)
        feature_names: Liste des noms des caractéristiques correspondantes
        save_path: Chemin pour sauvegarder l'image (optionnel)
        
    Returns:
        fig: L'objet figure matplotlib
    """
    weights_values = [float(w) for w in weights]
    features = list(feature_names)
    
    sorted_indices = np.argsort(np.abs(weights_values))[::-1]
    sorted_weights = [weights_values[i] for i in sorted_indices]
    sorted_features = [features[i] for i in sorted_indices]
    
    # couleurs
    colors = ['#e74c3c' if w < 0 else '#2ecc71' for w in sorted_weights]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(sorted_features)), sorted_weights, color=colors, edgecolor='black')
    
    for i, v in enumerate(sorted_weights):
        ax.text(v + 0.01 if v >= 0 else v - 0.1, i, f"{v:.4f}", va='center')
    
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    
    ax.set_title('Importance des caractéristiques dans le modèle')
    ax.set_xlabel('Valeur du coefficient')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Légende explicative
    ax.text(
        0.5, -0.15, 
        "Vert: effet positif sur la probabilité de sinistre\nRouge: effet négatif sur la probabilité de sinistre",
        transform=ax.transAxes, ha='center', va='center',
        bbox=dict(facecolor='lightgray', alpha=0.5, boxstyle='round,pad=0.5')
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def compare_model_weights(weights_list, model_names, feature_names, save_path=None):
    """
    Compare les poids de plusieurs modèles côte à côte.
    
    Args:
        weights_list: Liste de listes contenant les poids des différents modèles
        model_names: Liste des noms des modèles correspondants
        feature_names: Liste des noms des caractéristiques
        save_path: Chemin pour sauvegarder l'image (optionnel)
        
    Returns:
        fig: L'objet figure matplotlib
    """
    import pandas as pd
    
    # Créer un dataframe pour organiser les données
    df = pd.DataFrame()
    
    for i, weights in enumerate(weights_list):
        model_name = model_names[i]
        df[model_name] = weights
    
    # Ajouter les noms des features
    df.index = feature_names
    
    # Créer un graphique avec des barres groupées
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Définir les positions des barres
    x = np.arange(len(feature_names))
    width = 0.8 / len(model_names)  # Largeur des barres
    
    # Tracer les barres pour chaque modèle
    for i, model in enumerate(model_names):
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, df[model], width, label=model)
    
    # Ajouter une ligne horizontale à zéro
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Personnaliser le graphique
    ax.set_title('Comparaison des poids entre modèles')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('Valeur du coefficient')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder si un chemin est spécifié
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig