import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, log_loss
from sklearn.model_selection import train_test_split
import tempfile
import base64
from io import BytesIO

# Importer les classes depuis le code fourni
from clients import BaseClient, FedAvgClient, FedProxClient
from servers import BaseServer, FedAvgServer, FedOptServer

# Configuration de la page
st.set_page_config(
    page_title="Simulation d'Apprentissage Fédéré",
    page_icon="🧠",
    layout="wide"
)

# Titre de l'application
st.title("Simulation d'Apprentissage Fédéré")
st.markdown("Cette application permet de simuler et comparer différents algorithmes d'apprentissage fédéré.")

# Créer des onglets pour les différentes sections
tab1, tab2, tab3 = st.tabs(["Configuration", "Simulation", "Résultats"])

# Fonction pour calculer l'indice de Youden
def calculate_youden_index(y_true, y_pred_proba):
    """
    Calcule l'indice de Youden (sensibilité + spécificité - 1) pour différents seuils
    et retourne le seuil optimal avec son indice correspondant.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculer l'indice de Youden pour chaque seuil
    youden_indices = tpr - fpr  # équivalent à: sensibilité + spécificité - 1
    
    # Trouver le seuil optimal (celui qui maximise l'indice de Youden)
    optimal_idx = np.argmax(youden_indices)
    optimal_threshold = thresholds[optimal_idx]
    optimal_youden = youden_indices[optimal_idx]
    
    # Sensibilité et spécificité au seuil optimal
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

# Fonction pour préparer les données client
def prepare_client_data(
    df, 
    n_rounds, 
    features=["Power", "DriverAge", "Density", "Homme", "Diesel"], 
    target="Sinistre", 
    test_size=0.4
):
    """
    Prépare les données client pour l'apprentissage fédéré.
    """
    X = df[features].values
    y = df[target].values
    exposure = df["Exposure"].values

    # Split stratifié
    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
        X, y, exposure, test_size=test_size, stratify=y, random_state=42
    )

    # Création de mini-batches stratifiés
    indices_class0 = np.where(y_train == 0)[0]
    indices_class1 = np.where(y_train == 1)[0]

    np.random.seed(42)
    np.random.shuffle(indices_class0)
    np.random.shuffle(indices_class1)

    batch_size_0 = len(indices_class0) // n_rounds
    batch_size_1 = max(1, len(indices_class1) // n_rounds)

    batches = []
    for i in range(n_rounds):
        start0, end0 = i * batch_size_0, min((i + 1) * batch_size_0, len(indices_class0))
        start1, end1 = i * batch_size_1, min((i + 1) * batch_size_1, len(indices_class1))
        batch_indices = np.concatenate([
            indices_class0[start0:end0],
            indices_class1[start1:end1]
        ])
        np.random.shuffle(batch_indices)
        batches.append((
            X_train[batch_indices], 
            y_train[batch_indices], 
            exposure_train[batch_indices]
        ))

    # Modèle local pour la comparaison croisée
    local_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    local_model.fit(X_train, y_train)

    return (
        batches, 
        (X_test, y_test, exposure_test), 
        (X_train, y_train, exposure_train), 
        local_model
    )

# Fonction pour générer un lien de téléchargement pour une figure
def get_image_download_link(fig, filename, text):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Onglet de configuration
with tab1:
    st.header("Configuration des données et des paramètres")
    
    # Upload des fichiers CSV
    st.subheader("Chargement des données")
    uploaded_files = st.file_uploader("Téléchargez vos fichiers CSV", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        datasets = {}
        for file in uploaded_files:
            df = pd.read_csv(file)
            datasets[file.name.split('.')[0]] = df
            st.success(f"Fichier {file.name} chargé avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    
        # Paramètres de simulation
        st.subheader("Paramètres de simulation")
        col1, col2 = st.columns(2)
        
        with col1:
            n_rounds = st.slider("Nombre de rounds", min_value=5, max_value=50, value=10)
            local_epochs = st.slider("Époques locales", min_value=1, max_value=10, value=3)
            
        with col2:
            mu_prox = st.slider("Coefficient μ pour FedProx", min_value=0.001, max_value=0.1, value=0.01, format="%.3f")
            server_lr = st.slider("Taux d'apprentissage serveur pour FedOpt", min_value=0.01, max_value=0.5, value=0.1, format="%.2f")
        
        # Sélection des caractéristiques
        if len(datasets) > 0:
            first_dataset = list(datasets.values())[0]
            available_features = [col for col in first_dataset.columns if col not in ["Sinistre", "Exposure"]]
            
            st.subheader("Sélection des caractéristiques")
            selected_features = st.multiselect(
                "Choisissez les caractéristiques à utiliser",
                available_features,
                default=["Power", "DriverAge", "Density", "Homme", "Diesel"] if all(f in available_features for f in ["Power", "DriverAge", "Density", "Homme", "Diesel"]) else available_features[:5]
            )
            
            if not selected_features:
                st.warning("Veuillez sélectionner au moins une caractéristique.")
        
        # Stockage des paramètres dans la session
        if st.button("Enregistrer la configuration"):
            if len(datasets) > 0 and len(selected_features) > 0:
                st.session_state.datasets = datasets
                st.session_state.n_rounds = n_rounds
                st.session_state.local_epochs = local_epochs
                st.session_state.mu_prox = mu_prox
                st.session_state.server_lr = server_lr
                st.session_state.features = selected_features
                st.session_state.config_ready = True
                st.success("Configuration enregistrée avec succès!")
            else:
                st.error("Veuillez charger au moins un dataset et sélectionner des caractéristiques.")

# Onglet de simulation
with tab2:
    st.header("Simulation de l'apprentissage fédéré")
    
    if not hasattr(st.session_state, 'config_ready') or not st.session_state.config_ready:
        st.warning("Veuillez d'abord configurer les paramètres dans l'onglet Configuration.")
    else:
        # Préparation des données
        if st.button("Préparer les données"):
            with st.spinner("Préparation des données en cours..."):
                # Créer un dictionnaire pour stocker les données préparées
                client_data = {}
                local_models = {}
                X_test_dict = {}
                y_test_dict = {}
                exposure_test_dict = {}
                
                for client_id, df in st.session_state.datasets.items():
                    batches, (X_test, y_test, exposure_test), (X_train, y_train, exposure_train), local_model = prepare_client_data(
                        df, 
                        st.session_state.n_rounds, 
                        features=st.session_state.features,
                        test_size=0.4
                    )
                    
                    client_data[client_id] = batches
                    local_models[client_id] = local_model
                    X_test_dict[client_id] = X_test
                    y_test_dict[client_id] = y_test
                    exposure_test_dict[client_id] = exposure_test
                
                st.session_state.client_data = client_data
                st.session_state.local_models = local_models
                st.session_state.X_test_dict = X_test_dict
                st.session_state.y_test_dict = y_test_dict
                st.session_state.exposure_test_dict = exposure_test_dict
                st.session_state.data_ready = True
                
                st.success("Données préparées avec succès!")
        
        # Lancement de la simulation
        if hasattr(st.session_state, 'data_ready') and st.session_state.data_ready:
            if st.button("Lancer la simulation"):
                with st.spinner("Simulation en cours..."):
                    # Créer un dossier temporaire pour les résultats
                    results_dir = tempfile.mkdtemp()
                    os.makedirs(os.path.join(results_dir, "FedAvg"), exist_ok=True)
                    os.makedirs(os.path.join(results_dir, "FedProx"), exist_ok=True)
                    os.makedirs(os.path.join(results_dir, "FedOpt"), exist_ok=True)
                    
                    # Stocker les résultats
                    st.session_state.results = {
                        "FedAvg": {},
                        "FedProx": {},
                        "FedOpt": {}
                    }
                    
                    # Simulation FedAvg
                    st.subheader("Simulation FedAvg")
                    fedavg_clients = []
                    for client_id, batches in st.session_state.client_data.items():
                        fedavg_clients.append(FedAvgClient(batches, client_id, st.session_state.local_epochs))
                    
                    fedavg_server = FedAvgServer(fedavg_clients, st.session_state.features)
                    fedavg_model = fedavg_server.train(
                        st.session_state.X_test_dict,
                        st.session_state.y_test_dict,
                        st.session_state.exposure_test_dict
                    )
                    
                    st.session_state.results["FedAvg"]["server"] = fedavg_server
                    st.session_state.results["FedAvg"]["clients"] = fedavg_clients
                    st.session_state.results["FedAvg"]["model"] = fedavg_model
                    
                    # Simulation FedProx
                    st.subheader("Simulation FedProx")
                    fedprox_clients = []
                    for client_id, batches in st.session_state.client_data.items():
                        fedprox_clients.append(FedProxClient(batches, client_id, st.session_state.local_epochs, st.session_state.mu_prox))
                    
                    fedprox_server = FedAvgServer(fedprox_clients, st.session_state.features)
                    fedprox_model = fedprox_server.train(
                        st.session_state.X_test_dict,
                        st.session_state.y_test_dict,
                        st.session_state.exposure_test_dict
                    )
                    
                    st.session_state.results["FedProx"]["server"] = fedprox_server
                    st.session_state.results["FedProx"]["clients"] = fedprox_clients
                    st.session_state.results["FedProx"]["model"] = fedprox_model
                    
                    # Simulation FedOpt
                    st.subheader("Simulation FedOpt")
                    fedopt_clients = []
                    for client_id, batches in st.session_state.client_data.items():
                        fedopt_clients.append(FedAvgClient(batches, client_id, st.session_state.local_epochs))
                    
                    fedopt_server = FedOptServer(
                        fedopt_clients, 
                        st.session_state.features, 
                        server_lr=st.session_state.server_lr
                    )
                    fedopt_model = fedopt_server.train(
                        st.session_state.X_test_dict,
                        st.session_state.y_test_dict,
                        st.session_state.exposure_test_dict
                    )
                    
                    st.session_state.results["FedOpt"]["server"] = fedopt_server
                    st.session_state.results["FedOpt"]["clients"] = fedopt_clients
                    st.session_state.results["FedOpt"]["model"] = fedopt_model
                    
                    st.session_state.simulation_done = True
                    st.success("Simulation terminée avec succès!")

# Onglet de résultats
with tab3:
    st.header("Résultats et comparaison des algorithmes")
    
    if not hasattr(st.session_state, 'simulation_done') or not st.session_state.simulation_done:
        st.warning("Veuillez d'abord lancer la simulation dans l'onglet Simulation.")
    else:
        # Créer des sous-onglets pour chaque type de résultat
        result_tabs = st.tabs(["Évolution AUC", "Évolution Youden", "Poids", "Comparaison"])
        
        # Onglet Évolution AUC
        with result_tabs[0]:
            st.subheader("Évolution de l'AUC par algorithme")
            
            # Créer un graphique comparatif des AUC globaux
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for algo, color in zip(["FedAvg", "FedProx", "FedOpt"], ["blue", "green", "red"]):
                server = st.session_state.results[algo]["server"]
                ax.plot(
                    range(1, len(server.auc_history["Global"])+1),
                    server.auc_history["Global"],
                    "-o", color=color,
                    linewidth=2,
                    label=f"{algo}"
                )
            
            ax.set_xlabel("Rounds")
            ax.set_ylabel("AUC Score")
            ax.set_title("Comparaison de l'évolution de l'AUC global par algorithme")
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Afficher les AUC par client pour chaque algorithme
            for algo in ["FedAvg", "FedProx", "FedOpt"]:
                st.subheader(f"AUC par client - {algo}")
                
                server = st.session_state.results[algo]["server"]
                fig, ax = plt.subplots(figsize=(12, 8))
                
                ax.plot(
                    range(1, len(server.auc_history["Global"])+1),
                    server.auc_history["Global"],
                    "k-o",
                    linewidth=2,
                    label="Global"
                )
                
                for client_id, aucs in server.auc_history.items():
                    if client_id != "Global":
                        ax.plot(
                            range(1, len(aucs)+1),
                            aucs,
                            "-o",
                            linewidth=2,
                            label=client_id
                        )
                
                ax.set_xlabel("Rounds")
                ax.set_ylabel("AUC Score")
                ax.set_title(f"Évolution de l'AUC - {algo}")
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
        
        # Onglet Évolution Youden
        with result_tabs[1]:
            st.subheader("Évolution de l'indice de Youden par algorithme")
            
            # Créer un graphique comparatif des indices de Youden globaux
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for algo, color in zip(["FedAvg", "FedProx", "FedOpt"], ["blue", "green", "red"]):
                server = st.session_state.results[algo]["server"]
                youden_values = [data["youden_index"] for data in server.youden_history["Global"]]
                
                ax.plot(
                    range(1, len(youden_values)+1),
                    youden_values,
                    "-o",color=color,
                    linewidth=2,
                    label=f"{algo}"
                )
            
            ax.set_xlabel("Rounds")
            ax.set_ylabel("Indice de Youden")
            ax.set_title("Comparaison de l'évolution de l'indice de Youden global par algorithme")
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Afficher les indices de Youden par client pour chaque algorithme
            for algo in ["FedAvg", "FedProx", "FedOpt"]:
                st.subheader(f"Indice de Youden par client - {algo}")
                
                server = st.session_state.results[algo]["server"]
                fig, ax = plt.subplots(figsize=(12, 8))
                
                youden_values_global = [data["youden_index"] for data in server.youden_history["Global"]]
                ax.plot(
                    range(1, len(youden_values_global)+1),
                    youden_values_global,
                    "k-o",
                    linewidth=2,
                    label="Global"
                )
                
                for client_id in server.youden_history:
                    if client_id != "Global":
                        youden_values = [data["youden_index"] for data in server.youden_history[client_id]]
                        ax.plot(
                            range(1, len(youden_values)+1),
                            youden_values,
                            "-o",
                            linewidth=2,
                            label=client_id
                        )
                
                ax.set_xlabel("Rounds")
                ax.set_ylabel("Indice de Youden")
                ax.set_title(f"Évolution de l'indice de Youden - {algo}")
                ax.legend()
                ax.grid(True)
                
                st.pyplot(fig)
        
        # Onglet Poids
        with result_tabs[2]:
            st.subheader("Évolution des poids par algorithme")
            
            # Sélection de la caractéristique
            feature_idx = st.selectbox(
                "Sélectionnez une caractéristique",
                range(len(st.session_state.features)),
                format_func=lambda i: st.session_state.features[i]
            )
            
            # Créer un graphique comparatif des poids pour la caractéristique sélectionnée
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for algo, color in zip(["FedAvg", "FedProx", "FedOpt"], ["blue", "green", "red"]):
                server = st.session_state.results[algo]["server"]
                weights = [coef[feature_idx] for coef in server.weight_history["coef"]]
                
                ax.plot(
                    range(1, len(weights)+1),
                    weights,
                    "-o", color=color,
                    linewidth=2,
                    label=f"{algo}"
                )
            
            ax.set_xlabel("Rounds")
            ax.set_ylabel("Valeur du poids")
            ax.set_title(f"Comparaison de l'évolution du poids pour {st.session_state.features[feature_idx]}")
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
            
            # Évolution de l'intercept
            st.subheader("Évolution de l'intercept par algorithme")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for algo, color in zip(["FedAvg", "FedProx", "FedOpt"], ["blue", "green", "red"]):
                server = st.session_state.results[algo]["server"]
                intercepts = server.weight_history["intercept"]
                
                ax.plot(
                    range(1, len(intercepts)+1),
                    intercepts,
                    "-o", color=color,
                    linewidth=2,
                    label=f"{algo}"
                )
            
            ax.set_xlabel("Rounds")
            ax.set_ylabel("Valeur de l'intercept")
            ax.set_title("Comparaison de l'évolution de l'intercept")
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
        
        # Onglet Comparaison
        with result_tabs[3]:
            st.subheader("Comparaison des performances finales")
            
            # Créer un tableau comparatif
            comparison_data = []
            
            for algo in ["FedAvg", "FedProx", "FedOpt"]:
                server = st.session_state.results[algo]["server"]
                final_auc = server.auc_history["Global"][-1]
                final_youden = server.youden_history["Global"][-1]["youden_index"]
                final_threshold = server.youden_history["Global"][-1]["threshold"]
                
                comparison_data.append({
                    "Algorithme": algo,
                    "AUC final": final_auc,
                    "Indice de Youden final": final_youden,
                    "Seuil optimal": final_threshold
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.table(comparison_df)
            
            # Courbes ROC finales
            st.subheader("Courbes ROC finales")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for algo, color in zip(["FedAvg", "FedProx", "FedOpt"], ["blue", "green", "red"]):
                model = st.session_state.results[algo]["model"]
                
                # Combiner tous les ensembles de test
                all_y_true = []
                all_y_pred = []
                
                for client_id in st.session_state.X_test_dict:
                    X_test = st.session_state.X_test_dict[client_id]
                    y_test = st.session_state.y_test_dict[client_id]
                    exposure_test = st.session_state.exposure_test_dict[client_id]
                    
                    # Imputation si nécessaire
                    if np.isnan(X_test).any():
                        imputer = SimpleImputer(strategy='mean')
                        X_test_clean = imputer.fit_transform(X_test)
                    else:
                        X_test_clean = X_test
                    
                    # Prédiction
                    y_pred = model.predict_proba(X_test_clean)[:, 1] * exposure_test
                    
                    all_y_true.extend(y_test)
                    all_y_pred.extend(y_pred)
                
                # Tracer la courbe ROC
                fpr, tpr, _ = roc_curve(all_y_true, all_y_pred)
                auc = roc_auc_score(all_y_true, all_y_pred)
                
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    lw=2,
                    label=f"{algo} (AUC = {auc:.4f})"
                )
            
            ax.plot([0, 1], [0, 1], "k--", lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Taux de faux positifs")
            ax.set_ylabel("Taux de vrais positifs")
            ax.set_title("Comparaison des courbes ROC finales")
            ax.legend(loc="lower right")
            ax.grid(True)
            
            st.pyplot(fig)

# Ajouter du CSS pour améliorer l'apparence
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #0D47A1;
    }
    h3 {
        color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)
