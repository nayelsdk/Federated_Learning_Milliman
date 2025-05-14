import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_datasets(data_paths):
    """Charge les datasets depuis les chemins spécifiés"""
    datasets = {}
    for name, path in data_paths.items():
        datasets[name] = pd.read_csv(path).dropna()
    return datasets

def prepare_client_data(
    df, 
    n_rounds, 
    features=["Power", "DriverAge", "Density", "Sex", "Fuel_type"], 
    target="Sinistre", 
    test_size=0.4
):
    """
    Prépare les données client pour l'apprentissage fédéré.
    Retourne :
        - batches : liste de tuples (X_batch, y_batch, exposure_batch) pour chaque round
        - (X_test, y_test, exposure_test) : données de test
        - (X_train, y_train, exposure_train) : données d'entraînement
        - local_model : modèle logistique entraîné sur les données locales (pour évaluation croisée)
    """
    X = df[features].values
    y = df[target].values
    exposure = df["Exposure"].values

    # Split stratifié
    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
        X, y, exposure, test_size=test_size, stratify=y, random_state=42
    )


    #  mini-batches stratifiés
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

    # Modèle local pour la comparaison croisée (important pour l'évaluation FL)
    local_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    local_model.fit(X_train, y_train)

    return (
        batches, 
        (X_test, y_test, exposure_test), 
        (X_train, y_train, exposure_train), 
        local_model
    )
