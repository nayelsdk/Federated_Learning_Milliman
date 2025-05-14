import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTENC

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
    test_size=0.4,
    sampling_strategy=0.43,
    cat_features_indices=[3, 5]  # indices dans X des variables catégorielles (dans l'ordre des features)
):
    """
    Prépare les données client pour l'apprentissage fédéré avec data augmentation (SMOTENC).
    """
    df = df.dropna()
    X = df[features].values
    y = df[target].values
    exposure = df["Exposure"].values

    # Split train/test stratifié
    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
        X, y, exposure, test_size=test_size, stratify=y, random_state=42
    )
    # Data Augmentation uniquement sur le train
    cat_features = [3, 4]  # indices corrects pour Sex et Fuel_type
    smote_nc = SMOTENC(categorical_features=cat_features, sampling_strategy=0.43, random_state=42)
    X_train_aug, y_train_aug = smote_nc.fit_resample(X_train, y_train)

    # Ajuster l'exposition après suréchantillonnage (répéter les valeurs pour les nouvelles instances)
    n_augmented = len(X_train_aug) - len(X_train)
    exposure_aug = np.concatenate([
        exposure_train,
        np.random.choice(exposure_train[y_train == 1], size=n_augmented, replace=True)
    ])

    # Mini-batches stratifiés
    indices_class0 = np.where(y_train_aug == 0)[0]
    indices_class1 = np.where(y_train_aug == 1)[0]

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
            X_train_aug[batch_indices],
            y_train_aug[batch_indices],
            exposure_aug[batch_indices]
        ))

    # Modèle local pour l'évaluation (sur données non augmentées)
    local_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    local_model.fit(X_train_aug, y_train_aug)

    return (
        batches,
        (X_test, y_test, exposure_test),
        (X_train_aug, y_train_aug, exposure_aug),
        local_model
    )
