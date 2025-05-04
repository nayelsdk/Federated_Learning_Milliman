import pandas as pd
from sklearn.model_selection import train_test_split

def load_csv_data(client_id, include_exposure=False):
    # Chargement du fichier CSV en fonction du client_id avec chemins spécifiques
    if client_id == 0:
        df = pd.read_csv("/home/onyxia/work/Federated_Learning_Milliman/data/french_data.csv")
    elif client_id == 1:
        df = pd.read_csv("/home/onyxia/work/Federated_Learning_Milliman/data/belgium_data.csv")
    # Vous pouvez rajouter d'autres conditions pour d'autres clients si nécessaire
    else:
        raise ValueError("Invalid client_id")
    
    # Gestion de la colonne 'exposure' en fonction du paramètre include_exposure
    if include_exposure:
        exposure = df["Exposure"].values
        X = df.drop(columns=["Sinistre", "Exposure"]).values
    else:
        exposure = None
        X = df.drop(columns=["Sinistre"]).values

    y = df["Sinistre"].values

    # Division des données en ensembles de formation et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Division de la colonne exposure si nécessaire
    if include_exposure:
        exposure_train, exposure_test = train_test_split(exposure, test_size=0.2, random_state=42)
    else:
        exposure_train, exposure_test = None, None

    # Retourner les ensembles de données avec les colonnes d'exposition si nécessaire
    return X_train, X_test, y_train, y_test, exposure_train, exposure_test
