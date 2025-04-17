import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np

def prepare_datasets(*dfs, target="Sinistre", exposure_col="Exposure", test_size=0.25, random_state=42):
    results = {}

    for i, df in enumerate(dfs):
        name = f"dataset_{i+1}"
        df_clean = df.dropna()

        # Séparer X, y et Exposure
        X = df_clean.drop(columns=[target, exposure_col])
        y = df_clean[target]
        exposure = df_clean[exposure_col]

        feature_names = X.columns.tolist()

        # Split
        X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
            X, y, exposure, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled_df = pd.DataFrame(scaler.fit_transform(X_train),
                                         columns=X_train.columns,
                                         index=X_train.index)
        X_test_scaled_df = pd.DataFrame(scaler.transform(X_test),
                                        columns=X_test.columns,
                                        index=X_test.index)

        # Conversion en tensors
        X_train_scaled = torch.tensor(X_train_scaled_df.values, dtype=torch.float32)
        X_test_scaled = torch.tensor(X_test_scaled_df.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        exposure_train = torch.tensor(exposure_train.values, dtype=torch.float32).view(-1, 1)
        exposure_test = torch.tensor(exposure_test.values, dtype=torch.float32).view(-1, 1)

        # Résultat
        results[name] = {
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "exposure_train": exposure_train,
            "exposure_test": exposure_test,
            "scaler": scaler,
            "feature_names": feature_names
        }

    return results


def main():
    df_fr = pd.read_csv('data/french_data.csv')
    df_be = pd.read_csv('data/belgium_data.csv')
    df_eu = pd.read_csv('data/european_data.csv')

    results = prepare_datasets(df_fr, df_be, df_eu)

    for name, data in results.items():
        print(f"\n {name}")
        print("X_train_scaled shape :", data["X_train_scaled"].shape)
        print("X_test_scaled shape  :", data["X_test_scaled"].shape)
        print("y_train shape        :", data["y_train"].shape)
        print("y_test shape         :", data["y_test"].shape)
        print("exposure_train shape :", data["exposure_train"].shape)
        print("exposure_test shape  :", data["exposure_test"].shape)
        print("Features :", data["feature_names"])
        print("-" * 50)

if __name__ == "__main__":
    main()
