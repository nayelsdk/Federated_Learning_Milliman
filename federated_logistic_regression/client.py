import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from model import LogisticRegressionModel

def get_dataloaders_per_client(dataframes, label_col="target", batch_size=32, scale=True):
    """Convertit une liste de DataFrames en DataLoaders PyTorch pour chaque client"""
    loaders = []
    scaler = StandardScaler() if scale else None

    # Vérifie que tous les clients ont les mêmes colonnes explicatives
    base_cols = dataframes[0].drop(columns=[label_col]).columns
    for i, df in enumerate(dataframes[1:], start=1):
        if not all(df.drop(columns=[label_col]).columns == base_cols):
            raise ValueError(f"Les colonnes ne sont pas cohérentes entre les clients (problème entre client 0 et {i})")

    # Fit du scaler globalement (toutes les données combinées)
    if scale:
        all_X = pd.concat([df.drop(columns=[label_col]) for df in dataframes])
        scaler.fit(all_X)

    for df in dataframes:
        X = df.drop(columns=[label_col]).values.astype(np.float32)
        y = df[label_col].values.astype(np.float32)
        if scale:
            X = scaler.transform(X)
        X_tensor = torch.tensor(X)
        y_tensor = torch.tensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)

    return loaders, X.shape[1]  # retourne aussi input_dim

def get_local_models(num_clients, input_dim, device="cpu"):
    """Instancie un modèle local pour chaque client"""
    return [LogisticRegressionModel(input_dim).to(device) for _ in range(num_clients)]

import torch
import torch.nn as nn
from torch.optim import SGD

def local_update(model, dataloader, epochs, batch_size, learning_rate, global_weights=None, mu=0.0, device='cpu'):
    # Si global_weights est fourni, les utiliser pour la régularisation (par exemple pour FedProx)
    if global_weights is not None:
        global_weight_tensor = torch.tensor(global_weights[0], dtype=torch.float32).to(device)
        global_bias_tensor = torch.tensor(global_weights[1], dtype=torch.float32).to(device)
    
    # Entraînement local (logique d'entraînement ici)
    model.train()
    optimizer = SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            # Entraînement avec X_batch, y_batch
            optimizer.zero_grad()
            output = model(X_batch.to(device))
            loss = nn.BCELoss()(output.view(-1), y_batch.to(device).view(-1))

            if global_weights is not None:
                # Appliquer la régularisation FedProx
                weight_loss = mu * torch.sum((model.linear.weight - global_weight_tensor) ** 2)
                bias_loss = mu * torch.sum((model.linear.bias - global_bias_tensor) ** 2)
                loss += weight_loss + bias_loss

            loss.backward()
            optimizer.step()

    return model

