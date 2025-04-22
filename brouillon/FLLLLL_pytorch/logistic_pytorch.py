import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from prepare_datasets import prepare_datasets
from agregate_methods import federated_averaging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

def binary_cross_entropy(yhat, y):
    eps = 1e-8 
    return -torch.mean(y * torch.log(yhat + eps) + (1 - y) * torch.log(1 - yhat + eps))


class FederatedLogisticRegression(nn.Module):
    def __init__(self, epochs, nombre_de_boucles_avant_serveur, *dfs):
        super().__init__()
        self.epochs = epochs
        self.nombre_de_boucles_avant_serveur = nombre_de_boucles_avant_serveur
        self.dfs = dfs  
        self.results = None 
        self.n_clients = None
        self.input_dim = None
        self.w = None
        self.b = None
        self.dw = None
        self.db = None
        self.lr = 0.01

    def prepare_datasets(self):
        self.results = prepare_datasets(*self.dfs) 
        self.n_clients=len(self.results)
        self.input_dim = next(iter(self.results.values()))["X_train_scaled"].shape[1]
        self.w, self.b, self.dw, self.db = self.initialize_weights()

    def initialize_weights(self):
        self.w = torch.rand(self.input_dim, self.n_clients, dtype=torch.float32).to(device)
        self.b = torch.randn(self.n_clients, dtype=torch.float32).to(device)
        self.dw = torch.zeros_like(self.w).to(device)
        self.db = torch.zeros_like(self.b).to(device)
        return self.w, self.b, self.dw, self.db


    
    def forward(self):
        yhat={}
        for i,dataset in enumerate(self.results.keys()):
            X=self.results[dataset]["X_train_scaled"]
            w=self.w.T[i]
            b=self.b[i]
            z=torch.matmul(X, w) + b
            exposure=self.results[dataset]["exposure_train"].squeeze()
            print(f"[{dataset}] z: {z.shape}, exposure: {exposure.shape}")

            yhat[dataset]=sigmoid(z)*exposure
        return yhat

    
    def backward(self, x, yhat, y, client_idx):
        m = x.shape[0]
        self.dw[:, client_idx] = (1/m) * torch.matmul(x.T, (yhat - y)).squeeze()
        self.db[client_idx] = (1/m) * torch.sum(yhat - y)
        # correction pour le biais

    
    def optimize(self, client_idx):
        self.w[:, client_idx] -= self.lr * self.dw[:, client_idx]
        self.b[client_idx] -= self.lr * self.db[client_idx]


    def train_local(self):
        X_train = self.results["X_train_scaled"] # (dim, m)
        y_train = self.results["y_train"]         # (1, m)

        for _ in range(self.nombre_de_boucles_avant_serveur):
            yhat = model.forward(X_train)
            model.backward(X_train, yhat, y_train)
            model.optimize()

    def loss(self, yhat, y, n_observations):
        cost_function=(1/n_observations)*np.sum(-y*np.log(self.sigmoid()))
        
        return 
    def federated_training(self):
        
        len_datasets = [len(self.results[key]["X_train_scaled"]) for key in self.results]

        for i in range (epochs):
            if self.nombre_de_boucles_avant_serveur==0:
                self.train_local()
                
        return len_datasets
        


    
#################################################
# Exemple d'utilisation
#################################################
if __name__ == "__main__":
    df_fr = pd.read_csv('data/french_data.csv')
    df_be = pd.read_csv('data/belgium_data.csv')
    df_eu = pd.read_csv('data/european_data.csv')
    dfs = [df_fr, df_be]

    epochs = 100
    nombre_de_boucles_avant_serveur = 5
    model = FederatedLogisticRegression(epochs, nombre_de_boucles_avant_serveur, *dfs)

    model.prepare_datasets()

    print("Input dim :", model.input_dim)
    print("ddddd",model.results['dataset_1']["X_train_scaled"])

    print("Exemple X_train (dataset_1) :", model.results['dataset_1']["X_train_scaled"])
    print("Poids initiaux :", model.w)
    print("Biais initiaux :", model.b, "\n")

    yhat = model.forward()
    print("Résultat de forward :", yhat)
