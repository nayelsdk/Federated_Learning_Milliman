import copy
import torch
import numpy as np
from collections import defaultdict
from client import local_update
from model import LogisticRegressionModel
from client import local_update

def aggregate_models(client_models, client_sizes):
    total_samples = sum(client_sizes)
    new_model = copy.deepcopy(client_models[0])

    with torch.no_grad():
        for name, param in new_model.named_parameters():
            param.data.zero_()
            for model, size in zip(client_models, client_sizes):
                param.data += (size / total_samples) * model.state_dict()[name].data
    return new_model

def train_federated(train_loaders, input_dim, algo='fedavg', T=10, C=1.0, E=1, B=32,
                    lr=0.01, mu=0.0, eta=0.01, beta1=0.9, beta2=0.99, tau=1e-6, device='cpu'):
    K = len(train_loaders)  # nombre de clients
    global_model = LogisticRegressionModel(input_dim).to(device)
    global_weights = {name: param.clone().detach().to(device) for name, param in global_model.named_parameters()}

    client_models = []
    client_sizes = []

    # Loop sur les rounds d'entraînement
    for t in range(T):
        selected = np.random.choice(K, size=int(C * K), replace=False)  # Choisir les clients

        # Créer les modèles locaux
        for k in selected:
            local_model = copy.deepcopy(global_model)  # Assurer que chaque client a une copie du modèle global
            updated_model = local_update(
                local_model, train_loaders[k].dataset, E, B, lr,
                global_weights=global_weights if algo == "fedprox" else None,
                mu=mu if algo == "fedprox" else 0.0,
                device=device
            )

            client_models.append(updated_model)
            client_sizes.append(len(train_loaders[k].dataset))

        # Agrégation des poids
        global_model = aggregate_models(client_models, client_sizes)

    return global_model


