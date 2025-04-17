import torch

def federated_averaging(w: torch.Tensor, b: torch.Tensor, len_datasets: list):
    """
    Agrège les poids et biais via moyenne fédérée,
    puis les copie dans tous les clients pour synchronisation.

    Arguments :
        w (torch.Tensor): Poids locaux, taille (d, n_clients)
        b (torch.Tensor): Biais locaux, taille (n_clients,)
        len_datasets (list): Taille des jeux de données pour chaque client

    Retourne :
        tuple:
            - w_sync (torch.Tensor): Poids synchronisés, taille (d, n_clients)
            - b_sync (torch.Tensor): Biais synchronisés, taille (n_clients,)
    """
    total_samples = sum(len_datasets)
    n_clients = len(len_datasets)

    # Agrégation
    w_agg = torch.zeros(w.shape[0], dtype=torch.float32, device=w.device)
    b_agg = torch.tensor(0.0, dtype=torch.float32, device=b.device)

    for i in range(n_clients):
        pk = len_datasets[i] / total_samples
        w_agg += pk * w[:, i]
        b_agg += pk * b[i]

    # Réplication pour chaque client
    w_sync = w_agg.unsqueeze(1).repeat(1, n_clients)  # (d,) → (d, n_clients)
    b_sync = b_agg.repeat(n_clients)                  # scalaire → (n_clients,)

    return w_sync, b_sync


if __name__ == "__main__":
    w = torch.randn(4, 3)  # 4 features, 3 clients
    b = torch.randn(3)     # biais de chaque client
    lens = [250, 0, 250]

    w_agg, b_agg = federated_averaging(w, b, lens)
    print(w)
    print(b)
    print("w_agg:", w_agg)  # (4,)
    print("b_agg:", b_agg)  # scalaire
