import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_model(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy().flatten()
            y_pred.extend(preds)
            y_true.extend(y_batch.numpy())

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return auc, ap

def compare_global_local(global_model, local_models, dataloaders, device="cpu"):
    results = []
    for i, loader in enumerate(dataloaders):
        auc_global, ap_global = evaluate_model(global_model, loader, device)
        auc_local, ap_local = evaluate_model(local_models[i], loader, device)
        results.append({
            "dataset": i,
            "AUC_global": auc_global,
            "AUC_local": auc_local,
            "AP_global": ap_global,
            "AP_local": ap_local
        })
    return results

def plot_auc_heatmap(models, dataloaders, title="AUC Cross Heatmap", device="cpu"):
    K = len(dataloaders)
    heatmap = np.zeros((K, K))  # lignes : modèle, colonnes : dataset

    for i in range(K):
        for j in range(K):
            auc, _ = evaluate_model(models[i], dataloaders[j], device)
            heatmap[i, j] = auc

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="viridis", xticklabels=[f'D{j}' for j in range(K)],
                yticklabels=[f'M{i}' for i in range(K)])
    plt.title(title)
    plt.xlabel("Dataset utilisé pour le test")
    plt.ylabel("Modèle entraîné sur")
    plt.tight_layout()
    plt.show()
