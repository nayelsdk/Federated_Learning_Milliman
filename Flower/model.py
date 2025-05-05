import numpy as np
def fed_avg(weights_list):
    """
    Agrège les poids des clients via la moyenne simple (non pondérée).
    """
    if not weights_list:
        raise ValueError("Aucun poids reçu pour FedAvg.")
    
    coefs = np.array([w[0] for w in weights_list])
    intercepts = np.array([w[1] for w in weights_list])
    
    avg_coef = np.mean(coefs, axis=0)
    avg_intercept = np.mean(intercepts, axis=0)
    
    return avg_coef, avg_intercept
