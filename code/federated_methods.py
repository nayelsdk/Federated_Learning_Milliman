def federated_averaging(clients: list, verbose=True):
    avg_coefs = {}
    dataset_weights = []
    total_size = sum(len(client["lr"].X_train) for client in clients)

    for client in clients:
        size = len(client["lr"].X_train)
        weight = size / total_size
        dataset_weights.append(weight)

    if verbose:
        print("\n📊 Poids des datasets dans la moyenne fédérée :")
        for i, client in enumerate(clients):
            print(f" - {client['name']}: {dataset_weights[i]*100:.2f}%")

    # Moyenne pondérée des coefficients
    keys = clients[0]["coefs"].keys()
    for key in keys:
        weighted_sum = sum(client["coefs"][key] * dataset_weights[i] for i, client in enumerate(clients))
        avg_coefs[key] = weighted_sum

    return avg_coefs


def federated_prox(clients, mu=0.01, verbose=True):
    """
    mu : facteur de régularisation proximale
    """
    avg_coefs = {}
    dataset_weights = []
    total_size = sum(len(client["lr"].X_train) for client in clients)

    for client in clients:
        size = len(client["lr"].X_train)
        weight = size / total_size
        dataset_weights.append(weight)

    if verbose:
        print("\n📊 Poids des datasets dans la moyenne (FedProx) :")
        for i, client in enumerate(clients):
            print(f" - {client['name']}: {dataset_weights[i]*100:.2f}%")

    # Mise à jour locale avec pénalisation prox
    for i, client in enumerate(clients):
        global_coefs = client["lr"].get_coefficients()
        local_coefs = client["lr"].train_proximal(global_coefs, mu=mu)
        client["coefs"] = local_coefs

    # Moyenne pondérée des modèles des clients (après mise à jour avec régularisation)
    keys = clients[0]["coefs"].keys()
    for key in keys:
        avg_coefs[key] = sum(client["coefs"][key] * dataset_weights[i] for i, client in enumerate(clients))

    return avg_coefs



def federated_opt(clients, beta1=0.9, beta2=0.99, eta=1e-2, tau=1e-6, variant="adam", verbose=True):
    """
    Algorithme d'agrégation FedOpt, avec support pour les variantes :
    - "adam" : FedAdam
    - "yogi" : FedYogi
    - "adagrad" : FedAdagrad

    Args:
        beta1: hyperparamètre du moment d'ordre 1
        beta2: hyperparamètre du moment d'ordre 2
        eta: learning rate serveur
        tau: facteur de régularisation pour stabiliser la division
        variant: "adam", "yogi", ou "adagrad"
    """
    keys = clients[0]["coefs"].keys()
    n_clients = len(clients)

    # Initialisation de m et u (moments) si pas encore fait
    if not hasattr(federated_opt, "m"):
        federated_opt.m = {k: 0. for k in keys}
        federated_opt.u = {k: 0. for k in keys}
        federated_opt.w_global = clients[0]["coefs"].copy()  # initialisation

    # Calcul des Δk = wk - w_global
    delta_sum = {k: 0. for k in keys}
    for client in clients:
        for k in keys:
            delta = client["coefs"][k] - federated_opt.w_global[k]
            delta_sum[k] += delta / n_clients  # moyenne simple pour participation complète

    # Mise à jour des moments m et u
    for k in keys:
        delta_t = delta_sum[k]
        m_prev = federated_opt.m[k]
        u_prev = federated_opt.u[k]

        # Moment d'ordre 1
        m_t = beta1 * m_prev + (1 - beta1) * delta_t
        federated_opt.m[k] = m_t

        # Moment d'ordre 2 (selon variante)
        if variant == "adam":
            u_t = beta2 * u_prev + (1 - beta2) * (delta_t ** 2)
        elif variant == "yogi":
            sign = np.sign(u_prev - delta_t ** 2)
            u_t = u_prev - (1 - beta2) * sign * (delta_t ** 2)
        elif variant == "adagrad":
            u_t = u_prev + delta_t ** 2
        else:
            raise ValueError(f"⚠️ Variante inconnue : {variant}")

        federated_opt.u[k] = u_t

        # Mise à jour du poids global
        federated_opt.w_global[k] -= eta * m_t / (np.sqrt(u_t) + tau)

    if verbose:
        print(f"🧠 Mise à jour FedOpt - Variante : {variant}")

    return federated_opt.w_global.copy()
