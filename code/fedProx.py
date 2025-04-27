
def client_update(client, global_coefs, mu, epochs, batch_size, lr):
    X_train, y_train = client["lr"].X_train, client["lr"].y_train
    local_coefs = client["coefs"].copy()

    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch, y_batch = X_train[indices[start:end]], y_train[indices[start:end]]

            # Prédictions
            preds = X_batch @ np.array([local_coefs[k] for k in local_coefs])
            error = preds - y_batch

            # Mise à jour des poids
            for idx, key in enumerate(local_coefs):
                grad = (X_batch[:, idx] @ error) / len(X_batch)
                prox = mu * (local_coefs[key] - global_coefs[key])
                local_coefs[key] -= lr * (grad + prox)

    return local_coefs

def fedprox(clients, T, C, epochs, batch_size, lr, mu, verbose=True):
    total_size = sum(len(c["lr"].X_train) for c in clients)
    global_coefs = clients[0]["coefs"].copy()

    for t in range(T):
        if verbose:
            print(f"\n🔄 Round {t+1}/{T}")

        # Sélection aléatoire de C*K clients
        K = len(clients)
        K_prime = max(int(C * K), 1)
        selected_clients = np.random.choice(clients, K_prime, replace=False)

        dataset_weights = [len(c["lr"].X_train) / sum(len(sc["lr"].X_train) for sc in selected_clients) for c in selected_clients]

        # Mise à jour locale des clients sélectionnés
        local_updates = []
        for client in selected_clients:
            updated_coefs = client_update(client, global_coefs, mu, epochs, batch_size, lr)
            local_updates.append(updated_coefs)

        # Agrégation des poids
        new_global_coefs = {}
        for key in global_coefs:
            new_global_coefs[key] = sum(local_updates[i][key] * dataset_weights[i] for i in range(K_prime))

        global_coefs = new_global_coefs.copy()

    return global_coefs

