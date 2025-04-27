
def client_update(client, global_coefs, lr, epochs, batch_size):
    X_train, y_train = client["lr"].X_train, client["lr"].y_train
    local_coefs = global_coefs.copy()

    n_samples = X_train.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch, y_batch = X_train[indices[start:end]], y_train[indices[start:end]]

            preds = X_batch @ np.array([local_coefs[k] for k in local_coefs])
            error = preds - y_batch

            for idx, key in enumerate(local_coefs):
                grad = (X_batch[:, idx] @ error) / len(X_batch)
                local_coefs[key] -= lr * grad

    return local_coefs

def fedopt(clients, T, C, lr_local, lr_server, optimizer='fedadam', beta1=0.9, beta2=0.99, tau=1e-6, epochs=1, batch_size=32, verbose=True):
    total_clients = len(clients)
    global_coefs = clients[0]["coefs"].copy()

    # Initialisations
    m = {key: 0.0 for key in global_coefs}
    u = {key: 0.0 for key in global_coefs}

    for t in range(T):
        if verbose:
            print(f"\n🔄 Round {t+1}/{T}")

        K_prime = max(int(C * total_clients), 1)
        selected_clients = np.random.choice(clients, K_prime, replace=False)

        # Collecter les deltas locaux
        deltas = []
        for client in selected_clients:
            local_coefs = client_update(client, global_coefs, lr_local, epochs, batch_size)
            delta = {key: local_coefs[key] - global_coefs[key] for key in global_coefs}
            deltas.append(delta)

        # Agréger les deltas
        aggregated_delta = {key: np.mean([delta[key] for delta in deltas], axis=0) for key in global_coefs}

        # Mise à jour du moment m
        for key in global_coefs:
            m[key] = beta1 * m[key] + (1 - beta1) * aggregated_delta[key]

        # Mise à jour du second moment u
        for key in global_coefs:
            delta_squared = aggregated_delta[key] ** 2
            if optimizer.lower() == 'fedadagrad':
                u[key] += delta_squared
            elif optimizer.lower() == 'fedyogi':
                u[key] -= (1 - beta2) * np.sign(u[key] - delta_squared) * delta_squared
            elif optimizer.lower() == 'fedadam':
                u[key] = beta2 * u[key] + (1 - beta2) * delta_squared
            else:
                raise ValueError("Optimizer must be 'fedadam', 'fedadagrad', or 'fedyogi'.")

        # Mise à jour des poids globaux
        for key in global_coefs:
            global_coefs[key] -= lr_server * m[key] / (np.sqrt(u[key]) + tau)

    return global_coefs
