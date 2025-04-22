
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