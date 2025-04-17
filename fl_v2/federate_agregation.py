def federated_averaging(dataframes: dict, verbose=True):
    avg_coefs = {}
    dataset_weights = []
    total_size = sum(len(df["lr"].X_train) for df in dataframes)

    for df in dataframes:
        size = len(df["lr"].X_train)
        weight = size / total_size
        dataset_weights.append(weight)

    if verbose:
        print("📊 Poids des datasets dans la moyenne fédérée :")
        for i, df in enumerate(dataframes):
            name = df.get("name", f"Dataset {i+1}")
            print(f" - {name}: {dataset_weights[i]*100:.2f}%")

    # Moyenne pondérée des coefficients
    keys = dataframes[0]["coefs"].keys()
    for key in keys:
        weighted_sum = sum(df["coefs"][key] * dataset_weights[i] for i, df in enumerate(dataframes))
        avg_coefs[key] = weighted_sum

    return avg_coefs
