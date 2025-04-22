import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import f1_score


def plot_coeff_evolution(clients_history, global_history, coef_name, client_names):
    plt.figure(figsize=(12, 6))

    total_steps = len(clients_history[0])

    for i, client_hist in enumerate(clients_history):
        values = [step[coef_name] for step in client_hist]
        plt.plot(range(1, total_steps + 1), values, label=f"{client_names[i]} (local)")

    rounds = [i * (total_steps // len(global_history)) for i in range(len(global_history))]
    avg_values = [g[coef_name] for g in global_history]
    plt.plot(rounds, avg_values, 'ko--', label="FedAvg (global)", linewidth=2)

    plt.title(f"Évolution du coefficient '{coef_name}'")
    plt.xlabel("Épochs (local_epochs × rounds)")
    plt.ylabel("Valeur du coefficient")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()



def find_best_threshold(y_true, y_scores):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    best_f1 = 0

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh

def preprocess_from_df(df):
    df = df.dropna()
    exposure = df["Exposure"].values
    y = df["Sinistre"].values
    X = df.drop(columns=["Sinistre", "Exposure"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, exposure, X.columns


def evaluate_model_on_dataset(model, df):
    X_scaled, y, exposure, _ = preprocess_from_df(df)
    probs = model.predict_proba(X_scaled)[:, 1] * exposure
    auc = roc_auc_score(y, probs, sample_weight=exposure)
    return auc


def cross_evaluate_heatmap(client_names, df_split, local_models, saved_auc_diagonal, saved_auc_federated):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    results = pd.DataFrame(index=client_names + ["Fédéré"], columns=client_names)

    # Comparaison croisée des modèles locaux
    for i, source_name in enumerate(client_names):
        local_model = local_models[source_name]
        coef_dict = local_model.get_coefficients()

        for j, target_name in enumerate(client_names):
            if source_name == target_name:
                results.loc[source_name, target_name] = round(saved_auc_diagonal[target_name], 4)
            else:
                df = df_split[j].dropna()
                X = df.drop(columns=["Sinistre"])
                y = df["Sinistre"]
                exposure = X["Exposure"].values
                X = X.drop(columns=["Exposure"])
                X_scaled = StandardScaler().fit_transform(X)

                model = LogisticRegression()
                model.classes_ = np.array([0, 1])
                model.coef_ = np.array([[coef_dict[col] for col in X.columns]])
                model.intercept_ = np.array([coef_dict["Intercept"]])

                probs = model.predict_proba(X_scaled)[:, 1] * exposure
                auc = roc_auc_score(y, probs, sample_weight=exposure)
                results.loc[source_name, target_name] = round(auc, 4)

    for j, target_name in enumerate(client_names):
        auc = saved_auc_federated[target_name]
        results.loc["Fédéré", target_name] = round(auc, 4)

    plt.figure(figsize=(10, 6))
    sns.heatmap(results.astype(float), annot=True, fmt=".2f", cmap="coolwarm", vmin=0.5, vmax=1)
    plt.title("🌍 Matrice des AUC croisés (modèles locaux + fédéré)")
    plt.tight_layout()
    plt.savefig("results/heatmap_auc_croisée.png")
    plt.show()

def compare_logistic_models(
    y_true,
    y_scores_local,
    coeff_dict_local,
    y_scores_global,
    coeff_dict_global,
    name,
    auc_local=None,
    auc_global=None,
    ap_local=None,
    ap_global=None,
    save_path=None  
):


    best_thresh_local = find_best_threshold(y_true, y_scores_local)
    best_thresh_global = find_best_threshold(y_true, y_scores_global)

    y_pred_local = (y_scores_local >= best_thresh_local).astype(int)
    y_pred_global = (y_scores_global >= best_thresh_global).astype(int)

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Distribution
    df_all = pd.DataFrame({
        "score": np.concatenate([y_scores_local, y_scores_global]),
        "label": np.concatenate([y_true, y_true]),
        "type": ["Local"] * len(y_scores_local) + ["Global"] * len(y_scores_global)
    })

    for cls in [0, 1]:
        for kind in ["Local", "Global"]:
            subset = df_all[(df_all["label"] == cls) & (df_all["type"] == kind)]
            sns.kdeplot(subset["score"], ax=axs[0, 0],
                        linestyle="--" if kind == "Global" else "-",
                        color="blue" if cls == 0 else "orange",
                        label=f"Classe {cls} — {kind}")

    axs[0, 0].set_title(f"Distribution des probabilités — {name}")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. ROC
    fpr_l, tpr_l, _ = roc_curve(y_true, y_scores_local)
    fpr_g, tpr_g, _ = roc_curve(y_true, y_scores_global)
    axs[0, 1].plot(fpr_l, tpr_l, label=f"Local AUC = {auc_local:.2f}", color="blue")
    axs[0, 1].plot(fpr_g, tpr_g, label=f"Global AUC = {auc_global:.2f}", color="green", linestyle="--")
    axs[0, 1].plot([0, 1], [0, 1], 'k--')
    axs[0, 1].set_title(f"Courbe ROC — {name}")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Precision-Recall
    prec_l, rec_l, _ = precision_recall_curve(y_true, y_scores_local)
    prec_g, rec_g, _ = precision_recall_curve(y_true, y_scores_global)
    axs[1, 0].plot(rec_l, prec_l, label=f"Local AP = {ap_local:.2f}", color="blue")
    axs[1, 0].plot(rec_g, prec_g, label=f"Global AP = {ap_global:.2f}", color="green", linestyle="--")
    axs[1, 0].set_title(f"Courbe Precision-Recall — {name}")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Coefficients
    df_coef = pd.DataFrame({
        "Local": pd.Series(coeff_dict_local),
        "Global": pd.Series(coeff_dict_global)
    })
    df_coef.plot(kind='bar', ax=axs[1, 1], color=["blue", "green"])
    axs[1, 1].set_title(f"Comparaison des coefficients — {name}")
    axs[1, 1].grid(True)

    plt.suptitle(f"Comparaison Local vs Global — {name}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        fig.savefig(save_path)
        print(f"✅ Figure sauvegardée dans {save_path}")
    plt.close(fig)
