import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report)

def compare_logistic_models(y_true, y_pred_local, y_scores_local, coeff_dict_local,
                            y_pred_global, y_scores_global, coeff_dict_global, name=""):

    fig, axs = plt.subplots(2, 2, figsize=(14, 12)) 

    # --- 1. Distribution des probabilités ---
    df_local = pd.DataFrame({
        "Score": y_scores_local,
        "Vraie classe": y_true,
        "Type": "Local"
    })
    df_global = pd.DataFrame({
        "Score": y_scores_global,
        "Vraie classe": y_true,
        "Type": "Global"
    })
    df_all = pd.concat([df_local, df_global], ignore_index=True)

    styles = {"Local": "-", "Global": "--"}
    palette = {0: "blue", 1: "orange"}

    for classe in [0, 1]:
        for model_type in ["Local", "Global"]:
            subset = df_all[(df_all["Vraie classe"] == classe) & (df_all["Type"] == model_type)]
            label = f"Classe {classe} — {model_type}"
            sns.kdeplot(subset["Score"], label=label,
                        linestyle=styles[model_type], color=palette[classe], ax=axs[0, 0])

    axs[0, 0].set_title(f"Distribution des probabilités — {name}")
    axs[0, 0].set_xlabel("Probabilité prédite (classe 1)")
    axs[0, 0].set_ylabel("Densité")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- 2. Courbe ROC ---
    fpr_local, tpr_local, _ = roc_curve(y_true, y_scores_local)
    fpr_global, tpr_global, _ = roc_curve(y_true, y_scores_global)
    auc_local = auc(fpr_local, tpr_local)
    auc_global = auc(fpr_global, tpr_global)

    axs[0, 1].plot(fpr_local, tpr_local, label=f"Local AUC = {auc_local:.2f}", color="blue")
    axs[0, 1].plot(fpr_global, tpr_global, label=f"Global AUC = {auc_global:.2f}", color="green", linestyle="--")
    axs[0, 1].plot([0, 1], [0, 1], linestyle=':', color='gray')
    axs[0, 1].set_title(f"Courbe ROC — {name}")
    axs[0, 1].set_xlabel("FPR")
    axs[0, 1].set_ylabel("TPR")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- 3. Courbe Precision-Recall ---
    precision_l, recall_l, _ = precision_recall_curve(y_true, y_scores_local)
    precision_g, recall_g, _ = precision_recall_curve(y_true, y_scores_global)
    ap_l = average_precision_score(y_true, y_scores_local)
    ap_g = average_precision_score(y_true, y_scores_global)

    axs[1, 0].plot(recall_l, precision_l, label=f"Local AP = {ap_l:.2f}", color="blue")
    axs[1, 0].plot(recall_g, precision_g, label=f"Global AP = {ap_g:.2f}", color="green", linestyle="--")
    axs[1, 0].set_title(f"Courbe Precision-Recall — {name}")
    axs[1, 0].set_xlabel("Recall")
    axs[1, 0].set_ylabel("Precision")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- 4. Coefficients comparés (inclut intercept et tri stable) ---
    coef_df = pd.DataFrame({
        "Local": coeff_dict_local,
        "Global": coeff_dict_global
    })

    # Réindexe par ordre alphabétique (stable), intercept placé en premier si présent
    intercept_first = "Intercept" in coef_df.index
    coef_df_sorted = coef_df.sort_index()

    if intercept_first:
        intercept_row = coef_df_sorted.loc[["Intercept"]]
        coef_df_sorted = pd.concat([intercept_row, coef_df_sorted.drop("Intercept")])

    coef_df_sorted.plot(kind='bar', ax=axs[1, 1], color=["blue", "green"])
    axs[1, 1].set_title(f"Comparaison des coefficients — {name}")
    axs[1, 1].set_ylabel("Valeur du coefficient")
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].grid(True)

    plt.suptitle(f"Comparaison Local vs Global — {name}", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # --- Rapport texte ---
    print(f"\n📋 [Local] Classification Report — {name}")
    print(classification_report(y_true, y_pred_local))
    print(f"📌 AUC ROC (Local) : {auc_local:.4f}")
    print(f"📌 Average Precision (Local) : {ap_l:.4f}")
    print("🧮 Confusion Matrix (Local):\n", confusion_matrix(y_true, y_pred_local))

    print(f"\n📋 [Global] Classification Report — {name}")
    print(classification_report(y_true, y_pred_global))
    print(f"📌 AUC ROC (Global) : {auc_global:.4f}")
    print(f"📌 Average Precision (Global) : {ap_g:.4f}")
    print("🧮 Confusion Matrix (Global):\n", confusion_matrix(y_true, y_pred_global))
