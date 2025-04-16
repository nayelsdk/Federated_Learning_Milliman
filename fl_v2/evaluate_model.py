import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    roc_auc_score
)

def plot_proba_distribution(y_true, y_scores):
    plt.figure(figsize=(8, 5))
    sns.histplot(y_scores, bins=30, kde=True, hue=y_true, palette="Set1", stat="density", common_norm=False)
    plt.title("Distribution des probabilités prédites")
    plt.xlabel("Probabilité prédite (classe 1)")
    plt.ylabel("Densité")
    plt.grid(True)
    plt.show()


def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    sns.lineplot(x=fpr, y=tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.title("Courbe ROC")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 6))
    sns.lineplot(x=recall, y=precision, label=f"AP = {ap:.2f}")
    plt.title("Courbe Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.show()

def show_logistic_metrics(y_true, y_pred, y_scores):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    print("📌 AUC ROC :", round(auc_roc, 4))
    print("📌 Average Precision (PR AUC) :", round(ap, 4))
    print("\n📋 Classification Report :\n", report)
    print("🧮 Confusion Matrix :")
    print(cm)



def plot_logistic_coefficients(coeff_dict):
    coef_df = pd.DataFrame.from_dict(coeff_dict, orient='index', columns=['Coefficient'])
    coef_df = coef_df.drop('Intercept', errors='ignore')
    coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=coef_df.index, y=coef_df['Coefficient'], palette="vlag")
    plt.xticks(rotation=45, ha='right')
    plt.title("Importance des variables (coefficients)")
    plt.ylabel("Valeur du coefficient")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
