# Federated_Learning_Milliman

![Logo Milliman](logo_milliman.png)

<p align="center">Projet de Statistiques Appliquées en partenariat avec <strong>Milliman France</strong>, cabinet de conseil en actuariat, réalisé dans le cadre du cycle ingénieur de <strong>l'ENSAE Paris</strong>.</p>

Ce projet explore l'application du **Federated Learning (Apprentissage Fédéré)** dans le domaine assurantiel, en réponse aux contraintes de confidentialité des données et à l'hétérogénéité entre compagnies d’assurance.

Le projet vise à prédire la survenue d’un sinistre automobile tout en respectant les contraintes de confidentialité imposées par des réglementations comme le RGPD. Trois algorithmes d'apprentissage fédéré ont été testés : **FedAvg**, **FedProx**, et **FedOpt**.

## 🗂️ Données utilisées

- **freMTPL** (France)
- **beMTPL** (Belgique)
- **euMTPL** (Europe/Italie)

Sources : [CASDatasets](https://cas.uqam.ca/pub/web/CASdatasets-manual.pdf)

## 🔧 Installation

```bash
git clone https://github.com/nayelsdk/Federated_Learning_Milliman.git
cd Federated_Learning_Milliman
pip install -r requirements.txt
```

## 📁 Structure du projet

L'organisation du projet est la suivante :

```
FEDERATED_LEARNING_MILLIMAN/
│   └── raw_data/...                # Données brutes récupérées sur CASDatasets
├── clean_data_final/               # Scripts Jupyter et fichiers bruts pour nettoyage
│   ├── beMTPL.ipynb
│   ├── euMTPL.ipynb
│   ├── freMTPL.ipynb
│   ├── code-postaux-belge.csv
│   ├── provinces_italie.csv
│   └── wiki_scraping.csv
├── Stat_desc                       # Scripts Jupyter qui réalise l'ensemble des statistiques descriptives de nos données
│   ├── Stat_desc.ipynb
│   └── test_Stat_desc_image.ipynb
├── fl_insurance/
│   └── code/                       # Code source du projet
│       ├── clients.py              # Modélisation côté client
│       ├── data.py                 # Chargement et traitement des données
│       ├── evaluation.py           # Outils d’évaluation des modèles
│       ├── main.py                 # Script principal d'entraînement
│       ├── servers.py              # Comportement du serveur fédéré selon FedAvg et FedOpt
├── requirements.txt                # Ensemble des installations nécessaires pour que le projet tourne
├── rapport.pdf                     # Rapport final du projet avec tous les résultats
├── slides.pdf                      # Support visuel de la soutenance du 05/06/2025 (work in progress ... 🚧)
├── note_de_synthèse.pdf            # Note de synthèse du projet
```

Il est nécessaire de lancer les fichiers notebook de clean_data_final afin de faire apparaitre le dossier `data` puis le notebook `data_augmentation.ipynb`pour créer le dossier `data_augmentation`.

```
├── data/                           # Données nettoyées par clean_data_final
│   ├── belgium_data.csv
│   ├── french_data.csv
│   ├── european_data.csv
├── data_augmentation/...           # Données issues de data_augmentation.ipynb (comporte le même contenu que le dossier data)
```

## 👑 Remerciements

Nous remercions **François HU**, Head of AI Lab chez Milliman, ainsi que **Fallou NIAKH**, encadrant de la voie Actuariat à l’ENSAE Paris, pour leur accompagnement tout au long de ce projet.

Nous tenions également à remercier notre référente **Mme. Caroline HILLAIRET**, responsable de la voie Actuariat à l'ENSAE Paris, pour ses conseils et son cadrage de projet.

## ⬇️ Auteurs

- [Kevin ABE](https://www.linkedin.com/in/kevin-abe-a57a52253/)
- [BENABDESADOK Nayel](https://www.linkedin.com/in/nayel-benabdesadok)
- [Crespin HOUNKPEVI](https://www.linkedin.com/in/crespin-hounkpevi-074495297/)
- [REN Alexandre](https://www.linkedin.com/in/alexandre-ren-a53a04292)
