# 📊 Dashboard – Comparaison des modèles SegFormer vs EfficientNetB3

Ce dépôt contient l’application **Streamlit interactive** développée dans le cadre du **Projet 9 - Preuve de concept** pour l’entreprise DataSpace.  
Elle permet de comparer visuellement et quantitativement les prédictions des deux modèles suivants :

- **EfficientNetB3** (U-Net) – déployé sur **AWS EC2**
- **SegFormer B5** – déployé sur **Hugging Face Spaces**

---

## 🧪 Fonctionnalités principales

### 🔍 Exploration du dataset Cityscapes (EDA)
- Aperçu des images et masques annotés
- Statistiques : nombre d’images, répartition des splits (train/val), répartition des pixels par classe
- Visualisation dynamique avec Plotly

### 🤖 Comparaison des prédictions
- Upload d’une image d’entrée (.png) et d’un masque réel optionnel
- Appels aux deux **API cloud** :
  - `EfficientNetB3` via une **API FastAPI sur EC2**
  - `SegFormer` via une **API Hugging Face**
- Affichage côte à côte des masques prédits (colorisés)
- Évaluation des métriques (Dice, IoU) pour chaque modèle
- Analyse comparative automatique avec interprétation textuelle

---

## 🖼️ Légende des classes (Cityscapes – 8 classes)

| Classe       | ID | Couleur        |
|--------------|----|----------------|
| void         | 0  | Noir (`#000000`) |
| flat         | 1  | Violet (`#4B0082`) |
| construction | 2  | Gris (`#7C7C7C`) |
| object       | 3  | Orange (`#FF6600`) |
| nature       | 4  | Vert forêt (`#228B22`) |
| sky          | 5  | Cyan foncé (`#008B8B`) |
| human        | 6  | Rouge foncé (`#B22222`) |
| vehicle      | 7  | Bleu (`#0000FF`) |

---

## 🚀 Lancement de l’application

### Prérequis

- Python 3.10+
- Fichier `cityscapes_subset.csv` contenant les chemins `image_path`, `mask_path`, `split`, `px_class0..7`

### Installation

```bash
pip install -r requirements.txt
streamlit run code_dashboard.py
```

---

## 📁 Structure du dépôt

```
├── code_dashboard.py        → Application Streamlit principale
├── fonctions.py             → Fonctions utiles (prétraitement, métriques, affichage)
├── cityscapes_subset.csv    → Données pour EDA (à ajouter)
├── images/                  → Contient les exemples d’images et de masques (EDA)
├── README.md                → Présentation du projet
```

---

## 🔗 Déploiement des modèles

| Modèle        | Déploiement       | API utilisée                                 |
|---------------|-------------------|----------------------------------------------|
| EfficientNetB3 | AWS EC2 (FastAPI) | http://15.236.146.43:8000/predict/           |
| SegFormer B5  | Hugging Face      | https://jimsmith007-p9-api-segformer.hf.space |

---

## 👤 Auteur

Projet développé par **AnthonyJVID** dans le cadre du parcours *AI Engineer* chez OpenClassrooms – Mission 9 : preuve de concept.

---

## 📄 Licence

Ce projet est destiné à un usage pédagogique uniquement. Dataset Cityscapes © utilisé dans un but non commercial.
