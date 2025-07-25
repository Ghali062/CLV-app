# 💎 CLV Predictor – Application Streamlit

## 🎯 Objectif

Cette application Streamlit permet de prédire la **Customer Lifetime Value (CLV)** d’un abonné télécom à partir de ses caractéristiques contractuelles et comportementales. Elle aide les équipes marketing à **segmenter** les clients et à identifier ceux à **fort potentiel**.

---

## 🚀 Fonctionnalités

- Interface moderne avec design gradient
- Prédiction de la CLV à partir de 6 variables clés
- Catégorisation automatique des clients : Standard, Fidèle, Premium, VIP
- Calcul de la durée de vie estimée et des revenus annuels
- Recommandations personnalisées en fonction du profil client
- Affichage détaillé des données et analyse du risque de churn

---

## 🧠 Modèle embarqué

- **Modèle utilisé** : Pipeline de Machine Learning (Random Forest)
- **Téléchargement automatique** depuis Google Drive (`download_model.py`) si `clv_model_pipeline.pkl` est absent

---

## 🛠️ Installation

```bash
git clone https://github.com/ton-repo/clv-predictor.git
cd clv-predictor
pip install -r requirements.txt

## 🛠️ Lancement

```bash
streamlit run streamlit_CLV.py

## ✅ Performance du modèle

Le modèle CLV utilisé est un **Random Forest Regressor** entraîné sur des données clients télécom.

### 📊 Résultats de performance :

- **R² Score :**
  - **Training** : 0.958
  - **Test** : 0.856

- **Erreur moyenne absolue (MAE) :**
  - **Training** : 13.95 MAD
  - **Test** : **±30.16 MAD**

- **Erreur quadratique moyenne (RMSE) :**
  - **Training** : 38.91 MAD
  - **Test** : 81.65 MAD

- **Validation croisée (5-fold) – R² :**
  - Scores : 0.76, 0.83, 0.82, 0.84, 0.64
  - **Moyenne** : 0.778
  - **Écart-type** : ±0.15

### ✅ Interprétation :

- Le modèle explique plus de **85 %** de la variation de la CLV sur les données test.
- Les prédictions sont en moyenne proches de la réalité avec une erreur d’environ **30 MAD**.
- La validation croisée montre une **bonne robustesse** du modèle.
