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
