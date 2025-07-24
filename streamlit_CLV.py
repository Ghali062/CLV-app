import streamlit as st
import pandas as pd
import pickle
import subprocess
import os
import sys

# to execute this app : 
# pip install -r requirements.txt
# $ streamlit run streamlit_CLV.py    

# Télécharger le modèle si nécessaire
@st.cache_resource
def download_model_if_needed():
    """Télécharge le modèle depuis Google Drive si le fichier model.pkl n'existe pas"""
    model_path = "clv_model_pipeline.pkl"
    
    if not os.path.exists(model_path):
        st.info("🔄 Téléchargement du modèle en cours...")
        try:
            # Exécuter le script de téléchargement
            result = subprocess.run([sys.executable, "download_model.py"], 
                        capture_output=True, text=True, check=True)
            st.success("✅ Modèle téléchargé avec succès!")
        except subprocess.CalledProcessError as e:
            st.error(f"❌ Erreur lors du téléchargement du modèle: {e}")
            st.error(f"Sortie d'erreur: {e.stderr}")
            return None
    
    return model_path

# Charger le modèle entraîné (avec gestion de versions scikit-learn)
@st.cache_resource
def load_model():
    """Charge le modèle ML avec gestion d'erreur pour les versions scikit-learn"""
    model_path = download_model_if_needed()
    if model_path is None:
        return None
    
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        st.info("💡 Essayez de télécharger le modèle à nouveau ou vérifiez la compatibilité des versions.")
        return None

# Configuration de la page
st.set_page_config(
    page_title="Prédiction CLV - IAM",
    page_icon="📊",
    layout="wide"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .custom-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
        margin-top: 0.5rem;
    }
    
    .clv-standard {
        border-left: 4px solid #28a745;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    
    .clv-fidele {
        border-left: 4px solid #17a2b8;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
    }
    
    .clv-premium {
        border-left: 4px solid #ffc107;
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
    }
    
    .clv-vip {
        border-left: 4px solid #dc3545;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    
    .animated-card {
        transition: transform 0.3s ease;
    }
    
    .animated-card:hover {
        transform: translateY(-5px);
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stNumberInput > div > div {
        border-radius: 10px;
    }
    
    .custom-subheader {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="custom-header">
    <h1>🎯 Prédiction Customer Lifetime Value (CLV)</h1>
    <p>Optimisez la valeur de vos clients avec l'intelligence artificielle</p>
</div>
""", unsafe_allow_html=True)

# Charger le modèle
model = load_model()

if model is not None:
    # Interface utilisateur
    st.markdown('<h2 class="custom-subheader">📋 Informations Client</h2>', unsafe_allow_html=True)
    
    # Créer deux colonnes pour l'interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Données Démographiques")
        age = st.number_input("Age du client", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Sexe", ["Male", "Female"])
        plan_type = st.selectbox("Type de forfait", ["Prepaid", "Postpaid"])
    
    with col2:
        st.markdown("### 💼 Données Comportementales")
        monthly_usage = st.number_input("Usage mensuel (minutes)", min_value=0, value=500)
        contract_length = st.number_input("Durée du contrat (mois)", min_value=1, max_value=60, value=12)
        total_spend = st.number_input("Dépenses totales (MAD)", min_value=0, value=1000)
    
    # Bouton de prédiction
    if st.button("🔮 Prédire la CLV", type="primary", use_container_width=True):
        # Créer un DataFrame avec les données d'entrée
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Plan_Type': [plan_type],
            'Monthly_Usage': [monthly_usage],
            'Contract_Length': [contract_length],
            'Total_Spend': [total_spend]
        })
        
        # Faire la prédiction
        try:
            predicted_clv = model.predict(input_data)[0]
            
            # Déterminer la catégorie de CLV
            if predicted_clv < 1000:
                category = "Standard"
                category_class = "clv-standard"
                emoji = "🟢"
            elif predicted_clv < 3000:
                category = "Fidèle"
                category_class = "clv-fidele"
                emoji = "🔵"
            elif predicted_clv < 5000:
                category = "Premium"
                category_class = "clv-premium"
                emoji = "🟡"
            else:
                category = "VIP"
                category_class = "clv-vip"
                emoji = "🔴"
            
            # Affichage des résultats
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="text-align: center; color: #667eea;">📈 Résultats de la Prédiction</h2>
                <div class="metric-card {category_class} animated-card">
                    <h3>{emoji} Customer Lifetime Value</h3>
                    <div class="metric-value">{predicted_clv:,.0f} MAD</div>
                    <p style="margin-top: 1rem; font-weight: 600;">Catégorie: {category}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Métriques supplémentaires
            st.markdown("### 📊 Métriques Détaillées")
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                # Afficher la valeur annuelle (CLV / durée du contrat * 12)
                clv_annuel = (predicted_clv / contract_length) * 12 if contract_length > 0 else predicted_clv
                st.markdown(f"""
                <div class="metric-card animated-card">
                    <h3>📅 CLV Annuel Estimé</h3>
                    <div class="metric-value">{clv_annuel:.0f} MAD</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_met2:
                # Afficher le CLV mensuel estimé (CLV / durée du contrat)
                clv_mensuel = predicted_clv / contract_length if contract_length > 0 else predicted_clv / 12
                st.markdown(f"""
                <div class="metric-card animated-card">
                    <h3>📅 CLV Mensuel Estimé</h3>
                    <div class="metric-value">{clv_mensuel:.0f} MAD</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_met3:
                # Potentiel d'augmentation basé sur l'usage
                potentiel_croissance = predicted_clv * 1.2  # Estimation 20% d'augmentation
                st.markdown(f"""
                <div class="metric-card animated-card">
                    <h3>💰 Potentiel de Croissance</h3>
                    <div class="metric-value">{potentiel_croissance:,.0f} MAD</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommandations basées sur la catégorie CLV
            st.markdown("### 💡 Recommandations Personnalisées")
            
            if category == "Standard":
                st.info("""
                **Stratégies pour clients Standard:**
                - Proposer des offres d'upselling ciblées
                - Améliorer l'engagement par des communications personnalisées
                - Offrir des bonus de fidélité pour augmenter l'usage
                """)
            elif category == "Fidèle":
                st.success("""
                **Stratégies pour clients Fidèles:**
                - Maintenir la satisfaction par un service premium
                - Proposer des services complémentaires
                - Créer des programmes de parrainage
                """)
            elif category == "Premium":
                st.warning("""
                **Stratégies pour clients Premium:**
                - Offrir un service clientèle prioritaire
                - Proposer des offres exclusives
                - Développer des services sur-mesure
                """)
            else:  # VIP
                st.error("""
                **Stratégies pour clients VIP:**
                - Service clientèle dédié et personnalisé
                - Accès prioritaire aux nouvelles offres
                - Événements exclusifs et avantages premium
                """)
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {e}")
            st.info("Vérifiez que toutes les données sont correctement saisies.")

else:
    st.error("❌ Impossible de charger le modèle. Veuillez vérifier que le fichier 'clv_model_pipeline.pkl' existe.")
    st.info("💡 Assurez-vous que le script 'download_model.py' est présent et fonctionnel.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🏢 Développé pour IAM - Intelligence Artificielle & Machine Learning</p>
    <p>📧 Pour support technique, contactez l'équipe Data Science</p>
</div>
""", unsafe_allow_html=True)