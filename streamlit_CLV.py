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
            st.stop()
        except FileNotFoundError:
            st.error("❌ Script download_model.py non trouvé!")
            st.stop()
    
    return model_path

# Télécharger le modèle
model_path = download_model_if_needed()

# Configuration de la page
st.set_page_config(
    page_title="Prédiction du CLV Client", 
    page_icon="💎", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look moderne
st.markdown("""
<style>
    /* Styles globaux */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        color: #333;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Form styling */
    .stSelectbox > div > div {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stSelectbox > div > div:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    
    .stRadio > div {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }
    
    /* Success message styling */
    .prediction-result {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        margin: 2rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    /* Animation for cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated-card {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Header moderne
st.markdown("""
<div class="header-container animated-card">
    <div class="header-title">💎 CLV Predictor</div>
    <div class="header-subtitle">Prédiction Intelligente de la Valeur Client Lifetime</div>
</div>
""", unsafe_allow_html=True)

# Charger le pipeline sauvegardé
@st.cache_resource
def load_pipeline():
    with open("clv_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)

try:
    pipeline = load_pipeline()
    st.sidebar.success("✅ Modèle chargé avec succès")
except Exception as e:
    st.sidebar.error(f"❌ Erreur de chargement: {e}")
    st.stop()

# Sidebar avec informations
with st.sidebar:
    st.markdown("### 📊 Informations")
    st.markdown("""
    <div class="info-box">
        <strong>🎯 Objectif</strong><br>
        Prédire la valeur client lifetime (CLV) basée sur les caractéristiques du client et son comportement d'achat.
    </div>
    """, unsafe_allow_html=True)
    

# Fonctions pour les choix de valeurs (à adapter à ton projet si nécessaire)
mois_options = list(range(1, 13))
type_abonnement_options = [ 'Essentiel','Essentiel Plus',
       'ESSENTIEL SOCIAL MEDIAS','Forfait Liberté 3 H + 3 Go + 300 SMS', 'Forfait Liberté 11 Go + 1 H',
              'Forfait Liberté 11 H + 2 Go',
              'Forfait Liberté 13 Go +  4 H',
       '  Forfait Liberté 15 Go + 5 H',
       'Forfait Liberté 20 Go + 1 H',
       'Forfait Liberté 22 Go + 2 H',
       '   Forfait maitrisé 14 Go + 14 H',
       '   Forfait maitrisé 30 Go + 3 H', 'Forfait Maitrisé 45 Go + 5 H',
       '    Forfait maitrisé 30 Go + 12 H',
       '     Forfait Maitrisé 25 Go + 20 H',
       'Forfait maitrisé 22 H + 12 Go', 'Forfait maitrisé 30 Go + 30 H',
       '      Forfait Maitrisé 55 Go + 15 H',
       'Illimité mobile 259DH/mois',
       '       Illimité Mobile 439 DH/mois',
       '        Illimité Mobile 649 DH/mois']

typecarte_options = ["visa", "mastercard", "autre"]
pays_options = ['maroc', 'malte', 'france', "etats unis d'amerique", 'italie',
       'hongrie', 'inde', 'belgique', 'allemagne', 'bresil', 'oman',
       'algerie', 'tunisie', 'canada', 'vietnam', 'espagne', 'colombie',
       'autriche', 'norvege', 'portugal', 'royaume uni', 'irlande',
       'roumanie', 'hollande', 'finlande', 'suisse', 'republique tcheque',
       'republique slovaque', 'egypte', 'croatie', 'australie', 'pologne',
       'cameroun', 'suede', 'mexique', 'emirats arabes unis', 'anguilla',
       'jordanie', 'bosnie herzegovine', 'japon', 'gibraltar', 'russie',
       'namibie', 'slovenie', 'mauritanie', 'luxembourg', 'iran',
       'singapour', 'hong kong', 'angola', 'arabie saoudite', 'senegal',
       'montenegro', 'philippines', 'ile maurice', 'turquie', 'andorre',
       'antilles neerlandaises', 'chypre', 'koweit', 'chine',
       'coree du sud', 'madagascar', 'pakistan', 'bahrein', 'bulgarie',
       'danemark', 'lesotho', 'ukraine', "cote d'ivoire", 'grece',
       'lithuania', 'nouvelle caledonie', 'malaisie', 'syrie', 'gabon',
       'liban', 'uruguay', 'nouvelle zelande', 'chili', 'taiwan',
       'guernsey (uk)', 'nigeria', 'belarusse', 'armenie', 'jersey ( uk)',
       'costa rica', 'serbie', 'perou', 'estonie', 'lettonie',
       'virgin islands (usa)', 'gambie', 'iraq', 'cuba', 'georgie',
       'islande', 'afghanistan', 'soudan', 'iles comores', 'argentine',
       'ethiopie', 'mali', 'afrique du sud', 'acores', 'tchad',
       'palestine', 'benin', 'thailande', 'sri lanka', 'zimbabwe',
       'myanmar', 'bangladesh', 'indonesie', 'guatemala', 'burkina faso',
       'yemen', 'nicaragua', 'repulique dominicaine', 'guinee bissau',
       'saint lucia', 'djibouti']
recency_options = ["Recent","Moderate","Old", "Very Old"]

frequency_score_options = ["Single","Low", "Medium", "High"]

# Interface principale avec colonnes
col1, col2 = st.columns([2, 1])

with col1:
    # Formulaire de saisie moderne
    with st.form(key="clv_form"):
        st.markdown("### 👤 Profil Client")
        
        # Première ligne d'inputs
        col_left, col_right = st.columns(2)
        
        with col_left:
            mois_de_creation = st.selectbox(
                "📅 Mois de création", 
                mois_options,
                help="Mois de création du compte client"
            )
            type_abonnement = st.selectbox(
                "📱 Type d'abonnement", 
                type_abonnement_options,
                help="Type de forfait souscrit par le client"
            )
            pays = st.selectbox(
                "🌍 Pays", 
                pays_options,
                help="Pays de résidence du client"
            )
            avg_order_value = st.number_input(
                "💰 Montant d'abonnement mensuel (MAD)", 
                min_value=59, 
                value=59, 
                step=1,
                help="Montant mensuel de l'abonnement téléphonique"
            )
        
        with col_right:
            recency_category = st.selectbox(
                "⏰ Récence", 
                recency_options,
                help="Ancienneté de la dernière interaction"
            )
            frequency_score = st.selectbox(
                "🔄 Score de fréquence", 
                frequency_score_options,
                help="Fréquence d'achat du client"
            )
            age = st.selectbox(
                "🎂 Âge du client", 
                list(range(18, 80)),
                index=7,  # Default to 25 years old
                help="Âge du client en années"
            )
            is_foreign = st.radio(
                "🌐 Client étranger ?", 
                options=[("Non", 0), ("Oui", 1)], 
                horizontal=True,
                help="Le client réside-t-il à l'étranger ?"
            )[1]
        
        # Bouton de prédiction
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔮 Prédire la Valeur Client (CLV)")

with col2:
    # Panneau d'informations
    st.markdown("### 💡 Guide d'utilisation")
    st.markdown("""
    <div class="metric-card animated-card">
        <h3>🎯 Optimisation CLV</h3>
        <ul>
            <li><strong>Récence:</strong> Plus récent = meilleur CLV</li>
            <li><strong>Fréquence:</strong> Plus élevée = fidélité accrue</li>
            <li><strong>Abonnement:</strong> Revenus mensuels récurrents</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    

# Transformation des entrées, frequency_score : "Single", "Low", "Medium", "High" -> 1, 2, 3,4
transformations = {
     "frequency_score": {"Single": 1, "Low": 2, "Medium": 3, "High": 4},
}

# Transformation des entrées
# Store original values for later display but use transformed values for prediction
original_freq_score = frequency_score

# Apply transformations while keeping originals for display
if frequency_score in transformations["frequency_score"]:
    freq_score_for_model = str(transformations["frequency_score"][frequency_score])
else:
    freq_score_for_model = str(frequency_score)

# Extract just the boolean value from is_foreign selection
is_foreign_value = is_foreign  # This should already be just the integer value

# Préparation des données à prédire
if submit:
    with st.spinner('🔄 Calcul de la prédiction en cours...'):
        # Display dictionary with original values
        display_dict = {
            "mois_de_creation": mois_de_creation,
            "type_d'abonnement": type_abonnement,
            "pays": pays,
            "recency_category": recency_category,
            "age": age,
            "frequency_score": original_freq_score,
            "avg_order_value": avg_order_value,
            "is_foreign": is_foreign_value,
        }
        
        # Prediction dictionary with transformed values
        input_dict = {
            "mois_de_creation": mois_de_creation,
            "type_d'abonnement": type_abonnement,
            "pays": pays,
            "recency_category": recency_category,
            "age": float(age),  # Ensure numeric
            "frequency_score": freq_score_for_model,
            "avg_order_value": float(avg_order_value),  # Ensure numeric
            "is_foreign": int(is_foreign_value),  # Ensure integer
        }
        
        input_df = pd.DataFrame([input_dict])
        
        try:
            predicted_clv = pipeline.predict(input_df)[0]
            
            # Affichage des résultats avec style moderne
            st.markdown("---")
            
            # Résultat principal
            st.markdown(f"""
            <div class="prediction-result animated-card">
                <div style="font-size: 1.2rem; margin-bottom: 1rem;">💎 PRÉDICTION CLV</div>
                <div style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
                    {predicted_clv:,.0f} MAD
                </div>
                <div style="font-size: 1rem; opacity: 0.9;">
                    Valeur totale prédite de l'abonné télécom
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Classification du client (ajustée pour les abonnements télécoms)
            if predicted_clv < 3000:
                category = "🥉 Abonné Standard"
                color = "#95a5a6"
                recommendation = "Proposer des offres d'upgrade et services additionnels"
            elif predicted_clv < 6000:
                category = "🥈 Abonné Fidèle"
                color = "#3498db"
                recommendation = "Maintenir la satisfaction, proposer des services premium"
            elif predicted_clv < 10000:
                category = "🥇 Abonné Premium"
                color = "#e74c3c"
                recommendation = "Service client prioritaire, offres personnalisées"
            else:
                category = "💎 Abonné VIP"
                color = "#9b59b6"
                recommendation = "Gestionnaire dédié, avantages exclusifs"
            
            # Affichage en colonnes pour les métriques
            col_met1, col_met2, col_met3 = st.columns(3)
            
            with col_met1:
                st.markdown(f"""
                <div class="metric-card animated-card">
                    <h3>🏷️ Catégorie Abonné</h3>
                    <div style="color: {color}; font-size: 1.5rem; font-weight: 600;">
                        {category}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_met2:
                # Pour les télécoms, calculer la durée de vie estimée en mois
                duree_vie_mois = predicted_clv / avg_order_value
                st.markdown(f"""
                <div class="metric-card animated-card">
                    <h3>⏱️ Durée de Vie Estimée</h3>
                    <div class="metric-value">{duree_vie_mois:.1f} mois</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_met3:
                # Revenus annuels estimés
                revenus_annuels = avg_order_value * 12
                st.markdown(f"""
                <div class="metric-card animated-card">
                    <h3>� Revenus Annuels</h3>
                    <div class="metric-value">{revenus_annuels:,.0f} MAD</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Recommandations
            st.markdown(f"""
            <div class="info-box animated-card">
                <strong>💡 Recommandation:</strong> {recommendation}
            </div>
            """, unsafe_allow_html=True)
            
            # Données détaillées dans un expander
            with st.expander("📋 Détails de la prédiction"):
                st.markdown("### Données utilisées pour la prédiction:")
                
                # Créer un DataFrame plus lisible
                details_df = pd.DataFrame([
                    {"Paramètre": "Mois de création", "Valeur": f"{mois_de_creation}"},
                    {"Paramètre": "Type d'abonnement", "Valeur": type_abonnement},
                    {"Paramètre": "Pays", "Valeur": pays.title()},
                    {"Paramètre": "Récence", "Valeur": recency_category},
                    {"Paramètre": "Âge", "Valeur": f"{age} ans"},
                    {"Paramètre": "Score de fréquence", "Valeur": original_freq_score},
                    {"Paramètre": "Abonnement mensuel", "Valeur": f"{avg_order_value} MAD/mois"},
                    {"Paramètre": "Client étranger", "Valeur": "Oui" if is_foreign_value else "Non"},
                ])
                
                st.dataframe(details_df, use_container_width=True, hide_index=True)
                
                # Ajout d'informations spécifiques aux télécoms
                st.markdown("### 📊 Analyse Télécom:")
                
                col_tel1, col_tel2 = st.columns(2)
                
                with col_tel1:
                    st.metric("💸 Dépense Mensuelle", f"{avg_order_value} MAD")
                    st.metric("🔢 Durée Prédite", f"{duree_vie_mois:.1f} mois")
                
                with col_tel2:
                    st.metric("📈 Total sur 24 mois", f"{avg_order_value * 24:,.0f} MAD")
                    churn_risk = "Faible" if duree_vie_mois > 18 else "Modéré" if duree_vie_mois > 12 else "Élevé"
                    st.metric("⚠️ Risque de Churn", churn_risk)
                
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction : {e}")
            with st.expander("🔍 Détails de l'erreur"):
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; padding: 2rem;">
    <p>Développé avec ❤️ pour l'optimisation de la valeur des abonnés télécoms</p>
    <p style="font-size: 0.9rem;">© 2025 - Système de Prédiction CLV Télécom</p>
</div>
""", unsafe_allow_html=True)
