import streamlit as st
import requests
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données des clients (assurez-vous que client_data est chargé correctement)
client_data = pd.read_csv('data/source/client_data.csv')

# URL de l'API Flask déployée
API_BASE_URL = "https://safe-journey-16014-5c3485a368b4.herokuapp.com"

# Titre de l'application Streamlit
st.title("Prédiction de scoring client")

# ------------------------------------------------------------------------------------

# Fonction pour tirer un client ID aléatoire
def generate_random_client_id():
    return random.choice(client_data['num__SK_ID_CURR'])

# Vérifier si le client ID est déjà dans le session state
if 'random_client_id' not in st.session_state:
    st.session_state.random_client_id = generate_random_client_id()

# Disposition avec deux colonnes
col1, col2 = st.columns([5, 10])  # Ajustez les proportions si nécessaire

# ------------------------------------------------------------------------------------

# Widgets dans la colonne de gauche
with col1:
    st.header("Choix du client")  # Header pour la colonne gauche

    random_client_display = st.empty()
    random_client_display.write(f"Un client ID a été tiré au sort : {st.session_state.random_client_id}")

    if st.button("Tirer un nouveau client ID"):
        st.session_state.random_client_id = generate_random_client_id()
        random_client_display.write(f"Un nouveau client ID a été tiré au sort : {st.session_state.random_client_id}")

    client_id = st.number_input("Entrez l'ID du client ou utilisez celui sélectionné", 
                                min_value=1, 
                                value=int(st.session_state.random_client_id))  # Pré-remplir avec le client ID tiré au sort

    if st.button("Prédire"):
        if client_id:
            response = requests.get(f'https://safe-journey-16014-5c3485a368b4.herokuapp.com/predict?id={client_id}')
            if response.status_code == 200:
                data = response.json()
                st.write(f"Client ID: {data['client_id']}")
                st.write(f"Prédiction: {data['prediction']}")
                st.write(f"Probabilité de la prédiction: {data['prediction_proba']:.4f}")

                emoji = "😊" if data['prediction'] == 1 else "😟"
                if data['prediction'] == 1:
                    st.write(f"Prêt accordé: {emoji}")
                else:
                    st.write(f"Prêt refusé: {emoji}")
            else:
                st.write("Erreur: ", response.json()['error'])

# ------------------------------------------------------------------------------------

# Informations du client dans la colonne de droite via l'API
with col2:
    st.header("Informations du client")  # Header pour la colonne droite

    if client_id:
        response = requests.get(f"{API_BASE_URL}/get_client_info", params={'id': client_id})
        
        if response.status_code == 200:
            client_info = response.json().get("client_info", {})
            
            if client_info:
                # Transformer en DataFrame pour un affichage tabulaire
                client_info_df = pd.DataFrame(client_info.items(), columns=["Nom de la feature", "Valeur"])
                st.dataframe(client_info_df, width=1500, height=500)

            else:
                st.warning("Aucune information trouvée pour cet ID client.")
        else:
            st.error(f"Erreur API : {response.status_code} - {response.json().get('error', 'Erreur inconnue')}")

# ------------------------------------------------------------------------------------

# Visualisation des features
st.header("Visualisation des features")

client_info = client_data[client_data['num__SK_ID_CURR'] == client_id]
# Sélection de deux caractéristiques
feature1 = st.selectbox("Choisissez la première caractéristique", client_data.columns, index=client_data.columns.get_loc('num__AMT_INCOME_TOTAL'))
feature2 = st.selectbox("Choisissez la deuxième caractéristique", client_data.columns, index=client_data.columns.get_loc('num__AMT_CREDIT'))

# Détection du type des variables
is_feature1_numeric = pd.api.types.is_numeric_dtype(client_data[feature1])
is_feature2_numeric = pd.api.types.is_numeric_dtype(client_data[feature2])
fig, ax = plt.subplots(figsize=(8, 5))
if is_feature1_numeric and is_feature2_numeric:
    # Scatter plot pour 2 variables numériques
    sns.scatterplot(x=client_data[feature1], y=client_data[feature2], alpha=0.5, ax=ax, label="Tous les clients")
    sns.scatterplot(x=client_info[feature1], y=client_info[feature2], color='red', ax=ax, label="Client sélectionné", s=100)
    ax.set_title(f"Relation entre {feature1} et {feature2}")

elif not is_feature1_numeric and is_feature2_numeric:
    # Boxplot pour une variable catégorielle et une numérique
    sns.boxplot(x=client_data[feature1], y=client_data[feature2], ax=ax)

    # Ajout du client sélectionné avec un léger "jitter" pour éviter qu'il se superpose exactement à la médiane
    x_pos = np.where(client_data[feature1].unique() == client_info[feature1])[0][0]  
    ax.scatter(x_pos, client_info[feature2], color='red', s=100, label="Client sélectionné")
    ax.set_title(f"Distribution de {feature2} en fonction de {feature1}")
elif is_feature1_numeric and not is_feature2_numeric:
    # Même logique que ci-dessus mais avec les axes correctement ordonnés
    sns.boxplot(x=client_data[feature2], y=client_data[feature1], ax=ax)
    x_pos = np.where(client_data[feature2].unique() == client_info[feature2])[0][0]  
    ax.scatter(x_pos, client_info[feature1], color='red', s=100, label="Client sélectionné")
    ax.set_title(f"Distribution de {feature1} en fonction de {feature2}")
elif not is_feature1_numeric and not is_feature2_numeric:
    # Countplot pour 2 variables catégorielles
    sns.countplot(x=client_data[feature1], hue=client_data[feature2], ax=ax)
    ax.set_title(f"Répartition de {feature2} en fonction de {feature1}")
ax.legend()
st.pyplot(fig)

# ------------------------------------------------------------------------------------

st.header("Matrice de corrélation des features numériques")

# Sélectionner uniquement les colonnes numériques
numeric_cols = client_data.select_dtypes(include=['number']).columns[:15].tolist()

# Vérifier s'il y a au moins une colonne numérique
if numeric_cols:
    # Calculer la matrice de corrélation
    corr_matrix = client_data[numeric_cols].corr()

    # Sélectionner une feature
    selected_feature = st.selectbox("Choisissez une feature à mettre en évidence", numeric_cols)

    # Création d'un masque pour mettre en évidence la colonne et la ligne sélectionnées
    mask = np.zeros_like(corr_matrix)
    idx = numeric_cols.index(selected_feature)
    mask[idx, :] = 1  # Ligne
    mask[:, idx] = 1  # Colonne

    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
                cbar=True, mask=(1 - mask), alpha=0.7)  # Applique un masque léger sur les autres valeurs

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", center=0, linewidths=0.5, ax=ax,
                cbar=False, mask=mask, alpha=1)  # Mise en surbrillance de la feature sélectionnée

    # Ajouter un titre
    ax.set_title(f"Matrice de corrélation - {selected_feature} mis en évidence", fontsize=14)

    # Afficher la heatmap dans Streamlit
    st.pyplot(fig)

else:
    st.write("Aucune colonne numérique disponible pour calculer la matrice de corrélation.")

