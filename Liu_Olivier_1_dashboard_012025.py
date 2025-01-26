import streamlit as st
import requests
import random
import pandas as pd

# Charger les données des clients (assurez-vous que client_data est chargé correctement)
client_data = pd.read_csv('data/source/client_data.csv')

# Titre de l'application Streamlit
st.title("Prédiction de scoring client")

# Fonction pour tirer un client ID aléatoire
def generate_random_client_id():
    return random.choice(client_data['num__SK_ID_CURR'])

# Vérifier si le client ID est déjà dans le session state
if 'random_client_id' not in st.session_state:
    st.session_state.random_client_id = generate_random_client_id()

# Créer un espace vide pour afficher le client ID
random_client_display = st.empty()

# Afficher le client ID tiré au sort ou mis à jour
random_client_display.write(f"Un client ID a été tiré au sort : {st.session_state.random_client_id}")

# Bouton pour régénérer un nouveau client ID
if st.button("Tirer un nouveau client ID"):
    st.session_state.random_client_id = generate_random_client_id()
    random_client_display.write(f"Un nouveau client ID a été tiré au sort : {st.session_state.random_client_id}")

# Demander à l'utilisateur d'entrer un ID client
client_id = st.number_input("Entrez l'ID du client ou utilisez celui sélectionné", 
                            min_value=1, 
                            value=int(st.session_state.random_client_id))  # Pré-remplir avec le client ID tiré au sort

# Afficher un bouton pour lancer la prédiction
if st.button("Prédire"):

    if client_id:
        # # Faire la requête à l'API Flask
        # response = requests.get(f'http://127.0.0.1:5002/predict?id={client_id}')
        # Faire la requête à l'API Heroku
        response = requests.get(f'https://safe-journey-16014-5c3485a368b4.herokuapp.com/predict?id={client_id}')
        
        # Vérifier si la requête a réussi
        if response.status_code == 200:
            data = response.json()

            # Afficher la prédiction
            st.write(f"Client ID: {data['client_id']}")
            st.write(f"Prédiction: {data['prediction']}")
            st.write(f"Probabilité de la prédiction: {data['prediction_proba']:.4f}")

            # Ajouter un emoji selon la prédiction
            emoji = "😊" if data['prediction'] == 1 else "😟"
            if data['prediction'] == 1:
                st.write(f"Prêt accordé: {emoji}")
            else:
                st.write(f"Prêt refusé: {emoji}")
        else:
            st.write("Erreur: ", response.json()['error'])

# Afficher les informations descriptives du client
client_info = client_data[client_data['num__SK_ID_CURR'] == client_id]
if not client_info.empty:
    st.write("### Informations du client")
    st.write(client_info.T)  # Affichage des infos sous forme de tableau
else:
    st.write("Aucune information trouvée pour cet ID client.")



