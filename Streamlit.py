# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import load_model
# from PIL import Image

# # Charger le modèle
# MODEL_PATH = "screw_classifier.h5"

# @st.cache_resource
# def load_my_model():
#     return load_model(MODEL_PATH)

# model = load_my_model()

# # Classes
# class_names = ["bad_augmented", "good"]
# # Fonction de prédiction
# def predict(img, threshold=0.5):
#     IMG_SIZE = (224, 224)
    
#     # Convertir l'image en tableau numpy
#     img = img.resize(IMG_SIZE)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
#     img_array = img_array / 255.0  # Normaliser

#     # Prédiction
#     prediction = model.predict(img_array)
#     predicted_class = int(prediction[0][0] >= threshold)
#     confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

#     return class_names[predicted_class], confidence

# # Interface Streamlit
# st.title("🔩 Détection de vis défectueuses")
# st.write("Dépose une image et clique sur '📊 Lancer la prédiction'.")

# # Upload d'image
# uploaded_file = st.file_uploader("📂 Parcourir une image", type=["jpg", "png", "jpeg"])

# # Si une image est uploadée, on l'affiche
# if uploaded_file is not None:
#     img = Image.open(uploaded_file)
#     st.image(img, caption="🖼️ Image chargée", use_column_width=True)

#     # Ajouter un bouton pour exécuter la prédiction
#     if st.button("📊 Lancer la prédiction"):
#         with st.spinner("🔍 Analyse en cours..."):
#             label, confidence = predict(img)

#         # Affichage dynamique avec couleur
#         if label == "good":
#             st.success(f"✅ **Prédiction : {label}**")
#         else:
#             st.error(f"❌ **Prédiction : {label}**")

#         st.write(f"### 🔢 Confiance : **{confidence*100:.2f}%**")
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Charger le modèle
MODEL_PATH = "screw_classifier.h5"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# Classes
class_names = ["La vis n'est pas conforme", "La vis est conforme"]

# Fonction de prédiction
def predict(img, threshold=0.5):
    IMG_SIZE = (224, 224)
    
    # Convertir l'image en tableau numpy
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    img_array = img_array / 255.0  # Normaliser

    # Prédiction
    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] >= threshold)
    confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

    return class_names[predicted_class], confidence

# Interface Streamlit
st.title("🔩 Détection de vis défectueuses")
st.write("Dépose une image et clique sur '📊 Lancer la prédiction'.")

# Upload d'image
uploaded_file = st.file_uploader("📂 Parcourir une image", type=["jpg", "png", "jpeg"])

# Si une image est uploadée, on l'affiche
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼️ Image chargée", use_column_width=True)

    # Ajouter un bouton pour exécuter la prédiction
    if st.button("📊 Lancer la prédiction"):
        with st.spinner("🔍 Analyse en cours..."):
            label, confidence = predict(img)

        # Affichage dynamique avec couleur
        if label == "La vis est conforme":
            st.success(f"✅ **Prédiction : {label}**")
        else:
            st.error(f"❌ **Prédiction : {label}**")

        st.write(f"### 🔢 Confiance : **{confidence*100:.2f}%**")
