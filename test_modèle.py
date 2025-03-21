import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model



##########################Ce code boucle sur le répertoire screw_dataset/test/good ou bad qui contient des images et fait la prédicition #########################
# Charger le modèle
model_path = "screw_classifier.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Erreur : le fichier {model_path} n'existe pas !")

model = load_model(model_path)
class_names = ["bad_augmented", "good"]  # Vérifie que ça correspond à ton modèle

# Chemins des dossiers
test_bad_dir = "./screw_dataset/test/test_bad"
test_good_dir = "./screw_dataset/test/test_good"

# Fonction de prédiction
def predict_image(model, img_path):
    IMG_SIZE = (224, 224)  # Taille attendue par le modèle
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension batch
    img_array = img_array / 255.0  # Normaliser

    # Faire la prédiction
    prediction = model.predict(img_array)
    predicted_class = int(prediction[0][0] >= 0.5)  # Vérifie si 0 ou 1
    confidence = prediction[0][0] if predicted_class == 1 else 1 - prediction[0][0]

    return class_names[predicted_class], confidence

# Tester toutes les images
def test_all_images(test_dir, true_label):
    if not os.path.exists(test_dir):
        print(f"Le dossier {test_dir} n'existe pas !")
        return

    images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\nTesting images in {test_dir} ({len(images)} images)\n")

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        predicted_label, confidence = predict_image(model, img_path)
        print(f"{img_name} -> Prédiction : {predicted_label} ({confidence*100:.2f}%) | Réel : {true_label}")

# Tester les deux dossiers
test_all_images(test_bad_dir, "bad")
test_all_images(test_good_dir, "good")
