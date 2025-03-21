import os
import random
import cv2
import numpy as np
from tqdm import tqdm
#####################"##Ce code fait l'augmentation des images, en equilibrant les données entres elle ##############################""""
# Dossiers
dataset_path = "screw_dataset"
bad_path = os.path.join(dataset_path, "bad")
good_path = os.path.join(dataset_path, "good")
bad_aug_path = os.path.join(dataset_path, "bad_augmented")

# Créer le dossier de sortie s'il n'existe pas
os.makedirs(bad_aug_path, exist_ok=True)

# Compter les images
bad_images = [f for f in os.listdir(bad_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
good_count = len([f for f in os.listdir(good_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
bad_count = len(bad_images)

# Nombre d'images à générer
target_count = good_count - bad_count
if target_count <= 0:
    print("Le dataset est déjà équilibré.")
else:
    print(f"Génération de {target_count} images pour équilibrer le dataset...")

    generated = 0
    index = 0

    while generated < target_count:
        img_name = bad_images[index % bad_count]  # Boucle sur les images existantes
        img_path = os.path.join(bad_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # Rotation aléatoire (90, 180, 270 degrés)
        angle = random.choice([90, 180, 270])
        rotated_img = cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE,
                                       180: cv2.ROTATE_180,
                                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])

        # Sauvegarde
        new_name = f"aug_{generated}_{angle}_{img_name}"
        cv2.imwrite(os.path.join(bad_aug_path, new_name), rotated_img)
        generated += 1
        index += 1

    print("Augmentation terminée !")

import shutil

# Supprimer le répertoire 'bad' après l'augmentation
if os.path.exists(bad_path):
    shutil.rmtree(bad_path)
    print(f"Le répertoire {bad_path} a été supprimé.")
else:
    print(f"Le répertoire {bad_path} n'existe pas.")
