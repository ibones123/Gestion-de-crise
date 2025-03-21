#######################GOOOOOOOOOOOOOOOOOOOOOODD222222222222222222é############################################

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from tensorflow.keras.metrics import Recall
# Fixer le seed pour la reproductibilité
tf.random.set_seed(42)

# Paramètres
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.2
DATA_DIR = "./screw_dataset"

# Chargement du dataset
train_dataset = image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="training", seed=42,
    labels="inferred", label_mode="int", batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True
)
val_dataset = image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="validation", seed=42,
    labels="inferred", label_mode="int", batch_size=BATCH_SIZE, image_size=IMG_SIZE, shuffle=True
)

# Obtenir les classes
class_names = train_dataset.class_names
print("Classes détectées :", class_names)

# Modèle CNN amélioré
data_augmentation = tf.keras.Sequential([
     layers.RandomFlip("horizontal"),
     layers.RandomRotation(0.2),
     layers.RandomZoom(0.2)
])

# Normalisation
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

train_dataset = train_dataset.map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.map(preprocess).cache().prefetch(buffer_size=tf.data.AUTOTUNE)



model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(3, 3),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(3, 3),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')  # ✅ 1 seule sortie pour un problème binaire
])

# Compilation
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy',Recall()]
)



# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, verbose=1)
]

# Entraînement
EPOCHS = 30
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks, verbose=1)

# Évaluation
_, accuracy, recall = model.evaluate(val_dataset, verbose=1)
print(f"Accuracy : {accuracy:.4f}, Recall : {recall:.4f}")



y_true, y_pred_probs = [], []
for images, labels in val_dataset:
    y_true.extend(labels.numpy())  # Labels réels
    y_pred_probs.extend(model.predict(images))  # Probabilités prédites

# Convertir en numpy array
y_true = np.array(y_true)
y_pred_probs = np.array(y_pred_probs).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

# Matrice de confusion
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.title("Matrice de Confusion")
plt.show()

# Courbes ROC
plt.figure(figsize=(7, 5))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Classe {class_names[i]} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Faux positifs")
plt.ylabel("Vrais positifs")
plt.title("Courbe ROC")
plt.legend()
plt.show()

# Courbe précision-rappel
plt.figure(figsize=(7, 5))
for i in range(len(class_names)):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    plt.plot(recall, precision, label='Précision-Rappel')

plt.xlabel("Rappel")
plt.ylabel("Précision")
plt.title("Courbe Précision-Rappel")
plt.legend()
plt.show()

# Courbes de perte et d'accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.legend()
plt.title("Évolution de la perte")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Époques")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Évolution de l'accuracy")
plt.show()
#model.save("screw_classifier.h5")


# for images, labels in train_dataset.take(1):
#     print(f"Exemple de labels : {labels.numpy()}")





