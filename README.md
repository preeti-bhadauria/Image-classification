# Image-classification
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from google.colab import drive
drive.mount('/content/drive')

# Check image shapes & file sizes
from PIL import Image

for cls in os.listdir(base_path):
    cls_path = os.path.join(base_path, cls)
    for img in os.listdir(cls_path)[:4]:  # check few images
        img_path = os.path.join(cls_path, img)
        with Image.open(img_path) as im:
            print(cls, img, im.size)
  # Standardize images 
  IMG_SIZE = (128, 128)

X = []
y = []


label_map = {"flash": 0, "non_flash": 1}

for label in label_map:
    folder_path = os.path.join(base_path, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=IMG_SIZE
        )
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0

        X.append(img)
        y.append(label_map[label])

X = np.array(X)
y = np.array(y)
# Load training & validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train.shape
X_test.shape
# model selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

cnn = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(1, activation='sigmoid')
])
# compile
cnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# Train the model
cnn.fit(X_train, y_train, epochs=4)
# TEST
test_loss, test_acc = cnn.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# Confusion metrics
from sklearn.metrics import classification_report, confusion_matrix

y_pred = (cnn.predict(X_test) > 0.5).astype(int).ravel()

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["flash", "no flash"]))
# Add EarlyStopping to prevent overfitting---Optional
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='loss',
    patience=3,
    restore_best_weights=True
)
cnn.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stop]
)
# Plot Sample Images With Prediction
def plot_sample(X, y, index):
    plt.imshow(X[index])
    plt.axis("off")
    plt.title(f"True: {y[index]} | Pred: {y_pred[index]}")
    plt.show()
    
# Data augmentation improved generalization and reduced overfitting by exposing the model to diverse spatial variations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

datagen.fit(X_train)
# build model
cnn_aug = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_aug.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
# train
batch_x, batch_y = next(datagen.flow(X_train, y_train, batch_size=8))

print("Augmented batch X shape:", batch_x.shape)
print("Augmented batch y shape:", batch_y.shape)

history_aug = cnn_aug.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[early_stop]
)
