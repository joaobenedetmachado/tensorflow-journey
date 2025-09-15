import tensorflow as tf
import pandas as pd

TRAIN_DIR = x

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    imagem_size=(28, 28),
    batch_size=128,
    label_mode='binary'
) 

rescale_layer = tf.keras.layers.Rescaling(scale = 1./255)

train_dataset_scaled = train_dataset.map(
    lambda image, label : (rescale_layer(image), label)
)

train_dataset_final = (
    train_dataset_scaled
    .cache()
    .shuffle(buffer_size=1000)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

model = tf.keras.models.Sequential(
    tf.keras.Input(shape=(300, 300, 3)),
    tf.keras.layers.Conv2D(16,(3,3), activation='relu'), # relu = valores negativos sao descartados
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid") # classificacao binaria | homem x cavalo
)

