import tensorflow as tf
import pathlib

train_dir = '/home/joao/tensorflow-journey/dogs_x_cats/train'
val_dir = pathlib.Path('/home/joao/tensorflow-journey/dogs_x_cats/test1')

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150,150),
    batch_size=20,
    label_mode='binary'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(150,150),
    batch_size=20,
    label_mode='binary'
)

AUTOTUNE = tf.keras.AUTOTUNE


model = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),  # relu = valores negativos sao descartados
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.summary()