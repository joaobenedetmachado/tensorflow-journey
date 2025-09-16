import tensorflow as tf
import pathlib

train_dir = '/home/joao/tensorflow-journey/image_manipulation/train'
val_dir = pathlib.Path('/home/joao/tensorflow-journey/image_manipulation/test1')

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

data_augmentation = tf.keras.models.Sequential(
    [
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2, fill_mode='nearest'),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode='nearest'),
        tf.keras.layers.RandomZoom(0.2, fill_mode='nearest')
    ]
)

model_without_aug = create_model()

model_with_aug = tf.keras.models.Sequential(
    [
        data_augmentation,
        model_without_aug
    ]
)

model_with_aug.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
    metrics=['accuracy']
)