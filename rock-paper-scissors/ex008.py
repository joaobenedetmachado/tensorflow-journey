import tensorflow as tf

train_dir = '/home/joao/tensorflow-journey/rock-paper-scissors/rps'

test_dir = '/home/joao/tensorflow-journey/rock-paper-scissors/rps-test-set'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(150, 150),
    batch_size=20,
    label_mode='categorical'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(150, 150),
    batch_size=20,
    label_mode='categorical'
)
