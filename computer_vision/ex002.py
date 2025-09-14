import numpy as np
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_label) = fashion_mnist.load_data()

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)), # formato dos dados
    tf.keras.layers.Flatten(), # transforma o 28x28 em uma matriz simples linear
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax) # 10 pois tem 10 tipos de inputs, 10 tipos de roupas no caso
])

model.compile(optimizer = tf.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images, test_label)