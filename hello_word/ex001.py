# o hello word 

import tensorflow as tf
import numpy as np

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(units=1) # 1 neuronio so
        # no keras. o   Dense, é a camada de neuronios conectados
    ]
)

model.compile(optimizer='sgd', loss="mean_squared_error")

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500) # as epocas sao os loops, entao ele treina o x para encontrar o y nesse caso, 500 vezes

print(f'model predicion: {model.predict(np.array([10.0]), verbose=0).item():.5f}')