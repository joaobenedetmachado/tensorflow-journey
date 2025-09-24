import tensorflow as tf

vocab_size = 10000  # Example vocabulary size

model = tf.keras.Sequential([
    tf.keras.InputLayer(input_shape=(None,)),
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

