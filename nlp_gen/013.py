import tensorflow as tf
import numpy as np

data = 'In the town of Athy one Jeremy Lanigan\nBattered away til he hadn\'t a pound.\nHis father died and made him a'

corpus = data.lower().split('\n')

vectorize_layer = tf.keras.layers.TextVectorization()

vectorize_layer.adapt(corpus)

vocabulary = vectorize_layer.get_vocabulary()
vocab_size = len(vocabulary)

input_sequences = []

for line in corpus:
    sequence = vectorize_layer(line).numpy()
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i + 1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

print(max_sequence_len)

input_sequences = np.array(tf.keras.utils.pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding='pre'
))

