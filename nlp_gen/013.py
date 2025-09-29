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

print(input_sequences)

xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(max_sequence_len-1,)),
    tf.keras.layers.Embedding(vocab_size,64),\
    tf.keras.layers.LSTM(20), # RNN onde a sequencial importa
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    xs, ys, epochs=500
)

seed_text = "Laurence went to"

next_sequence = vectorize_layer(seed_text)

prob = model.predict(next_sequence, verbose=0)

predicted = np.argmax(prob, axis=-1)[0]

output_word = vocabulary[predicted]

seed_text += " " + output_word

print(seed_text)