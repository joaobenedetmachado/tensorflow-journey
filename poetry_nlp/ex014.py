import tensorflow as tf
import numpy as np
data = open('/home/joao/tensorflow-journey/poetry_nlp/data.txt').read()

print(data)

input_sequences = []


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


input_sequences = np.array(tf.keras.utils.pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding='pre'
))

print(input_sequences)

xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)

labels = input_sequences[:,-1]


model  = tf.keras.Sequential([
    tf.keras.Input(shape=(max_sequence_len-1,)),
    tf.keras.layers.Embedding(2704, 100),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)), # sequencia importa
    tf.keras.layers.Dense(2704, activation='softmax')
])

adam = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.fit(xs, ys, epochs=100)

seed_text = "help me obi-wan kenobi youre my only hope"

# Define total words to predict
next_words = 100

# Loop until desired length is reached
for _ in range(next_words):

	# Generate the integer sequence of the current line
	sequence = vectorize_layer(seed_text)

	# Pad the sequence
	sequence = tf.keras.utils.pad_sequences([sequence], maxlen=max_sequence_len-1, padding='pre')

	# Feed to the model and get the probabilities for each index
	probabilities = model.predict(sequence, verbose=0)

	# Get the index with the highest probability
	predicted = np.argmax(probabilities, axis=-1)[0]

	# Ignore if index is 0 because that is just the padding.
	if predicted != 0:

		# Look up the word associated with the index.
		output_word = vocabulary[predicted]

		# Combine with the seed text
		seed_text += " " + output_word

# Print the result
print(seed_text)