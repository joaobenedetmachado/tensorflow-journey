import tensorflow as tf

sentences = [
    "I love machine learning",
    "I love deep learning",
    "You love Machine learning",
    "Do think Deep Learning is fun?",
]

vectorize_layer = tf.keras.layers.TextVectorization()

vectorize_layer.adapt(sentences)

vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False)

sequence = vectorize_layer(sentences)

for index, word in enumerate(vocabulary):
    print(f"{index}: {word}")

print(sequence)

sentences_dataset = tf.data.Dataset.from_tensor_slices(sentences)

sequences = sentences_dataset.map(vectorize_layer)

print(sequences)

for sentence, sequence in zip(sentences, sequences):
    print(f"{sentence} => {sequence}")

new_data = [
    "I love AI",
    "I love ML",
    "You love AI",
    "Do you think AI is fun?",
]

new_vectorize_layer = tf.keras.layers.TextVectorization()

new_vectorize_layer.adapt(new_data)

test_data = [
    "i really love ai and ml",
    "do you think ai is the future?",
]

test_sequence = new_vectorize_layer(test_data)

print(test_sequence)

new_vocabulary = new_vectorize_layer.get_vocabulary(include_special_tokens=False)

new_sequence = new_vectorize_layer(new_data)

for index, word in enumerate(new_vocabulary):
    print(f"{index}: {word}")
