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