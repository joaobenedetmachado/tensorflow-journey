import tensorflow as tf

sentences = [
    "I love machine learning",
    "I love deep learning",
]

vectorize_layer = tf.keras.layers.TextVectorization()

vectorize_layer.adapt(sentences)

vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False)

print(vocabulary)
# [np.str_('love'), np.str_('learning'), np.str_('i'), np.str_('machine'), np.str_('deep')]