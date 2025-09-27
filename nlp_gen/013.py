import tensorflow as tf

data = 'In the town of Athy one Jeremy Lanigan\nBattered away til he hadn\'t a pound.\nHis father died and made him a'

corpus = data.lower().split('\n')

vectorize_layer = tf.keras.layers.TextVectorization()

vectorize_layer.adapt(corpus)

vocabulary = vectorize_layer.get_vocabulary()
vocab_size = len(vocabulary)
