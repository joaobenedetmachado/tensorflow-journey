import tensorflow_datasets as tfds
import tensorflow as tf

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

single_ex = list(imdb['train'].take(1))[0]

print(single_ex)

# test train "split"

train_data, test_data = imdb['train'], imdb['test']

train_reviews = train_data.map(lambda review, label: review)
train_labels = train_data.map(lambda review, label: label)

test_reviews = test_data.map(lambda review, label: review)
test_labels = test_data.map(lambda review, label: label)

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=10000)

vectorize_layer.adapt(train_reviews)

def padding(sequences):
    sequences = sequences.ragged_batch(batch_size=sequences.cardinality())
    sequences = sequences.get_single_element()

    padded_seq = tf.keras.utils.pad_sequences(sequences.numpy(), maxlen=120,
                                              truncating='post', padding='pre')
    
    padded_seq = tf.data.Dataset.from_tensor_slices(padded_seq)

    return padded_seq


train_sequences = train_reviews.map(lambda text: vectorize_layer(text).apply(padding))
test_sequences = test_reviews.map(lambda text: vectorize_layer(text).apply(padding))
