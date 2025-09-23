import tensorflow as tf
import keras_nlp
import tensorflow_datasets as tfds

imdb = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_reviews = imdb['train'].map(lambda review, label: review)
train_labels = imdb['train'].map(lambda review, label: label)

keras.nlp.tokenizers.compute_word_piece_tokenizer(
    train_reviews,
    vocabulary_size=10000,
    lowercase=True,
    reserved_tokens=["[PAD]", "[UNK]"],
    vocabulary_output_file="vocab.txt"
)

subword_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary="vocab.txt",
    lowercase=True,
    oov_token="[UNK]"
)