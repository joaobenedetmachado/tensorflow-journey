import tensorflow_datasets as tfds

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

single_ex = list(imdb['train'].take(1))[0]

print(single_ex)

# test train "split"

train_data, test_data = imdb['train'], imdb['test']

train_reviews = train_data.map(lambda review, label: review)
train_labels = train_data.map(lambda review, label: label)

test_reviews = test_data.map(lambda review, label: review)
test_labels = test_data.map(lambda review, label: label)