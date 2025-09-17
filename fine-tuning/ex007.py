import tensorflow as tf

local_weights_file = r'/home/joao/tensorflow-journey/fine-tuning/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

pre_treined_model = tf.keras.applications.inception_v3.InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_treined_model.load_weights(local_weights_file)

pre_treined_model.summary()

last_layer = pre_treined_model.get_layer('mixed7')

last_output = last_layer.output

x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

model = tf.keras.Model(pre_treined_model.input, x)

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    epochs=20,
    verbose=2
)