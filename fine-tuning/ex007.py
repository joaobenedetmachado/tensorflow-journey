import tensorflow as tf

local_weights_file = r'c:\Users\joao.193922\tensorflow-journey\fine-tuning\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_treined_model = tf.keras.applications.inception_v3.InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_treined_model.load_weights(local_weights_file)

pre_treined_model.summary()