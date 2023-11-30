# write a code to check the model checkpoint tf_model.h5 and config.json

import tensorflow as tf

# load the model with config.json
config = tf.compat.v1.keras.models.model_from_json(open('config.json').read())
# load the weights
config.load_weights('tf_model.h5')
# print the model summary
print(config.summary())