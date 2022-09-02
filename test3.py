import tensorflow as tf
from tensorflow import keras



# imported = tf.saved_model.load('./model')
# fun = imported.signatures['serving_default']
# print(imported)
# print(fun(tf.ones([1,28,28,3])))

reconstructed_model = keras.models.load_model("my_model")
