#  声音特征提取
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import matplotlib.pyplot as plt

# tf.audio.encode_wav(
#     audio, sample_rate, name=None
# )
print(tf.__version__)
