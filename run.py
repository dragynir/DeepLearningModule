import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



SEED = 2020
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.experimental.AUTOTUNE


image = tf.image.decode_jpeg(tf.io.read_file('res\image.jpg'))


