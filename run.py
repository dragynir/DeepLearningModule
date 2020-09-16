import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import augmentations.image_augmentations as T

SEED = 2020
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.experimental.AUTOTUNE


augs = T.Compose([
    T.RandomBrightness(0.5, p=1.0),
    # T.RandomCentralCrop(0.5, 0.7, p=1.0),
    T.RandomPad((0.2, 0.4), (0.3, 0.4), p=1.0)
])

image = tf.image.decode_jpeg(tf.io.read_file('res\image.jpg'))
aug_image = augs(image)


fig, ax = plt.subplots(1, 2, figsize=(14, 14))
ax[0].imshow(image.numpy())
ax[1].imshow(aug_image.numpy())
plt.show()