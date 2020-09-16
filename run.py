import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import augs.imaugs as T


SEED = 213
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.experimental.AUTOTUNE

segm_augs = T.SegmCompose([
    T.RandomPad((0.2, 0.4), (0.3, 0.4), p=1.0),
])

augs = T.Compose([
    T.OneOf([

        T.RandomHue(0.3, p=1.0),
        T.RandomBrightness(0.5, p=1.0)

        ]),
    T.OneOf([
        T.RandomCentralCrop(0.5, 0.7, p=1.0),
        T.RandomPad((0.2, 0.4), (0.3, 0.4), p=1.0)
    ]),
    T.RandomCentralCrop(0.4, 0.5),
    T.GaussianNoise(0.0, 1.2),
    T.RandomPad((0.2, 0.4), (0.3, 0.4), p=1.0),
    T.RandomBrightness(0.5, p=1.0)
])

image = tf.image.decode_jpeg(tf.io.read_file('res\image.jpg'))
aug_image = augs(image)
aug2_image, _ = segm_augs(image, image)


fig, ax = plt.subplots(1, 3, figsize=(14, 14))
ax[0].imshow(image.numpy())
ax[1].imshow(aug_image.numpy())
ax[2].imshow(aug2_image.numpy())
plt.show()