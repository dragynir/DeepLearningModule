import segmentation_models as sm
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import os


os.environ["SM_FRAMEWORK"] = 'tf.keras'

from blocks import decoder_conv_bn_prelu



def build_effnetb0_unet(
        input_shape,
        classes=1,
        activation='sigmoid',
        decoder_filters=(128, 64, 32, 16),          
):

    tf.keras.backend.clear_session()

    model = sm.Unet('efficientnetb0', input_shape=input_shape)

    backbone = tf.keras.Model(inputs=model.input, outputs=model.get_layer('block4a_activation').output)

    skip_connection_layers = ('block3b_expand_activation', 'block2b_expand_activation', 'block1a_activation')

    skips = ([backbone.get_layer(name=i).output for i in skip_connection_layers])

    x = backbone.output

    resize_width, resize_height = input_shape[:2]
    decode_steps = len(decoder_filters)

    for i in range():
        if i < len(skips):
            skip = skips[i]
        else:
            skip = None

        up_width = resize_width / (2 ** (decode_steps - i - 1))
        up_height = resize_height / (2 ** (decode_steps - i - 1))

        x = decoder_conv_bn_prelu(x, decoder_filters[i], up_width, up_height, skip)
        
    x = Conv2D(
            filters=classes,
            kernel_size=(3, 3),
            padding='same',
            use_bias=True,
            activation=activation,
            kernel_initializer='glorot_uniform'
        )(x)

    model = tf.keras.Model(inputs=backbone.input, outputs=x)

    return model
