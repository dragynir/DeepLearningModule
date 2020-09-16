
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, PReLU
from tensorflow.keras.layers import Lambda, concatenate, add
from tensorflow.keras.regularizers import l2

def bilinear_resize(x, rsize_x, rsize_y):
    return tf.image.resize(x, [rsize_x, rsize_y])

def conv_bn_prelu(x, filters, kernal_size=(3,3), strides=1,
                kernel_initializer='he_uniform', kernel_regularizer=None):

    x = Conv2D(filters, kernel_size=(3, 3), strides=strides,
                padding='same', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(x)

    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)

    return x

def decoder_conv_bn_prelu(x, filters, up_resize_x, up_resize_y, skip=None):

    x = Lambda(lambda r: bilinear_resize(r, up_resize_x, up_resize_y))(x)

    if skip is not None:
        x = concatenate([x, skip], axis = 3)

    x = conv_bn_prelu(x, filters)
    x = conv_bn_prelu(x, filters)

    return x

def resnet_identity(x, filters):

    x_skip = x
    f1, f2 = filters

    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    x = add([x, x_skip])
    x = PReLU(shared_axes=[1, 2])(x)

    return x



def resnet_conv(x, s, filters):

    x_skip = x
    f1, f2 = filters
    s1, s2 = s

    x = MaxPool2D(pool_size=(s1, s2), padding='valid')(x)
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)

    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)

    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    x_skip = MaxPool2D(pool_size=(s1, s2), padding='valid')(x_skip)
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x_skip)

    x_skip = BatchNormalization()(x_skip)

    x = add([x, x_skip])
    x = PReLU(shared_axes=[1, 2])(x)

    return x



def ctc_lambda_func(args):
    '''

        y_true, y_pred, input_length, label_length = args

    '''
    y_true, y_pred, input_length, label_length = args
    y_pred = y_pred[:, :, :]
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
