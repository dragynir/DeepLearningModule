import segmentation_models as sm
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.layers import GRU, Bidirectional
from tensorflow.keras.layers import Dense, Activation


from blocks import ctc_lambda_func, resnet_conv



def text_to_labels(text, letters):
    return np.asarray(list((map(lambda x: letters.index(x), text))))

def labels_to_text(labels, letters):
    return np.asarray(list((map(lambda x: letters[x], labels))))




def build_effnetb0_gru_ocr(
    letters_count,
    max_len_str,
    input_shape=(128, 1024),
    backbone_conv_filters=[256, 512],
    cnn_reshape_size=(32, 512),
    rnn_size=256,
    rnn_dropout=0.3,
    optimizer=tf.keras.optimizers.Adam(),
):


    '''
        block4a_expand_activation (Acti (None, 16, 64, 240) 
        block2b_activation (Activation) (None, 32, 128, 144) 
        block6a_activation
    '''

    model = sm.Unet('efficientnetb0', input_shape=input_shape, activation='sigmoid', decoder_block_type='upsampling')

    backbone = tf.keras.Model(inputs=model.input, outputs=model.get_layer('block6a_activation').output)

    backbone_output = backbone.output

    x = resnet_conv(backbone_output, s=[3, 1], filters=backbone_conv_filters)

    x = tf.keras.layers.Reshape(target_shape=cnn_reshape_size, name='reshape')(x)
    
    x = Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', dropout=rnn_dropout))(x)

    x = Bidirectional(GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', dropout=rnn_dropout))(x)

    x = BatchNormalization()(x)

    x = Dense(letters_count, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))(x)
    y_pred = Activation('softmax', name='softmax')(x)


    y_true = tf.keras.layers.Input(name='y_true', shape=[max_len_str], dtype='float32')

    input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')

    label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

    loss = tf.keras.layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_true, y_pred, input_length, label_length])

    model = tf.keras.Model(inputs=[backbone.input, y_true, input_length, label_length], outputs=loss)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    
    return model