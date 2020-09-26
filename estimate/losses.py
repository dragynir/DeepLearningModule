import tensorflow.keras.backend as K
import tensorflow as tf


def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    """

    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    [batch_size, widht, height, classes]

    """
    
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1)) 

    numerator = 2. * K.sum(y_pred * y_true, axes)

    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)

    # average over classes and batch
    return 1 - K.mean((numerator + epsilon) / (denominator + epsilon)) 

def dice_coef_loss(y_true, y_pred):
    """
        1 - DC loss
        y_true: tensor

        y_pred: tensor
    """

    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def multi_class_focal_loss(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    Focal loss for multi-class or multi-label problems
         Alpha controls the weight when the true value y_true is 1/0
                 The weight of 1 is alpha, and the weight of 0 is 1-alpha.
         When your model is under-fitting and you have difficulty learning, you can try to apply this function as a loss.
         When the model is too aggressive (whenever it tends to predict 1), try to reduce the alpha
         When the model is too inert (whenever it always tends to predict 0, or a fixed constant, it means that no valid features are learned)
                 Try to increase the alpha and encourage the model to predict 1.
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss_func(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # alpha where 1, and 1 - alpha where 0
        alpha_t = y_true * alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)

        # p or 1 - p 
        y_t = tf.math.multiply(y_true, y_pred) + tf.math.multiply(1-y_true, 1-y_pred)
        ce = -tf.math.log(y_t)

        weight = tf.math.pow(tf.math.subtract(1., y_t), gamma)
        fl = tf.math.multiply(tf.math.multiply(weight, ce), alpha_t)

        fl = tf.math.reduce_sum(fl, axis=-1)

        loss = tf.math.reduce_mean(fl)
        return loss

    return loss_func


def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
         Focal loss for binary classification problems
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    epsilon = 1.e-7

    def loss_func(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (tf.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (tf.ones_like(y_true)-y_true)*(tf.ones_like(y_true)-y_pred) + epsilon
        focal_loss = -alpha_t * tf.math.pow((tf.ones_like(y_true) - p_t), gamma) * tf.math.log(p_t)

        return tf.math.reduce_mean(focal_loss)
    return loss_func


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss