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