"""
Two loss functions that might be used in PHIQNet.
"""
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy


def distribution_loss(y_true, y_pred):
    """
    Loss on quality score distributions
    :param y_true: y_true
    :param y_pred: y_pred
    :return: loss
    """
    mos_scales = np.array([1, 2, 3, 4, 5])
    return K.mean(K.square((y_pred - y_true) * mos_scales))  # MSE


def ordinal_loss(y_true, y_pred):
    """
    A simple ordinal loss based on quality score distributions
    :param y_true: y_true
    :param y_pred: y_pred
    :return: loss
    """
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    # return (1.0 + model_weights) * sigmoid_focal_crossentropy(y_true, y_pred)
    return (1.0 + weights) * categorical_crossentropy(y_true, y_pred)