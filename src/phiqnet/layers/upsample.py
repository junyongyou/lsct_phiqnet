import tensorflow as tf
from tensorflow.keras.layers import Layer


class Upsample(Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        """Upsamples a tensor
        :param inputs: List of [source, target] tensors
        :param kwargs:
        :return: Upsampled tensor
        """
        source, target = inputs
        target_shape = tf.keras.backend.shape(target)
        if tf.keras.backend.image_data_format() == 'channels_first':
            source = tf.transpose(source, (0, 2, 3, 1))
            output = tf.image.resize(source, (target_shape[2], target_shape[3]),
                                     method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = tf.transpose(output, (0, 3, 1, 2))
            return output
        else:
            return tf.image.resize(source, (target_shape[1], target_shape[2]),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def compute_output_shape(self, input_shape):
        """
        Compute the output shapes given the input shapes
        :param input_shape: List of input shapes [boxes, classification, other[0], other[1], ...]
        :return: Tuple representing the output shapes
        """
        if tf.keras.backend.image_data_format() == 'channels_first':
            return (input_shape[0][0], input_shape[0][1]) + input_shape[1][2:4]
        else:
            return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)
