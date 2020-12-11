import tensorflow as tf
from tensorflow.keras import initializers 
from tensorflow.keras import regularizers
from tensorflow.keras import constraints

from tensorflow.keras import activations 
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Layer, Embedding


class Attention(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False, **kwargs):

        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        # self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        # self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_shape_list = input_shape.as_list()

        self.W = self.add_weight(shape=((input_shape_list[-1], input_shape_list[-1])),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape_list[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        # self.u = self.add_weight(shape=(input_shape_list[-1],),
        #                          initializer=self.init,
        #                          name='{}_u'.format(self.name),
        #                          regularizer=self.u_regularizer,
        #                          constraint=self.u_constraint)

        super(Attention, self).build(input_shape.as_list())

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = tf.tensordot(x, self.W, axes=1)

        if self.bias:
            uit += self.b

        uit = activations.tanh(uit)

        a = activations.softmax(uit, axis=1)
        output = x * a
        result = K.sum(output, axis=1)

        return result

        # ait = tf.tensordot(uit, self.u, axes=1)
        #
        # a = activations.exponential(ait)
        #
        # # apply mask after the exp. will be re-normalized next
        # if mask is not None:
        #     # Cast the mask to floatX to avoid float64 upcasting in theano
        #     a *= tf.cast(mask, K.floatx())
        #
        # # in some cases especially in the early stages of training the sum may be almost zero
        # # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        # a /= tf.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        #
        # a = K.expand_dims(a)
        # weighted_input = x * a
        # result = K.sum(weighted_input, axis=1)
        #
        # if self.return_attention:
        #     return [result, a]
        # return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            #TODO use TensorShape here, as done in the else statement.   I'm not sure
            # if this is returning a single tensor, or a list of two so leaving this undone for now.  Suspect this will
            # need to complete if using Sequential rather than Functional API
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return tf.TensorShape([input_shape[0].value, input_shape[-1].value])

