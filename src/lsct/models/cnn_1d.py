from tensorflow.keras.layers import Layer, Conv1D, Input, Dropout, MaxPool1D, Masking
import tensorflow.keras.backend as K
from tensorflow.keras import Model
import tensorflow as tf


class CNN1D(Layer):
    def __init__(self, filters=(32, 64), pooling_sizes=(4, 4), kernel_size=3, stride_size=1, using_dropout=True,
                 using_bias=False, dropout_rate=0.1, **kwargs):
        """
        1D CNN model
        :param filters: filter numbers in the CNN blocks
        :param pooling_sizes: max pooling size in each block
        :param kernel_size: kernel size of CNN layer
        :param stride_size: stride of CNN layer
        :param using_dropout: flag to use dropout or not
        :param using_bias: flag to use bias in CNN or not
        :param dropout_rate: dropout rate if using it
        :param kwargs: other config prams
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.using_dropout = using_dropout
        self.conv1d = []
        self.pooling = []
        self.dropout = []
        for i, s_filter in enumerate(filters):
            self.conv1d.append(Conv1D(s_filter,
                                      kernel_size,
                                      padding='same',
                                      strides=stride_size,
                                      use_bias=using_bias,
                                      name='conv{}'.format(i)
                                      ))
            self.pooling.append(MaxPool1D(pool_size=pooling_sizes[i], name='pool{}'.format(i)))
            if using_dropout:
                self.dropout = Dropout(rate=dropout_rate)

        super(CNN1D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CNN1D, self).build(input_shape)

    def call(self, x, mask=None):
        for i in range(len(self.conv1d)):
            x = self.conv1d[i](x)
            x = self.pooling[i](x)
            if self.using_dropout:
                x = self.dropout(x)
        x = K.squeeze(x, axis=-2)
        return x

    def compute_output_shape(self, input_shape):
        return 1, self.filters[-1]


if __name__ == '__main__':
    input_shape = (16, 5 * 256)
    filters = [32, 64, 128, 256]
    pooling_sizes = [2, 2, 2, 2]
    inputs = Input(shape=input_shape)
    x = CNN1D(filters=filters, pooling_sizes=pooling_sizes)(inputs)
    model = Model(inputs=inputs, outputs=x)
    model.summary()
