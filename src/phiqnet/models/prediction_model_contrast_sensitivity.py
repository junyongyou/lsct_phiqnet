from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Average, \
    multiply, Lambda, Conv2D, Concatenate
from tensorflow.keras import backend as K


def channel_spatial_attention(input_feature, n_quality_levels, name, return_feature_map=False, return_features=False):
    """
    Attention model for implementing channel and spatial attention in IQA
    :param input_feature: feature maps from FPN or backbone
    :param n_quality_levels: 1 for MOS prediction and 5 for score distribution
    :param name: name of individual layers
    :param return_feature_map: flag to return feature map or not
    :param return_features: flag to return feature vector or not
    :return: output of attention module
    """
    channel_input = input_feature
    spatial_input = input_feature

    channel = input_feature.shape[-1]
    shared_dense_layer = Dense(channel,
                               kernel_initializer='he_normal',
                               use_bias=True,
                               bias_initializer='zeros',
                               activation='sigmoid'
                               )

    avg_pool_channel = GlobalAveragePooling2D()(channel_input)
    avg_pool_channel = Reshape((1, channel))(avg_pool_channel)
    avg_pool_channel = shared_dense_layer(avg_pool_channel)

    max_pool_channel = GlobalMaxPooling2D()(channel_input)
    max_pool_channel = Reshape((1, channel))(max_pool_channel)
    max_pool_channel = shared_dense_layer(max_pool_channel)

    channel_weights = Average()([avg_pool_channel, max_pool_channel])

    avg_pool_spatial = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(spatial_input)
    max_pool_spatial = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(spatial_input)
    spatial_weights = Concatenate(axis=3)([avg_pool_spatial, max_pool_spatial])
    spatial_weights = Conv2D(filters=1,
                             kernel_size=7,
                             strides=1,
                             padding='same',
                             activation='sigmoid',
                             kernel_initializer='he_normal',
                             use_bias=False)(spatial_weights)

    outputs = multiply([multiply([input_feature, channel_weights]), spatial_weights])

    if return_feature_map:
        return outputs

    outputs = GlobalAveragePooling2D(name=name + '_avg_pool')(outputs)
    if return_features:
        return outputs

    if n_quality_levels > 1:
        outputs = Dense(n_quality_levels, activation='softmax', name=name + '_fc_prediction')(outputs)
    else:
        outputs = Dense(n_quality_levels, activation='linear', name=name + 'fc_prediction')(outputs)

    return outputs
