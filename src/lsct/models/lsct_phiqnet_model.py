from tensorflow.keras.layers import Input, TimeDistributed
from tensorflow.keras.models import Model

from lsct.models.cnn_1d import CNN1D
from lsct.models.video_quality_transformer import VideoQualityTransformer


def create_model(clip_length=16, feature_length=1280, cnn_filters=(32, 64), pooling_sizes=(4, 4),
                 transformer_params=(2, 64, 4, 64), strides=1, dropout_rate=0.1):
    """
    Create the LSCT-PHIQNet model for NR-VQA
    :param clip_length: clip length
    :param feature_length: length of frame PHIQNet features, default is 1280=5*256
    :param cnn_filters: CNN filters for the 1D CNN
    :param pooling_sizes: Pooling sizes for the 1D CNN
    :param transformer_params: Transformer parameters
    :param strides: stride in 1D CNN
    :param dropout_rate: dropout rate for both 1D CNN and Transformer
    :return: the LSCT-PHIQNet model
    """
    using_dropout = dropout_rate > 0
    cnn_model = CNN1D(filters=cnn_filters, pooling_sizes=pooling_sizes, stride_size=strides, using_dropout=using_dropout,
                      dropout_rate=dropout_rate)
    input_shape = (None, clip_length, feature_length)

    inputs = Input(shape=input_shape)
    x = TimeDistributed(cnn_model)(inputs)

    transformer = VideoQualityTransformer(
        num_layers=transformer_params[0],
        d_model=transformer_params[1],
        num_heads=transformer_params[2],
        mlp_dim=transformer_params[3],
        dropout=dropout_rate,
    )
    x = transformer(x)

    model = Model(inputs=inputs, outputs=x)

    return model
