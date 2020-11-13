from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Masking, BatchNormalization, Dropout, Input, Bidirectional
from tensorflow.keras.models import Model

from lsct.models.cnn_1d import CNN1D


def create_cnn_lstm_model(clip_length, feature_length=1280, cnn_filters=(32, 64), pooling_sizes=(4, 4),
                          lstm_filters=(32, 64), mlp_filters=(64, 32, 8), using_dropout=True, using_bidirectional=False,
                          using_cnn=True, dropout_rate=0.1):
    """
    Create CNN-LSTM model for VQA
    :param clip_length: clip length
    :param feature_length: feature length
    :param cnn_filters: filters in 1D CNN
    :param pooling_sizes: pooling sizes in 1D CNN
    :param lstm_filters: filters in LSTM
    :param mlp_filters: filters in the MLP head
    :param using_dropout: flag to use dropout or not
    :param using_bidirectional: flag to use bidirectional LSTM or not
    :param using_cnn: flag to use 1D CNN or not
    :param dropout_rate: dropout rate
    :return: CNN-LSTM model
    """
    if using_cnn:
        cnn_model = CNN1D(filters=cnn_filters, pooling_sizes=pooling_sizes, using_dropout=using_dropout,
                          dropout_rate=dropout_rate)
        input_shape = (None, clip_length, feature_length)
    else:
        input_shape = (None, clip_length)
    inputs = Input(shape=input_shape)
    if using_cnn:
        x = TimeDistributed(cnn_model)(inputs)
    else:
        x = inputs
    x = Masking(mask_value=0.)(x)
    for i, lstm_filter in enumerate(lstm_filters):
        if i < len(lstm_filters) - 1:
            if using_bidirectional:
                x = Bidirectional(LSTM(lstm_filter, return_sequences=True))(x)
            else:
                x = LSTM(lstm_filter, return_sequences=True)(x)
        else:
            if using_bidirectional:
                x = Bidirectional(LSTM(lstm_filter))(x)
            else:
                x = LSTM(lstm_filter)(x)

    for mlp_filter in mlp_filters:
        x = Dense(mlp_filter)(x)
        x = BatchNormalization()(x)
        if using_dropout:
            x = Dropout(dropout_rate)(x)

    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model
