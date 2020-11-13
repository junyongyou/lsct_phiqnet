"""
Reference: Path Aggregation Network for Instance Segmentation., CVPR'18.
"""
from tensorflow.keras.layers import Conv2D, Add
from phiqnet.layers.upsample import Upsample


def build_pan(C2, C3, C4, C5, feature_size=256, name='pan_', conv_on_P=False):
    """
    Create the PAN layers on top of the backbone features
    :param C2: Feature stage C2 from the backbone
    :param C3: Feature stage C3 from the backbone
    :param C4: Feature stage C4 from the backbone
    :param C5: Feature stage C5 from the backbone
    :param feature_size: feature size to use for the resulting feature levels, set as the lowest channel dimension in the feature maps, i.e., 256
    :param name: name for naming the layer
    :param conv_on_P: flag to use or not another conv-layer on feature maps
    :return: pyramidical feature maps [N2, N3, N4, N5]
    """
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C5_reduced')(C5)
    P5_upsampled = Upsample(name=name + 'P5_upsampled')([P5, C4])

    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C4_reduced')(C4)
    P4 = Add(name=name + 'P4_merged')([P5_upsampled, P4])
    P4_upsampled = Upsample(name=name + 'P4_upsampled')([P4, C3])

    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C3_reduced')(C3)
    P3 = Add(name=name + 'P3_merged')([P4_upsampled, P3])
    P3_upsampled = Upsample(name=name + 'P3_upsampled')([P3, C2])

    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C2_reduced')(C2)
    P2 = Add(name=name + 'P2_merged')([P3_upsampled, P2])

    if conv_on_P:
        P5 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P5')(P5)
        P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P4')(P4)
        P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P3')(P3)
        P2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P2')(P2)

    N2 = P2

    N2_reduced = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='N2_reduced')(N2)
    N3 = Add(name=name + 'N3_merged')([N2_reduced, P3])
    N3 = Conv2D(feature_size, kernel_size=3, strides=1, activation='relu', padding='same', name=name + 'N3')(N3)

    N3_reduced = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='N3_reduced')(N3)
    N4 = Add(name=name + 'N4_merged')([N3_reduced, P4])
    N4 = Conv2D(feature_size, kernel_size=3, strides=1, activation='relu', padding='same', name=name + 'N4')(N4)

    N4_reduced = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='N4_reduced')(N4)
    N5 = Add(name=name + 'N5_merged')([N4_reduced, P5])
    N5 = Conv2D(feature_size, kernel_size=3, strides=1, activation='relu', padding='same', name=name + 'N5')(N5)

    return [N2, N3, N4, N5]