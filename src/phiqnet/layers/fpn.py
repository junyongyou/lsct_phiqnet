"""
Reference: Feature pyramid networks for object detection, CVPR'17.
"""
from tensorflow.keras.layers import Conv2D, Add
from phiqnet.layers.upsample import Upsample


def build_fpn(C2, C3, C4, C5, feature_size=256, name='fpn_'):
    """
    Create the FPN layers on top of the backbone features
    :param C2: Feature stage C2 from the backbone
    :param C3: Feature stage C3 from the backbone
    :param C4: Feature stage C4 from the backbone
    :param C5: Feature stage C5 from the backbone
    :param feature_size: feature size to use for the resulting feature levels, set as the lowest channel dimension in the feature maps, i.e., 256
    :param name: name for naming the layer
    :return: pyramidical feature maps [P2, P3, P4, P5, P6]
    """

    # upsample C5 to get P5 from the FPN paper
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C5_reduced')(C5)
    P5_upsampled = Upsample(name=name + 'P5_upsampled')([P5, C4])
    P5 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P5')(P5)

    # add P5 elementwise to C4
    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C4_reduced')(C4)
    P4 = Add(name=name + 'P4_merged')([P5_upsampled, P4])
    P4_upsampled = Upsample(name=name + 'P4_upsampled')([P4, C3])
    P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P4')(P4)

    # add P4 elementwise to C3
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C3_reduced')(C3)
    P3 = Add(name=name + 'P3_merged')([P4_upsampled, P3])
    P3_upsampled = Upsample(name=name + 'P3_upsampled')([P3, C2])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P3')(P3)

    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name=name + 'C2_reduced')(C2)
    P2 = Add(name=name + 'P2_merged')([P3_upsampled, P2])
    P2 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name=name + 'P2')(P2)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name=name + 'P6')(C5)

    return [P2, P3, P4, P5, P6]


def build_non_fpn(C2, C3, C4, C5, feature_size=256):
    """
    If no FPS is used, then use a bottle-neck layer to change the channel dimension to 256
    :param C2: Feature stage C2 from the backbone
    :param C3: Feature stage C3 from the backbone
    :param C4: Feature stage C4 from the backbone
    :param C5: Feature stage C5 from the backbone
    :param feature_size: feature size to use for the resulting feature levels, set as the lowest channel dimension in the feature maps, i.e., 256
    :return: pyramidical feature maps [P2, P3, P4, P5, P6]
    """
    P2 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P2_bottleneck')(C2)
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P3_bottleneck')(C3)
    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P4_bottleneck')(C4)
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5_bottleneck')(C5)
    P6 = Conv2D(feature_size, kernel_size=1, strides=2, padding='same', name='P6_bottleneck')(C5)
    return [P2, P3, P4, P5, P6]

