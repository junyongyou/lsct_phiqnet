"""
Main function to build PHIQNet.
"""
from phiqnet.layers.fpn import build_fpn, build_non_fpn
from phiqnet.layers.bi_fpn import build_bifpn
from phiqnet.layers.pan import build_pan
from phiqnet.backbone.ResNest import ResNest
from tensorflow.keras.layers import Input, Dense, Average, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from phiqnet.models.prediction_model_contrast_sensitivity import channel_spatial_attention
from phiqnet.backbone.resnet50 import ResNet50
from phiqnet.backbone.resnet_family import ResNet18
from phiqnet.backbone.resnet_feature_maps import ResNet152v2, ResNet152
from phiqnet.backbone.vgg16 import VGG16


def phiq_net(n_quality_levels, input_shape=(None, None, 3), naive_backbone=False, backbone='resnet50', fpn_type='fpn',
             attention_module=True):
    """
    Build PHIQNet
    :param n_quality_levels: 1 for MOS prediction and 5 for score distribution
    :param input_shape: image input shape, keep as unspecifized
    :param naive_backbone: flag to use backbone only, i.e., without neck and head, if set to True
    :param backbone: backbone networks (resnet50/18/152v2, resnest, vgg16, etc.)
    :param fpn_type: chosen from 'fpn', 'bi-fpn' or 'pan'
    :param attention_module: flag to use or not attention module
    :return: PHIQNet model
    """
    inputs = Input(shape=input_shape)
    n_classes = None
    return_feature_maps = True
    if naive_backbone:
        n_classes = 1
        return_feature_maps = False
    fc_activation = None
    verbose = False
    if backbone == 'resnest50':
        backbone_model = ResNest(verbose=verbose,
                                 n_classes=n_classes, dropout_rate=0, fc_activation=fc_activation,
                                 blocks_set=[3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                                 stem_width=32, avg_down=True, avd=True, avd_first=False,
                                 return_feature_maps=return_feature_maps).build(inputs)
    elif backbone == 'resnest34':
        backbone_model = ResNest(verbose=verbose,
                                 n_classes=n_classes, dropout_rate=0, fc_activation=fc_activation,
                                 blocks_set=[3, 4, 6, 3], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                                 stem_width=16, avg_down=True, avd=True, avd_first=False, using_basic_block=True,
                                 return_feature_maps=return_feature_maps).build(inputs)
    elif backbone == 'resnest18':
        backbone_model = ResNest(verbose=verbose,
                                 n_classes=n_classes, dropout_rate=0, fc_activation=fc_activation,
                                 blocks_set=[2, 2, 2, 2], radix=2, groups=1, bottleneck_width=64, deep_stem=True,
                                 stem_width=16, avg_down=True, avd=True, avd_first=False, using_basic_block=True,
                                 return_feature_maps=return_feature_maps).build(inputs)
    elif backbone == 'resnet50':
        backbone_model = ResNet50(inputs,
                                  return_feature_maps=return_feature_maps)
    elif backbone == 'resnet18':
        backbone_model = ResNet18(input_tensor=inputs,
                                  weights=None,
                                  include_top=False)
    elif backbone == 'resnet152v2':
        backbone_model = ResNet152v2(inputs)
    elif backbone == 'resnet152':
        backbone_model = ResNet152(inputs)
    elif backbone == 'vgg16':
        backbone_model = VGG16(inputs)
    else:
        raise NotImplementedError

    if naive_backbone:
        backbone_model.summary()
        return backbone_model

    C2, C3, C4, C5 = backbone_model.outputs
    pyramid_feature_size = 256
    if fpn_type == 'fpn':
        fpn_features = build_fpn(C2, C3, C4, C5, feature_size=pyramid_feature_size)
    elif fpn_type == 'pan':
        fpn_features = build_pan(C2, C3, C4, C5, feature_size=pyramid_feature_size)
    elif fpn_type == 'bifpn':
        for i in range(3):
            if i == 0:
                fpn_features = [C3, C4, C5]
            fpn_features = build_bifpn(fpn_features, pyramid_feature_size, i)
    else:
        fpn_features = build_non_fpn(C2, C3, C4, C5, feature_size=pyramid_feature_size)

    PF = []
    for i, P in enumerate(fpn_features):
        if attention_module:
            PF.append(channel_spatial_attention(P, n_quality_levels, 'P{}'.format(i)))
        else:
            outputs = GlobalAveragePooling2D(name='avg_pool_{}'.format(i))(P)
            outputs = Dense(n_quality_levels, activation='softmax', name='fc_prediction_{}'.format(i))(outputs)
            PF.append(outputs)
    outputs = Average(name='PF_average')(PF)

    # pyramids = Concatenate(axis=1)(PF)
    # outputs = Dense(1, activation='linear', name='final_fc', use_bias=True)(pyramids)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


if __name__ == '__main__':
    input_shape = [None, None, 3]
    # model = phiq_net(n_quality_levels=5, input_shape=input_shape, backbone='resnet152v2')
    # model = phiq_net(n_quality_levels=5, input_shape=input_shape, backbone='resnet50')
    model = phiq_net(n_quality_levels=5, input_shape=input_shape, backbone='vgg16')
