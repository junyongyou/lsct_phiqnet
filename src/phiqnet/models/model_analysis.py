"""
This script is for model analysis, i.e., correlation analysis, generation of activation maps...
Not about model training and evaluation.
"""
import os
import numpy as np
from PIL import Image
import tensorflow.keras.backend as K
import scipy.stats
from phiqnet.utils.imageset_handler import get_image_scores, get_image_score_from_groups
from phiqnet.layers.fpn import build_fpn, build_non_fpn
from phiqnet.layers.bi_fpn import build_bifpn
from phiqnet.layers.pan import build_pan
from phiqnet.backbone.ResNest import ResNest
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Lambda
from tensorflow.keras.models import Model
from phiqnet.models.prediction_model_contrast_sensitivity import channel_spatial_attention
from phiqnet.backbone.resnet50 import ResNet50
from phiqnet.backbone.resnet_family import ResNet18
from phiqnet.models.image_quality_model import phiq_net
import matplotlib.pyplot as plt
from phiqnet.backbone.resnet_feature_maps import ResNet152v2, ResNet152
from phiqnet.backbone.vgg16 import VGG16
# import seaborn


def phiq_subnet(n_quality_levels, input_shape=(None, None, 3), naive_backbone=False, backbone='resnet50',
                fpn_type='fpn', pooling='max', return_backbone_maps=False, attention_module=True,
                return_feature_maps=True, return_features=False):
    """
    Generate the feature extraction model based on PHIQNet and use its weights
    :param n_quality_levels: quality level, should be 5
    :param input_shape: image input shape, keep as unspecifized
    :param naive_backbone: flag to use backbone only, i.e., without neck and head, if set to True
    :param backbone: backbone networks (resnet50/18/152v2, resnest, vgg16, etc.), should be 'resnet50
    :param fpn_type: chosen from 'fpn', 'bi-fpn' or 'pan', should be 'fpn'
    :param attention_module: flag to use or not attention module
    :param pooling: choose between 'max' and 'mean'
    :param return_backbone_maps: flag to return backbone activation maps
    :param return_feature_maps: flag to return feature maps or not
    :param return_features: flag to return features or not
    :return: features or feature maps or backbone maps (i.e., activation maps)
    """
    inputs = Input(shape=input_shape)
    n_classes = None
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
    if return_backbone_maps:
        backbone_maps = []
        for backbone_map in backbone_model.outputs:
            if pooling == 'mean':
                backbone_maps.append(Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(backbone_map))
            else:
                backbone_maps.append(Lambda(lambda x: K.max(x, axis=3, keepdims=True))(backbone_map))
        model = Model(inputs=inputs, outputs=backbone_maps)
        model.summary()
        return model

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
            if return_features:
                F = channel_spatial_attention(P, n_quality_levels, 'P{}'.format(i), return_features=True)
                PF.append(F)
            else:
                if return_feature_maps:
                    F = channel_spatial_attention(P, n_quality_levels, 'P{}'.format(i), return_feature_map=True)
                    if pooling == 'mean':
                        F = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(F)
                    else:
                        F = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(F)
                    PF.append(F)
        else:
            outputs = GlobalAveragePooling2D(name='avg_pool_{}'.format(i))(P)
            outputs = Dense(n_quality_levels, activation='softmax', name='fc_prediction_{}'.format(i))(outputs)
            PF.append(outputs)
    outputs = PF

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def evaluation_scales(model, image_files, scores, imagenet_pretrain=True):
    """
    Calculate the correlation of quality prediction at different spatial scales
    :param model: PHIQNet model
    :param image_files: image file paths
    :param scores: image MOS values
    :param imagenet_pretrain: use ImageNet pretrain
    :return:
    """
    predictions_all = []
    mos_scores = []
    mos_scales = np.array([1, 2, 3, 4, 5])

    for image_file, score in zip(image_files, scores):
        image = Image.open(image_file)
        image = np.asarray(image, dtype=np.float32)
        if imagenet_pretrain:
            image /= 127.5
            image -= 1.
        else:
            image[:, :, 0] -= 117.27205081970828
            image[:, :, 1] -= 106.23294835284031
            image[:, :, 2] -= 94.40750328714887
            image[:, :, 0] /= 59.112836751661085
            image[:, :, 1] /= 55.65498543815568
            image[:, :, 2] /= 54.9486100975773

        score = np.sum(np.multiply(mos_scales, score))
        prediction_scales = model.predict(np.expand_dims(image, axis=0))
        predicted = []
        for prediction in prediction_scales:
            predicted.append(np.sum(np.multiply(mos_scales, prediction[0])))

        predictions_all.append(predicted)
        mos_scores.append(score)
        print('Real score: {}, predicted: {}'.format(score, predicted))

    predictions_all = np.array(predictions_all)

    for i in range(predictions_all.shape[1]):
        predictions = np.transpose(predictions_all[:, i])
        PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
        SRCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        print('At scale {}: PLCC: {}, SROCC: {}, RMSE: {}, MAD: {}'.format(i + 2, PLCC, SRCC, RMSE, MAD))


def write_array_image(array):
    sq_array = np.squeeze(np.squeeze(array, axis=0), axis=-1)
    sq_array = 255. * (sq_array - sq_array.min()) / (sq_array.max() - sq_array.min())
    return np.array(sq_array, dtype=np.uint8)


def generate_activations(image_file, pooling, activation_model=None):
    """
    Generate the activation maps from PHIQNet
    :param image_file: image file path
    :param pooling: pooling method, 'max' or 'mean'
    :param activation_model: PHIQNet feature model
    :return: writen to activation map image
    """
    if activation_model is None:
        activation_model = phiq_subnet(n_quality_levels=5,
                                       naive_backbone=False,
                                       backbone='resnet50',
                                       fpn_type='fpn',
                                       pooling=pooling)
        weights = r'..\\model_weights\PHIQNet.h5'
        activation_model.load_weights(weights, by_name=True)

    # image_file = r'..\databases\val\koniq_normal\69161613.jpg'
    result_folder = r'..\databases\activation_maps'
    image_name = os.path.splitext(os.path.basename(image_file))[0]
    result_folder = os.path.join(result_folder, image_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    image = np.asarray(Image.open(image_file), dtype=np.float32)
    image /= 127.5
    image -= 1.
    activations = activation_model.predict(np.expand_dims(image, axis=0))
    for i, activation in enumerate(activations):
        Image.fromarray(write_array_image(activation)).save(
            os.path.join(result_folder, 'PF{}_{}.jpg'.format(i + 2, pooling)))


def correlation_scales():
    """
    Correlation analysis at different spatial scales
    :return:
    """
    original_model = phiq_net(n_quality_levels=5)
    weights = r'..\\model_weights\PHIQNet.h5'

    # feature_model = Model(inputs=model.input, outputs=model.layers[190].layers[14].output)
    model = Model(inputs=original_model.input, outputs=original_model.layers[-1].input)
    model.load_weights(weights, by_name=True)

    folders = [r'..\\databases\train\koniq_normal',]
               # r'..\\databases\train\koniq_small',
               # r'..\\databases\train\live']
    # folders = [r'..\\databases\val\koniq_normal',]
    #            r'..\\databases\val\koniq_small',
    #            r'..\\databases\val\live']

    koniq_mos_file = r'..\\databases\koniq10k_images_scores.csv'
    live_mos_file = r'..\\databases\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file, using_single_mos=False)
    image_file_groups, score_groups = get_image_score_from_groups(folders, image_scores)

    image_files = []
    scores = []
    for train_image_file_group, train_score_group in zip(image_file_groups, score_groups):
        image_files.extend(train_image_file_group)
        scores.extend(train_score_group)

    evaluation_scales(model, image_files, scores, imagenet_pretrain=True)


def generate_backbone_maps(image_file, pooling, original_model, feature_model):
    result_folder = r'..\databases\activation_maps'

    image_name = os.path.splitext(os.path.basename(image_file))[0]
    result_folder = os.path.join(result_folder, image_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    image = np.asarray(Image.open(image_file), dtype=np.float32)
    image /= 127.5
    image -= 1.

    # feature_model.load_weights(imagenet_weights, by_name=True)
    activations = original_model.predict(np.expand_dims(image, axis=0))
    for i, activation in enumerate(activations):
        Image.fromarray(write_array_image(activation)).save(
            os.path.join(result_folder, 'C{}_{}_O.jpg'.format(i + 2, pooling)))

    activations = feature_model.predict(np.expand_dims(image, axis=0))
    for i, activation in enumerate(activations):
        Image.fromarray(write_array_image(activation)).save(
            os.path.join(result_folder, 'C{}_{}_I.jpg'.format(i + 2, pooling)))
    t = 0


def draw_correlations():
    n_groups = 5
    plcc = (0.9489493836885121, 0.9523660711489486, 0.6677299730093762, 0.9488514991823601, 0.940967333512691)
    srocc = (0.9609349753201494, 0.9694109035977723, 0.568864738828304, 0.9689382097673477, 0.9496932104108254)
    rmse = (0.18370818730351188, 0.2117558405987572, 0.48530752914183534, 0.3494118143568881, 0.40784585702538817)

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.65
    gap = 0.01

    rects1 = plt.bar(index, plcc, bar_width,
                     alpha=opacity,
                     color='lightcoral',
                     # edgecolor='black',
                     # hatch='*',
                     label='PLCC')
    # rects1 = seaborn.barplot(plcc)

    rects2 = plt.bar(index + bar_width + gap, srocc, bar_width,
                     alpha=opacity,
                     color='olive',
                     # edgecolor='black',
                     # hatch='x',
                     label='SROCC')

    rects3 = plt.bar(index + 2 * bar_width + 2 * gap, rmse, bar_width,
                     alpha=opacity,
                     color='skyblue',
                     # edgecolor='black',
                     # hatch='o',
                     label='RMSE')

    # plt.xlabel('Scales')
    # plt.ylabel('Criteria')
    plt.title('Evaluation criteria at individual scales')
    plt.xticks(index + bar_width, ('Scale 2', 'Scale 3', 'Scale 4', 'Scale 5', 'Scale 6'))
    plt.legend()

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # correlation_scales()
    # draw_correlations()
    image_folder = r'..\\databases\train\koniq_normal'
    image_files = os.listdir(image_folder)

    # pooling = 'max'
    pooling = 'max'

    activation_model = phiq_subnet(n_quality_levels=5,
                                   naive_backbone=False,
                                   backbone='resnet50',
                                   fpn_type='fpn',
                                   pooling=pooling,
                                   return_feature_maps=True,
                                   return_backbone_maps=False,
                                   return_features=False)
    weights = r'..\\model_weights\PHIQNet.h5'
    activation_model.load_weights(weights, by_name=True)

    original_model = phiq_subnet(n_quality_levels=5, pooling=pooling, return_backbone_maps=True,
                                 return_feature_maps=False, return_features=False)
    imagenet_weights = r'..\pretrained_weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    original_model.load_weights(imagenet_weights, by_name=True)

    feature_model = phiq_subnet(n_quality_levels=5, pooling=pooling, return_backbone_maps=True,
                                return_feature_maps=False, return_features=True)
    feature_model.load_weights(weights, by_name=True)

    for image_file in image_files:
        image_file = os.path.join(image_folder, image_file)
        generate_activations(image_file, pooling, activation_model)
        generate_backbone_maps(image_file, pooling, original_model, feature_model)
        print('{} done'.format(image_file))
