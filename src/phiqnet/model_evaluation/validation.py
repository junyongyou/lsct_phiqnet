"""
This script is to evaluate (calculate the evaluation criteria PLCC, SROCC, RMSE) on individual testing sets.
PHIQNet should be first generated, and then the model_weights file is loaded.
"""
import tensorflow as tf
from phiqnet.models.image_quality_model import phiq_net
from phiqnet.utils.imageset_handler import get_image_scores, get_image_score_from_groups
from phiqnet.model_evaluation.evaluation import ModelEvaluation


def val_main(args):
    if args['n_quality_levels'] > 1:
        using_single_mos = False
    else:
        using_single_mos = True

    if args['model_weights'] is not None and ('resnet' in args['backbone'] or args['backbone'] == 'inception'):
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    val_folders = [
                    # r'..\databases\val\koniq_normal',]
                   r'..\databases\val\koniq_small',]
                   # r'..\databases\train\live',
                   # r'..\databases\val\live']

    koniq_mos_file = r'..\databases\koniq10k_images_scores.csv'
    live_mos_file = r'..\databases\live_mos.csv'

    image_scores = get_image_scores(koniq_mos_file, live_mos_file, using_single_mos=using_single_mos)
    test_image_file_groups, test_score_groups = get_image_score_from_groups(val_folders, image_scores)

    test_image_files = []
    test_scores = []
    for test_image_file_group, test_score_group in zip(test_image_file_groups, test_score_groups):
        test_image_files.extend(test_image_file_group)
        test_scores.extend(test_score_group)

    model = phiq_net(n_quality_levels=args['n_quality_levels'],
                     naive_backbone=args['naive_backbone'],
                     backbone=args['backbone'],
                     fpn_type=args['fpn_type'])
    model.load_weights(args['model_weights'])

    # model1 = phiq_net(n_quality_levels=args['n_quality_levels'],
    #                  naive_backbone=args['naive_backbone'],
    #                  backbone=args['backbone'],
    #                  fpn_type=args['fpn_type'])
    # model1.load_weights(r'..\\model_weights\PHIQNet.h5', by_name=True)
    # model.load_weights(args['model_weights'])
    # for i in range(250):
    #     extracted_weights = model1.layers[i].get_weights()
    #     model.layers[i].set_weights(extracted_weights)

    evaluation = ModelEvaluation(model, test_image_files, test_scores, using_single_mos,
                                 imagenet_pretrain=imagenet_pretrain)
    plcc, srcc, rmse = evaluation.__evaluation__()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

    args = {}
    args['n_quality_levels'] = 5
    args['naive_backbone'] = False
    args['backbone'] = 'resnet50'
    args['fpn_type'] = 'fpn'
    # args['model_weights'] = r'..\databases\results\resnet50_mos_attention_fpn\38_0.0008_0.0208_0.0998_0.2286.h5'
    # args['model_weights'] = r'..\databases\results\resnet50_mos_attention_fpn_lr\96_0.0040_0.0488_0.1023_0.2326.h5'
    # args['model_weights'] = r'..\databases\results\resnet50_mos_attention_bifpn_lr\65_0.0080_0.0699_0.0621_0.1871.h5'
    # args['model_weights'] = r'..\databases\results\resnet50_distribution_attention_fpn_lr\61_0.8988_0.1192_1.0691_0.2386.h5'
    # args['model_weights'] = r'..\databases\results_distribution_loss\resnet50_distribution_attention_bifpn_lr\107_0.0269_0.8673_0.1975_1.0134.h5'
    # args['model_weights'] = r'..\databases\results_distribution_loss\resnet50_distribution_attention_fpn_lr_avg\117_0.0183_0.8621_0.2032_1.0449.h5'
    # args['model_weights'] = r'..\databases\results_distribution_loss\\resnet50_distribution_fpn_lr_avg\118_0.0255_0.8632_0.2084_1.0571.h5'
    # args['model_weights'] = r'..\databases\results_distribution_loss\resnet50_distribution_fpn_lr_avg_cbam_finetune\32_0.0792_0.8892_0.2181_1.0748.h5'
    # args['model_weights'] = r'..\databases\experiments\resnet50_distribution_attention_fpn_finetune\117_0.8532_1.0189.h5'
    # args['model_weights'] = r'..\databases\experiments\resnet50_mos_attention_fpn\74_0.0027_0.1180.h5'
    # args['model_weights'] = r'..\databases\experiments\resnet50_distribution_attention_fpn_no_imageaug\91_0.8545_1.0103.h5'
    # args['model_weights'] = r'..\databases\experiments\resnet50_mos_attention_fpn_finetune\45_0.0003_0.0950.h5'
    # args['model_weights'] = r'..\databases\experiments\koniq_normal\resnet50_mos_attention_fpn\37_0.0102_0.0499.h5'
    args['model_weights'] = r'..\databases\experiments\koniq_normal\resnet50_distribution_attention_fpn_finetune\09_0.8493_0.9294.h5'

    val_main(args)