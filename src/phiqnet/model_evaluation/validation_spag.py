from phiqnet.models.image_quality_model import phiq_net
from phiqnet.utils.imageset_handler import get_image_score_from_groups
from phiqnet.model_evaluation.evaluation import ModelEvaluation


def get_image_scores(mos_file):
    image_files = {}
    with open(mos_file, 'r+') as f:
        lines = f.readlines()
        for line in lines:
            content = line.split(',')
            image_file = content[0]
            score = float(content[1]) / 25. + 1
            image_files[image_file] = score

    return image_files


def val_main(args):
    if args['n_quality_levels'] > 1:
        using_single_mos = False
    else:
        using_single_mos = True

    if args['model_weights'] is not None and ('resnet' in args['backbone'] or args['backbone'] == 'inception'):
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    val_folders = [r'F:\SPAG_image_quality_dataset\TestImage']
    spag_mos_file = r'..\databases\spag\image_mos.csv'
    image_scores = get_image_scores(spag_mos_file)
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

    evaluation = ModelEvaluation(model, test_image_files, test_scores, using_single_mos,
                                 imagenet_pretrain=imagenet_pretrain)
    result_file = r'..\databases\spag\result.csv'
    plcc, srcc, rmse = evaluation.__evaluation__(result_file)


if __name__ == '__main__':
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # tf.config.experimental.set_visible_devices(gpus[1], 'GPU')

    args = {}
    # args['result_folder'] = r'..\databases\results'
    args['n_quality_levels'] = 5
    args['naive_backbone'] = False
    args['backbone'] = 'resnet50'
    args['fpn_type'] = 'fpn'
    args['model_weights'] = r'..\\model_weights\PHIQNet.h5'

    val_main(args)