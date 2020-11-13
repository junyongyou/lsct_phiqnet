from phiqnet.train.train import train_main

if __name__ == '__main__':
    args = {}
    args['multi_gpu'] = 0
    args['gpu'] = 0

    args['result_folder'] = r'..\databases\experiments'
    args['n_quality_levels'] = 1

    args['train_folders'] = [r'..\databases\train\koniq_normal',
                             r'..\databases\train\koniq_small',
                             r'..\databases\train\live']
    args['val_folders'] = [r'..\databases\val\koniq_normal',
                           r'..\databases\val\koniq_small',
                           r'..\databases\val\live']
    args['koniq_mos_file'] = r'..\databases\koniq10k_images_scores.csv'
    args['live_mos_file'] = r'..\databases\live_mos.csv'

    args['naive_backbone'] = False
    args['backbone'] = 'resnet50'
    args['model_weights'] = r'..\databases\experiments\resnet50_mos_attention_fpn\119_0.0005_0.0990.h5'
    args['initial_epoch'] = 0

    args['lr_base'] = 2e-8
    args['lr_schedule'] = True
    args['batch_size'] = 4
    args['epochs'] = 120

    args['fpn_type'] = 'fpn'
    args['attention_module'] = True

    args['image_aug'] = True

    train_main(args)
