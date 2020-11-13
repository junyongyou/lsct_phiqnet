from phiqnet.train.train import train_main

if __name__ == '__main__':
    args = {}
    args['multi_gpu'] = 0
    args['gpu'] = 1

    args['result_folder'] = r'..\databases\experiments\koniq_all'
    args['n_quality_levels'] = 1

    args['train_folders'] = [r'..\databases\train\koniq_normal',
                             r'..\databases\train\koniq_small',]
                             # r'..\databases\train\live']
    args['val_folders'] = [r'..\databases\val\koniq_normal',
                           r'..\databases\val\koniq_small',]
                           # r'..\databases\val\live']
    args['koniq_mos_file'] = r'..\databases\koniq10k_images_scores.csv'
    args['live_mos_file'] = r'..\databases\live_mos.csv'

    args['naive_backbone'] = False
    args['backbone'] = 'resnet50'
    args['model_weights'] = r'..\pretrained_weights\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    args['initial_epoch'] = 0

    args['lr_base'] = 1e-4 / 2
    args['lr_schedule'] = True
    args['batch_size'] = 4
    args['epochs'] = 120

    args['fpn_type'] = 'fpn'
    args['attention_module'] = True

    args['image_aug'] = True

    train_main(args)
