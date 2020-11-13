import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from phiqnet.models.image_quality_model import phiq_net
from callbacks.callbacks import create_callbacks
from phiqnet.train.plot_train import plot_history
from phiqnet.utils.imageset_handler import get_image_scores, get_image_score_from_groups
from phiqnet.train.group_generator import GroupGenerator
from callbacks.evaluation_callback_generator import ModelEvaluationIQGenerator
from callbacks.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler


def check_args(args):
    if 'result_folder' not in args:
        exit('Result folder must be specified')
    if 'train_folders' not in args:
        exit('Train folders must be specified')
    if 'val_folders' not in args:
        args['val_folders'] = None
        print('Warning, check validation folders specified')
    if 'koniq_mos_file' not in args:
        exit('KonIQ MOS file must be specified')
    if 'live_mos_file' not in args:
        exit('LIVE-wild MOS file must be specified')

    if 'n_quality_levels' not in args:
        exit('Number of quality levels (1 or 5) must be specified')

    if 'epochs' not in args:
        args['epochs'] = 120
    if 'lr_base' not in args:
        args['lr_base'] = 1e-4 / 2
    if 'naive_backbone' not in args:
        args['naive_backbone'] = False
    if 'backbone' not in args:
        args['backbone'] = 'resnet50'
    if 'model_weights' not in args:
        args['model_weights'] = None
    if 'initial_epoch' not in args:
        args['initial_epoch'] = 0
    if 'fpn_type' not in args:
        args['fpn_type'] = 'fpn'
    if 'attention_module' not in args:
        args['attention_module'] = True
    if 'freeze_backbone' not in args:
        args['freeze_backbone'] = False
    if 'lr_schedule' not in args:
        args['lr_schedule'] = True
    if 'batch_size' not in args:
        args['batch_size'] = 4
    if 'image_aug' not in args:
        args['image_aug'] = True
    if 'multi_gpu' not in args:
        args['multi_gpu'] = 0
    if 'gpu' not in args:
        args['gpu'] = 0

    return args


def train_main(args):
    args = check_args(args)

    if args['multi_gpu'] == 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args['gpu']], 'GPU')

    result_folder = args['result_folder']
    model_name = args['backbone']

    # Define loss function according to prediction objective (score distribution or MOS)
    if args['n_quality_levels'] > 1:
        using_single_mos = False
        loss = 'categorical_crossentropy'
        metrics = None
        model_name += '_distribution'
    else:
        using_single_mos = True
        # metrics = 'mae'
        metrics = None
        loss = 'mse'
        model_name += '_mos'

    if args['attention_module']:
        model_name += '_attention'

    model_name += '_naive' if args['naive_backbone'] else '_' + args['fpn_type']

    if args['model_weights'] is None:
        model_name += '_nopretrain'
    if args['lr_base'] < 1e-4 / 2:
        model_name += '_finetune'
    if not args['image_aug']:
        model_name += '_no_imageaug'

    # Create PHIQNet model
    optimizer = Adam(args['lr_base'])
    if args['multi_gpu'] > 0:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            # Everything that creates variables should be under the strategy scope.
            # In general this is only model construction & `compile()`.
            model = phiq_net(n_quality_levels=args['n_quality_levels'],
                             naive_backbone=args['naive_backbone'],
                             backbone=args['backbone'],
                             fpn_type=args['fpn_type'],
                             attention_module=args['attention_module'])
            model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    else:
        model = phiq_net(n_quality_levels=args['n_quality_levels'],
                         naive_backbone=args['naive_backbone'],
                         backbone=args['backbone'],
                         fpn_type=args['fpn_type'],
                         attention_module=args['attention_module'])
        model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    # Load Imagenet pretrained model_weights or existing model_weights for fine-tune
    if args['model_weights'] is not None:
        print('Load model_weights: {}'.format(args['model_weights']))
        if args['lr_base'] < 1e-4 / 2:
            model.load_weights(args['model_weights'])
        else:
            model.load_weights(args['model_weights'], by_name=True)

    # Freeze backbone layers for training
    if args['freeze_backbone']:
        model_name += '_freeze_backbone'
        for i, layer in enumerate(model.layers):
            if i < 175:
                layer.trainable = False

    if args['model_weights'] is not None and ('resnet' in args['backbone'] or args['backbone'] == 'inception'):
        imagenet_pretrain = True
    else:
        imagenet_pretrain = False

    # Define train and validation data
    image_scores = get_image_scores(args['koniq_mos_file'], args['live_mos_file'], using_single_mos=using_single_mos)
    train_image_file_groups, train_score_groups = get_image_score_from_groups(args['train_folders'], image_scores)
    train_generator = GroupGenerator(train_image_file_groups,
                                     train_score_groups,
                                     batch_size=args['batch_size'],
                                     image_aug=args['image_aug'],
                                     imagenet_pretrain=imagenet_pretrain)
    train_steps = train_generator.__len__()

    if args['val_folders'] is not None:
        test_image_file_groups, test_score_groups = get_image_score_from_groups(args['val_folders'], image_scores)
        validation_generator = GroupGenerator(test_image_file_groups,
                                              test_score_groups,
                                              batch_size=args['batch_size'],
                                              image_aug=False,
                                              imagenet_pretrain=imagenet_pretrain)
        validation_steps = validation_generator.__len__()

        # evaluation_generator = GroupGenerator(test_image_file_groups,
        #                                       test_score_groups,
        #                                       batch_size=1,
        #                                       image_aug=False,
        #                                       imagenet_pretrain=imagenet_pretrain)
        evaluation_generator = None

        evaluation_callback = ModelEvaluationIQGenerator(validation_generator,
                                                         using_single_mos,
                                                         evaluation_generator=evaluation_generator,
                                                         imagenet_pretrain=imagenet_pretrain)
    else:
        evaluation_callback = None
        validation_generator = None
        validation_steps = 0

    result_folder = os.path.join(result_folder, model_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Create callbacks including evaluation and learning rate scheduler
    callbacks = create_callbacks(model_name,
                                 result_folder,
                                 evaluation_callback,
                                 checkpoint=True,
                                 early_stop=False,
                                 metrics=metrics)

    warmup_epochs = 10
    if args['lr_schedule']:
        total_train_steps = args['epochs'] * train_steps
        warmup_steps = warmup_epochs * train_steps
        warmup_lr = WarmUpCosineDecayScheduler(learning_rate_base=args['lr_base'],
                                               total_steps=total_train_steps,
                                               warmup_learning_rate=0.0,
                                               warmup_steps=warmup_steps,
                                               hold_base_rate_steps=30 * train_steps,
                                               verbose=1)
        callbacks.append(warmup_lr)

    # Train
    model_history = model.fit(x=train_generator,
                              epochs=args['epochs'],
                              steps_per_epoch=train_steps,
                              validation_data=validation_generator,
                              validation_steps=validation_steps,
                              verbose=1,
                              shuffle=False,
                              callbacks=callbacks,
                              initial_epoch=args['initial_epoch'],
                              )
    # model.save(os.path.join(result_folder, model_name + '.h5'))
    plot_history(model_history, result_folder, model_name)
