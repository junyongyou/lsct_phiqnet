"""
Run the training of Transformer-PHIQNet, i.e., applying Transformer on frame features without using the 1D CNN
As the model has much more parameters than LSCT-PHIQNet, the batch size must be reduced to avoid OOM error.
"""
import os
import tensorflow as tf
import numpy as np
import glob
from tensorflow.keras.optimizers import Adam, SGD

from callbacks.callbacks import create_callbacks
from callbacks.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from lsct.train.video_clip_feature_generator import VideoClipFeatureGenerator
from lsct.utils.gather_video_ids import gather_all_vids
from callbacks.evaluation_vq_generator import ModelEvaluationGeneratorVQ
from lsct.models.video_quality_transformer import VideoQualityTransformer


def check_args(args):
    if 'result_folder' not in args:
        exit('Result folder must be specified')
    if 'meta_file' not in args:
        exit('Meta file of videos and MOS must be specified')
    if 'vids_meta' not in args:
        args['vids_meta'] = None

    if 'model_name' not in args:
        args['model_name'] = 'lsct'
    if 'ugc_chunk_pickle' not in args or 'ugc_chunk_folder' not in args or '' not in args:
        args['ugc_chunk_pickle'] = None
        args['ugc_chunk_folder'] = None
        args['ugc_chunk_folder_flipped'] = None
    if 'database' not in args:
        args['database'] = ['live', 'konvid', 'ugc']
    if 'validation' not in args:
        args['validation'] = 'validation'

    if 'epochs' not in args:
        args['epochs'] = 400
    if 'lr_base' not in args:
        args['lr_base'] = 1e-3
    if 'batch_size' not in args:
        args['batch_size'] = 8
    if 'lr_schedule' not in args:
        args['lr_schedule'] = True
    if 'multi_gpu' not in args:
        args['multi_gpu'] = 0
    if 'gpu' not in args:
        args['gpu'] = 0

    if 'do_finetune' not in args:
        args['do_finetune'] = True

    return args


def identify_best_weights(result_folder, history, best_plcc):
    pos = np.where(history['plcc'] == best_plcc)[0][0]
    pos_loss = '{}_{:.4f}'.format(pos + 1, history['loss'][pos])
    all_weights_files = glob.glob(os.path.join(result_folder, '*.h5'))
    for all_weights_file in all_weights_files:
        weight_file = os.path.basename(all_weights_file)
        if weight_file.startswith(pos_loss):
            best_weights_file = all_weights_file
            return best_weights_file
    return None


def remove_non_best_weights(result_folder, best_weights_files):
    all_weights_files = glob.glob(os.path.join(result_folder, '*.h5'))
    for all_weights_file in all_weights_files:
        if all_weights_file not in best_weights_files:
            os.remove(all_weights_file)


def train_main(args):
    """
    Main function to train LSCT-PHIQNet
    :param args: arguments for training
    :return: Max PLCC from the training
    """
    args = check_args(args)
    result_folder = args['result_folder']
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model_name = args['model_name']

    # train and test videos will be randomly split based on random seed
    train_vids, test_vids = gather_all_vids(all_vids_pkl=args['vids_meta'])

    clip_length = args['clip_length']
    model_name += '_clip_{}'.format(clip_length)

    epochs = args['epochs']

    # Model parameters
    transformer_params = args['transformer_params']
    dropout_rates = args['dropout_rate']

    if args['multi_gpu'] == 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args['gpu']], 'GPU')
        tf.config.experimental_run_functions_eagerly(True)
        model = VideoQualityTransformer(
            num_layers=transformer_params[0],
            d_model=transformer_params[1],
            num_heads=transformer_params[2],
            mlp_dim=transformer_params[3],
            dropout=dropout_rates,
        )
    else:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with strategy.scope():
            model = VideoQualityTransformer(
                num_layers=transformer_params[0],
                d_model=transformer_params[1],
                num_heads=transformer_params[2],
                mlp_dim=transformer_params[3],
                dropout=dropout_rates,
            )

    optimizer = Adam(args['lr_base'])
    loss = 'mse'
    metrics = 'mae'
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    # clip_or_frame='frame' must be used in the generator
    if args['validation'] == 'validation':
        train_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                    train_vids,
                                                    batch_size=args['batch_size'],
                                                    clip_length=args['clip_length'],
                                                    random_ratio=0.25,
                                                    training=True,
                                                    ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                    ugc_chunk_folder=args['ugc_chunk_folder'],
                                                    ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                    clip_or_frame='frame',
                                                    database=args['database'])
        test_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                    test_vids,
                                                    batch_size=args['batch_size'],
                                                    clip_length=args['clip_length'],
                                                    random_ratio=0.25,
                                                    training=False,
                                                    ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                    ugc_chunk_folder=args['ugc_chunk_folder'],
                                                    ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                    clip_or_frame='frame',
                                                    database=args['database'])
    else:
        all_vids = train_vids + test_vids
        train_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                    all_vids,
                                                    batch_size=args['batch_size'],
                                                    clip_length=args['clip_length'],
                                                    random_ratio=0.25,
                                                    training=True,
                                                    ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                    ugc_chunk_folder=args['ugc_chunk_folder'],
                                                    ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                    clip_or_frame='frame',
                                                    database=['konvid', 'ugc'])
        test_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                    all_vids,
                                                    batch_size=args['batch_size'],
                                                    clip_length=args['clip_length'],
                                                    random_ratio=0.25,
                                                    training=True,
                                                    ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                    ugc_chunk_folder=args['ugc_chunk_folder'],
                                                    ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                    clip_or_frame='frame',
                                                    database=['live'])

    evaluation_callback = ModelEvaluationGeneratorVQ(test_generator, None)
    callbacks = create_callbacks(model_name,
                                 result_folder,
                                 evaluation_callback,
                                 checkpoint=True,
                                 early_stop=True,
                                 metrics=metrics)

    train_steps = train_generator.__len__()
    if args['lr_schedule']:
        warmup_epochs = 10
        total_train_steps = epochs * train_steps
        warmup_steps = warmup_epochs * train_steps
        warmup_lr = WarmUpCosineDecayScheduler(learning_rate_base=args['lr_base'],
                                               total_steps=total_train_steps,
                                               warmup_learning_rate=0.0,
                                               warmup_steps=warmup_steps,
                                               # hold_base_rate_steps=0,
                                               hold_base_rate_steps=10 * train_steps,
                                               verbose=1)
        callbacks.append(warmup_lr)

    model_history = model.fit(
        x=train_generator,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=test_generator,
        validation_steps=test_generator.__len__(),
        verbose=1,
        shuffle=False,
        callbacks=callbacks,
    )

    max_plcc_pretrain = np.max(model_history.history['plcc'])
    info = 'Pretrain: epochs: {}, MAX PLCC: {}\n'.format(len(model_history.history['plcc']), max_plcc_pretrain)
    print(info)

    best_weights_file = identify_best_weights(result_folder, model_history.history, callbacks[3].best)
    remove_non_best_weights(result_folder, [best_weights_file])

    # do fine-tuning
    if args['do_finetune'] and best_weights_file:
        del (callbacks[-1])
        model.load_weights(best_weights_file)
        finetune_lr = 1e-4
        if args['lr_schedule']:
            warmup_lr_finetune = WarmUpCosineDecayScheduler(learning_rate_base=finetune_lr,
                                                            total_steps=total_train_steps,
                                                            warmup_learning_rate=0.0,
                                                            warmup_steps=warmup_steps,
                                                            hold_base_rate_steps=10 * train_steps,
                                                            verbose=1)
            callbacks.append(warmup_lr_finetune)
        finetune_optimizer = SGD(learning_rate=finetune_lr, momentum=0.9)
        model.compile(loss=loss, optimizer=finetune_optimizer, metrics=[metrics])

        finetune_model_history = model.fit(
            x=train_generator,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_data=test_generator,
            validation_steps=test_generator.__len__(),
            verbose=1,
            shuffle=False,
            callbacks=callbacks,
        )

        max_plcc_finetune = np.max(finetune_model_history.history['plcc'])
        info = 'Finetune: epochs: {}, MAX PLCC: {}\n'.format(len(finetune_model_history.history['plcc']),
                                                             max_plcc_finetune)
        print(info)

    if args['do_finetune']:
        return max([max_plcc_pretrain, max_plcc_finetune])
    return max_plcc_pretrain


if __name__ == '__main__':
    args = {}
    args['result_folder'] = r'C:\vq_datasets\results\tmp'
    args['vids_meta'] = r'..\\meta_data\all_vids.pkl'
    args['meta_file'] = r'..\\meta_data\all_video_mos.csv'

    # if ugc_chunk_pickle is used, then the folders containing PHIQNet features of UGC chunks must be specified
    args['ugc_chunk_pickle'] = None  # r'..\\meta_data\ugc_chunks.pkl'
    args['ugc_chunk_folder'] = r'.\frame_features\ugc_chunks'
    args['ugc_chunk_folder_flipped'] = r'.\frame_features_flipped\ugc_chunks'

    args['database'] = ['live', 'konvid', 'ugc']

    transformer_params_range = [
        [2, 32, 4, 64],
        [2, 64, 4, 64],
        [4, 32, 4, 64],
        [4, 64, 4, 64],
        [4, 64, 4, 128],
        [4, 64, 8, 128],
        [4, 64, 8, 256],
        [4, 128, 8, 256]
    ]
    dropout_rate_range = [0.0, 0.1, 0.2, 0.4]

    args['dropout_rate'] = 0.1
    args['clip_length'] = 16

    args['batch_size'] = 8

    args['lr_base'] = 1e-3
    args['epochs'] = 400

    args['multi_gpu'] = 0
    args['gpu'] = 0

    args['validation'] = 'validation'

    args['do_finetune'] = True

    for dropout_rate in dropout_rate_range:
        for transformer_params in transformer_params_range:
            print('Dropout rate: {}, Transformer: {}'.format(dropout_rate, transformer_params))

            args['transformer_params'] = transformer_params
            args['dropout_rate'] = dropout_rate

            args['model_name'] = 'lsct_{}_{}'.format(dropout_rate, transformer_params)
            train_main(args)


