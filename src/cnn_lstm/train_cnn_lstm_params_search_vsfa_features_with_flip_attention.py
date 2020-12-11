"""
Run the training of CNN-LSTM-PHIQNet, i.e., using LSTM to replace Transformer in LSCT-PHIQNet.
"""
import os
import tensorflow as tf
import numpy as np
import glob
from pickle import load
from tensorflow.keras.optimizers import Adam, SGD

from callbacks.callbacks import create_callbacks
from callbacks.warmup_cosine_decay_scheduler import WarmUpCosineDecayScheduler
from lsct.train.video_clip_feature_generator_vsfa import VideoClipFeatureGenerator
from lsct.utils.gather_video_ids import gather_all_vids
from callbacks.evaluation_vq_generator import ModelEvaluationGeneratorVQ
from lsct.models.cnn_lstm_model import create_cnn_lstm_model


def check_args(args):
    if 'result_folder' not in args:
        exit('Result folder must be specified')
    if 'meta_file' not in args:
        exit('Meta file of videos and MOS must be specified')
    if 'vids_meta' not in args:
        args['vids_meta'] = None

    if 'model_name' not in args:
        args['model_name'] = 'cnn_lstm'
    if 'ugc_chunk_pickle' not in args or 'ugc_chunk_folder' not in args or '' not in args:
        args['ugc_chunk_pickle'] = None
        args['ugc_chunk_folder'] = None
        args['ugc_chunk_folder_flipped'] = None
    if 'database' not in args:
        args['database'] = ['live', 'konvid', 'ugc']
    if 'validation' not in args:
        args['validation'] = 'validation'

    if 'cnn_filters' not in args:
        args['cnn_filters'] = [32, 64]
    if 'pooling_sizes' not in args:
        if len(args['cnn_filters']) == 2:
            args['pooling_sizes'] = [4, 4]
        elif len(args['cnn_filters']) == 3:
            args['pooling_sizes'] = [4, 2, 2]
        else:
            args['pooling_sizes'] = [2, 2, 2, 2]
    else:
        if len(args['cnn_filters']) != len(args['pooling_sizes']):
            print('WARN: Filters and pooling sizes in 1D CNN must be match, pooling sizes changed based on filters')
            if len(args['cnn_filters']) == 2:
                args['pooling_sizes'] = [4, 4]
            elif len(args['cnn_filters']) == 3:
                args['pooling_sizes'] = [4, 2, 2]
            else:
                args['pooling_sizes'] = [2, 2, 2, 2]

    if 'clip_length' not in args:
        args['clip_length'] = 16

    if 'lstm_filters' not in args:
        args['lstm_filters'] = [32, 64]
    if 'mlp_filters' not in args:
        args['mlp_filters'] = [64, 32, 8]

    if 'epochs' not in args:
        args['epochs'] = 400
    if 'lr_base' not in args:
        args['lr_base'] = 1e-3
    if 'batch_size' not in args:
        args['batch_size'] = 32
    if 'lr_schedule' not in args:
        args['lr_schedule'] = True
    if 'multi_gpu' not in args:
        args['multi_gpu'] = 0
    if 'gpu' not in args:
        args['gpu'] = 0

    if 'do_finetune' not in args:
        args['do_finetune'] = True

    return args


def identify_best_weights(result_folder, history, pos):
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


def train_main(args, train_vids=None, test_vids=None):
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

    if train_vids == None or test_vids == None:
        # train and test videos will be randomly split based on random seed
        train_vids, test_vids = gather_all_vids(all_vids_pkl=args['vids_meta'])

    clip_length = args['clip_length']
    model_name += '_clip_{}'.format(clip_length)

    epochs = args['epochs']

    feature_length = 4096

    if args['multi_gpu'] == 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[args['gpu']], 'GPU')
        tf.config.experimental_run_functions_eagerly(True)
        model = create_cnn_lstm_model(clip_length=args['clip_length'],
                                      feature_length=feature_length,
                                      cnn_filters=args['cnn_filters'],
                                      lstm_filters=args['lstm_filters'],
                                      mlp_filters=args['mlp_filters'],
                                      using_attention=True)
    else:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        with strategy.scope():
            model = create_cnn_lstm_model(clip_length=args['clip_length'],
                                          feature_length=feature_length,
                                          cnn_filters=args['cnn_filters'],
                                          lstm_filters=args['lstm_filters'],
                                          mlp_filters=args['mlp_filters'],
                                          using_attention=True)

    optimizer = Adam(args['lr_base'])
    loss = 'mse'
    metrics = 'mae'
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])

    if args['validation'] == 'validation':
        train_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                    train_vids,
                                                    batch_size=args['batch_size'],
                                                    clip_length=args['clip_length'],
                                                    random_ratio=0.25,
                                                    flip=True,
                                                    training=True,
                                                    ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                    ugc_chunk_folder=args['ugc_chunk_folder'],
                                                    ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                    database=args['database'])
        test_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                   test_vids,
                                                   batch_size=args['batch_size'],
                                                   clip_length=args['clip_length'],
                                                   random_ratio=0.25,
                                                   flip=False,
                                                   training=False,
                                                   ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                   ugc_chunk_folder=args['ugc_chunk_folder'],
                                                   ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                   database=args['database'])
    else:
        all_vids = train_vids + test_vids
        train_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                    all_vids,
                                                    batch_size=args['batch_size'],
                                                    clip_length=args['clip_length'],
                                                    random_ratio=0.25,
                                                    flip=True,
                                                    training=True,
                                                    ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                    ugc_chunk_folder=args['ugc_chunk_folder'],
                                                    ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                    database=['konvid', 'ugc'])
        test_generator = VideoClipFeatureGenerator(args['meta_file'],
                                                   all_vids,
                                                   batch_size=args['batch_size'],
                                                   clip_length=args['clip_length'],
                                                   random_ratio=0.25,
                                                   flip=False,
                                                   training=False,
                                                   ugc_chunk_pickle=args['ugc_chunk_pickle'],
                                                   ugc_chunk_folder=args['ugc_chunk_folder'],
                                                   ugc_chunk_folder_flipped=args['ugc_chunk_folder_flipped'],
                                                   database=['live'])

    evaluation_callback = ModelEvaluationGeneratorVQ(test_generator, None)
    callbacks = create_callbacks(model_name,
                                 result_folder,
                                 evaluation_callback,
                                 checkpoint=True,
                                 early_stop=True,
                                 metrics=metrics)

    train_steps = train_generator.__len__()
    total_train_steps = epochs * train_steps
    warmup_epochs = 10
    warmup_steps = warmup_epochs * train_steps
    if args['lr_schedule']:
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

    # best_plcc = callbacks[3].best
    pos = np.where(model_history.history['plcc'] == max_plcc_pretrain)[0][0]
    rmse = model_history.history['rmse'][pos]
    srcc = model_history.history['srcc'][pos]
    info = 'Pretrain: epochs: {}, MAX PLCC: {}, RMSE: {}, SROCC: {}\n'.format(len(model_history.history['plcc']), max_plcc_pretrain, rmse, srcc)
    print(info)

    best_weights_file = identify_best_weights(result_folder, model_history.history, pos)
    remove_non_best_weights(result_folder, [best_weights_file])

    if not best_weights_file:
        return max_plcc_pretrain, rmse, srcc

    # do fine-tuning
    if args['do_finetune'] and best_weights_file:
        del (callbacks[-1])
        model.load_weights(best_weights_file)
        finetune_lr = 1e-4/2
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

        # best_plcc = callbacks[3].best
        pos_finetune = np.where(finetune_model_history.history['plcc'] == max_plcc_finetune)[0][0]
        rmse_finetune = finetune_model_history.history['rmse'][pos_finetune]
        srcc_finetune = finetune_model_history.history['srcc'][pos_finetune]

        info = 'Finetune: epochs: {}, MAX PLCC: {}\n'.format(len(finetune_model_history.history['plcc']),
                                                             max_plcc_finetune)
        print(info)

        if max_plcc_finetune > max_plcc_pretrain:
            return max_plcc_finetune, rmse_finetune, srcc_finetune

    return max_plcc_pretrain, rmse, srcc


if __name__ == '__main__':
    args = {}
    args['result_folder'] = r'C:\vq_datasets\results\cnn_lstm_resnet_vsfa_with_flip_attention'
    if not os.path.exists(args['result_folder']):
        os.makedirs(args['result_folder'])
    args['vids_meta'] = r'C:\lsct_phiqnet\src\lsct\meta_data\all_vids.pkl'
    args['meta_file'] = r'C:\lsct_phiqnet\src\lsct\meta_data\all_video_mos_Resnet50_vsfa.csv'

    # if ugc_chunk_pickle is used, then the folders containing PHIQNet features of UGC chunks must be specified
    args['ugc_chunk_pickle'] = None  # r'..\\meta_data\ugc_chunks.pkl'
    args['ugc_chunk_folder'] = r'.\frame_features\ugc_chunks'
    args['ugc_chunk_folder_flipped'] = r'.\frame_features_flipped\ugc_chunks'

    args['database'] = ['live', 'konvid', 'ugc']
    # args['database'] = ['konvid']

    cnn_filters_range = [
        # [16, 32],
        [32, 64],
        # [32, 64, 128],
        [32, 64, 128, 256]
    ]
    lstm_params_range = [
        # [16, 16],
        # [16, 32],
        [32, 32],
        # [32, 64],
        [16, 32, 64]
    ]
    mlp_params_range = [
        # [16, 8],
        [32, 16],
        [64, 32, 8]
    ]

    args['batch_size'] = 64

    args['lr_base'] = 1e-3
    args['epochs'] = 300

    args['multi_gpu'] = 0
    args['gpu'] = 1

    args['validation'] = 'validation'

    args['do_finetune'] = True

    result_record_file = os.path.join(args['result_folder'], 'all_nochunks_with_flip.csv')
    runs = 4
    all_plcc = np.zeros((runs, len(cnn_filters_range), len(lstm_params_range), len(mlp_params_range)))

    for m in range(runs):
        # train_vids, test_vids = gather_all_vids(all_vids_pkl=args['vids_meta'])
        train_vids, test_vids = load(open(r'C:\vq_datasets\random_splits\split_{}.pkl'.format(m + 2), 'rb'))

        for i, cnn_filters in enumerate(cnn_filters_range):
            for j, lstm_params in enumerate(lstm_params_range):
                for k, mlp_params in enumerate(mlp_params_range):
                    # if m == 0 and i == 0 and j == 0:
                    #     break
                    if not os.path.exists(result_record_file):
                        record_file = open(result_record_file, 'w+')
                    else:
                        record_file = open(result_record_file, 'a')

                    args['cnn_filters'] = cnn_filters
                    # No need to define pooling sizes for 1D CNN, which will be defined in check_args() in train

                    args['lstm_filters'] = lstm_params
                    args['mlp_filters'] = mlp_params
                    args['model_name'] = 'cnn_{}_lstm_{}_mlp_{}'.format(cnn_filters, lstm_params, mlp_params)

                    plcc, rmse, srocc = train_main(args, train_vids, test_vids)

                    record_file.write(
                        'Run: {}, CNN: {}, LSTM: {}, MLP: {}, plcc: {}, rmse: {}, srocc: {}\n'.format(
                            m + 2, cnn_filters, lstm_params, mlp_params, plcc, rmse, srocc))

                    all_plcc[m, i, j, k] = plcc
                    record_file.flush()
                    record_file.close()
        print(np.mean(np.array(all_plcc), axis=0))


