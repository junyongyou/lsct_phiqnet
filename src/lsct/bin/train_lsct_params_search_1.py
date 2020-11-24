from lsct.train.train import train_main
import numpy as np
import os
from lsct.utils.gather_video_ids import gather_all_vids

"""
Search for best hyper-parameters of LSCT-PHIQNet
"""
if __name__ == '__main__':
    args = {}
    args['result_folder'] = r'C:\vq_datasets\results\tmp'
    args['vids_meta'] = r'..\\meta_data\all_vids.pkl'
    args['meta_file'] = r'..\\meta_data\all_video_mos.csv'

    # if ugc_chunk_pickle is used, then the folders containing PHIQNet features of UGC chunks must be specified
    args['ugc_chunk_pickle'] = None  # r'..\\meta_data\ugc_chunks.pkl'
    args['ugc_chunk_folder'] = r'.\frame_features\ugc_chunks'
    args['ugc_chunk_folder_flipped'] = r'.\frame_features_flipped\ugc_chunks'

    # args['database'] = ['live', 'konvid', 'ugc']
    args['database'] = ['konvid']

    cnn_filters_range = [
        [16, 32],
        # [32, 64],
        # [32, 64, 128],
        # [32, 64, 128, 256]
    ]
    transformer_params_range = [
        [2, 16, 2, 32],
        # [2, 16, 4, 32],
        # [2, 32, 4, 64],
        # [2, 64, 4, 64],
        # [4, 32, 4, 64],
        # [4, 64, 4, 64],
        # [4, 64, 4, 128],
        # [4, 64, 8, 128],
        # [4, 64, 8, 256],
        # [4, 128, 8, 256],
        # [8, 256, 8, 512]
    ]

    args['dropout_rate'] = 0.1
    args['clip_length'] = 16

    args['batch_size'] = 32

    args['lr_base'] = 1e-3/2
    args['epochs'] = 300

    args['multi_gpu'] = 0
    args['gpu'] = 0

    args['validation'] = 'validation'

    args['do_finetune'] = True

    result_record_file = os.path.join(args['result_folder'], 'konvid_nochunks.csv')
    runs = 5
    all_plcc = np.zeros((runs, len(cnn_filters_range), len(transformer_params_range)))

    for k in range(runs):
        train_vids, test_vids = gather_all_vids(all_vids_pkl=args['vids_meta'])

        for i, cnn_filters in enumerate(cnn_filters_range):
            for j, transformer_params in enumerate(transformer_params_range):
                if not os.path.exists(result_record_file):
                    record_file = open(result_record_file, 'w+')
                else:
                    record_file = open(result_record_file, 'a')

                args['cnn_filters'] = cnn_filters
                # No need to define pooling sizes for 1D CNN, which will be defined in check_args() in train

                args['transformer_params'] = transformer_params
                args['model_name'] = 'lsct_{}_{}'.format(cnn_filters, transformer_params)

                plcc = train_main(args, train_vids, test_vids)

                record_file.write('Run: {}, CNN: {}, Transformer: {}, plcc: {}\n'.format(k, cnn_filters, transformer_params, plcc))

                all_plcc[k, i, j] = plcc
                print('Run: {}, CNN: {}, Transformer: {}, plcc: {}\n'.format(k, cnn_filters, transformer_params, plcc))
                record_file.flush()
                record_file.close()
        print(np.mean(np.array(all_plcc), axis=0))
