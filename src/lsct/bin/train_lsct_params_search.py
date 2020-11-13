from lsct.train.train import train_main


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


    cnn_filters_range = [
        [16, 32],
        [32, 64],
        [32, 64, 128],
        [32, 64, 128, 256]
    ]
    transformer_params_range = [
        [2, 32, 4, 64],
        [2, 64, 4, 64],
        [4, 32, 4, 64],
        [4, 64, 4, 64],
        [4, 64, 4, 128],
        [4, 64, 8, 128],
        [4, 64, 8, 256],
        [4, 128, 8, 256],
        [8, 256, 8, 512]
    ]

    args['dropout_rate'] = 0.1
    args['clip_length'] = 16

    args['batch_size'] = 32

    args['lr_base'] = 1e-3
    args['epochs'] = 400

    args['multi_gpu'] = 0
    args['gpu'] = 1

    args['validation'] = 'validation'

    args['do_finetune'] = False

    for cnn_filters in cnn_filters_range:
        for transformer_params in transformer_params_range:
            print('CNN: {}, Transformer: {}'.format(cnn_filters, transformer_params))
            args['cnn_filters'] = cnn_filters
            # No need to define pooling sizes for 1D CNN, which will be defined in check_args() in train

            args['transformer_params'] = transformer_params
            args['model_name'] = 'lsct_{}_{}'.format(cnn_filters, transformer_params)
            train_main(args)
