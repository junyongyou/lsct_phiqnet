from lsct.train.train import train_main


"""
Search for best clip length for LSCT-PHIQNet. 
It is noted that max pooling sizes are adaptive to the clip length.
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

    args['database'] = ['live', 'konvid', 'ugc']
    # args['database'] = ['konvid']

    args['transformer_params'] = [2, 64, 4, 64]
    args['dropout_rate'] = 0.1
    args['cnn_filters'] = [32, 64]

    clip_length_range = [8, 16, 24, 32, 64]
    pooling_sizes_range = [[4, 2],
                          [4, 4],
                          [6, 4],
                          [8, 4],
                          [8, 8]]

    args['batch_size'] = 32
    args['lr_base'] = 1e-3
    args['epochs'] = 140

    args['multi_gpu'] = 0
    args['gpu'] = 0

    args['validation'] = 'validation'

    args['do_finetune'] = False

    for clip_length, pooling_sizes in zip(clip_length_range, pooling_sizes_range):
        print('Clip length: {}'.format(clip_length))
        args['clip_length'] = clip_length
        args['pooling_sizes'] = pooling_sizes
        train_main(args)
