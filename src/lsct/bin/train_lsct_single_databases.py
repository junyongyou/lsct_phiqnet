from lsct.train.train import train_main


"""
General train script of LSCT-PHIQNet on single one or two databases
"""
if __name__ == '__main__':
    args = {}
    args['result_folder'] = r'C:\vq_datasets\results\lsct'
    args['vids_meta'] = r'..\\meta_data\all_vids.pkl'
    args['meta_file'] = r'..\\meta_data\all_video_mos.csv'

    # if ugc_chunk_pickle is used, then the folders containing PHIQNet features of UGC chunks must be specified
    args['ugc_chunk_pickle'] = None # r'..\\meta_data\ugc_chunks.pkl'
    args['ugc_chunk_folder'] = r'.\frame_features\ugc_chunks'
    args['ugc_chunk_folder_flipped'] = r'.\frame_features_flipped\ugc_chunks'

    args['database'] = ['konvid', 'ugc']
    # args['database'] = ['konvid']
    # args['database'] = ['ugc']

    args['model_name'] = 'lsct'

    args['transformer_params'] = [2, 64, 4, 64]
    args['dropout_rate'] = 0.1
    args['cnn_filters'] = [32, 64]
    # args['pooling_sizes'] = [4, 4]
    args['clip_length'] = 16

    args['batch_size'] = 32

    args['lr_base'] = 1e-3
    args['epochs'] = 400

    args['multi_gpu'] = 0
    args['gpu'] = 1

    args['validation'] = 'validation'

    args['do_finetune'] = False

    train_main(args)
