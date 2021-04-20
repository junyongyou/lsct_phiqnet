from lsct.train.train import train_main
from pickle import load


"""
Run the training of LSCT-PHIQNet for 10 times with randomly split train and test sets
"""
if __name__ == '__main__':
    args = {}
    args['result_folder'] = r'C:\vq_datasets\results\lsct_triq_features'
    # args['vids_meta'] = r'..\\meta_data\all_vids.pkl'
    args['meta_file'] = r'C:\lsct_phiqnet\src\lsct\meta_data\all_video_mos_triq.csv'

    # if ugc_chunk_pickle is used, then the folders containing PHIQNet features of UGC chunks must be specified
    args['ugc_chunk_pickle'] = None  # r'..\\meta_data\ugc_chunks.pkl'
    # args['ugc_chunk_folder'] = r'.\frame_features\ugc_chunks'
    # args['ugc_chunk_folder_flipped'] = r'.\frame_features_flipped\ugc_chunks'

    args['database'] = ['live', 'konvid', 'ugc']

    args['model_name'] = 'lsct_triq'

    args['transformer_params'] = [2, 32, 8, 64]
    args['dropout_rate'] = 0.1
    args['cnn_filters'] = [32, 64]
    args['pooling_sizes'] = [4, 4]
    args['clip_length'] = 16

    args['batch_size'] = 32

    args['lr_base'] = 1e-3
    args['epochs'] = 200

    args['multi_gpu'] = 1
    args['gpu'] = 1

    args['validation'] = 'validation'

    args['do_finetune'] = True

    for m in range(10):
        train_vids, test_vids = load(open(r'C:\vq_datasets\random_splits\split_{}.pkl'.format(m), 'rb'))
        train_main(args, train_vids, test_vids)
