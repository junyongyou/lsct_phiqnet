from lsct.utils.gather_video_ids import gather_all_vids
from pickle import load, dump


vids = r'C:\lsct_phiqnet\src\lsct\meta_data\all_vids.pkl'
for i in range(10):
    train_vids, test_vids = gather_all_vids(all_vids_pkl=vids)
    dump([train_vids, test_vids], open(r'C:\vq_datasets\random_splits\split_{}.pkl'.format(i), 'wb'))