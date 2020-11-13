"""
This script is to collect all video IDs and possibly to dump them
a video ID is: database name_video name
"""
import os
import glob
from pickle import load, dump
from sklearn.model_selection import train_test_split
from random import shuffle


def gather_live_konvid_vids(video_folder, database):
    """
    LIVE-VQC and KonViD-1k video IDs
    :param video_folder:
    :param database:
    :return:
    """
    vids = []
    for file in glob.glob(os.path.join(video_folder, '*.mp4')):
        vid = os.path.splitext(os.path.basename(file))[0]
        vids.append('{}_{}'.format(database, vid))
    return vids


def gather_ugc_vids(video_folders):
    """
    YouTube-UGC video IDs
    :param video_folders: list of folders of YouTube-UGC video
    :return: video IDs
    """
    ugc_vids = []
    for video_folder in video_folders:
        files = glob.glob(os.path.join(video_folder, '*.mkv'))
        for file in files:
            vid = os.path.splitext(os.path.basename(file))[0]
            ugc_vids.append('ugc_{}'.format(vid))
    return ugc_vids


def gather_all_vids(all_vids_pkl=None, test_ratio=0.2, random_state=None):
    if all_vids_pkl:
        all_vids = load(open(all_vids_pkl, 'rb'))
    else:
        live_vids = gather_live_konvid_vids(r'.\live_vqc_Video', 'live')
        konvid_vids = gather_live_konvid_vids(r'.\KoNViD_1k_videos', 'konvid')
        ugc_vids = gather_ugc_vids([r'.\ugc_test', r'.\ugc_train', r'.\ugc_validation'])
        all_vids = live_vids + konvid_vids + ugc_vids

        # the video IDs can be dumped here, for later use in training
        dump(all_vids, open(r'.\all_vids.pkl', 'wb'))
    shuffle(all_vids)
    train_vids, test_vids = train_test_split(all_vids, test_size=test_ratio, random_state=random_state)
    return train_vids, test_vids


if __name__ == '__main__':
    # live_video_folder = r'.\live_vqc_Video'
    # konvid_video_folder = r'.\KoNViD_1k_videos'
    # live_vids, live_fps = gather_live_konvid_vids(live_video_folder, 'live')
    # konvid_vids, konvid_fps = gather_live_konvid_vids(konvid_video_folder, 'konvid')
    #
    # ugc_video_folders = [r'.\ugc_test', r'.\ugc_train', r'.\ugc_validation']
    # gather_ugc_vids(ugc_video_folders)

    info = load(open(r'..\\meta_data\ugc_chunks.pkl', 'rb'))
    t = 0
