import pandas as pd
import numpy as np
import os
from ffmpeg.video_handler import VideoHandler
from pickle import load, dump


def get_video_handler():
    ffmpeg_exe = r'..\\ffmpeg\ffmpeg.exe'
    ffprobe_exe = r'..\\ffmpeg\ffprobe.exe'
    video_handler = VideoHandler(ffprobe_exe, ffmpeg_exe)
    return video_handler


def get_video_path(vid):
    # Video folders of YouTube-UGC database must be specified
    video_folders = [
        r'.\ugc_test',
        r'.\ugc_train',
        r'.\ugc_validation'
    ]
    for video_folder in video_folders:
        if os.path.exists(os.path.join(video_folder, vid + '.mkv')):
            return os.path.join(video_folder, vid + '.mkv')

    return None


def get_chunk_mos_Resnet():
    """
        Extract frame features from Resnet50 of individual chunks and stored in Numpy npy files
        :return: Dictionary containing video id, full MOS, chunk1 MOS, chunk2 MOS, chunk3 MOS
        """
    chunk_mos_dict = dict()
    ugc_mos_file = r'C:\vq_datasets\ugc_mos_original.xlsx'
    ugc_mos = pd.read_excel(ugc_mos_file)

    frame_feature_folder = r'C:\vq_datasets\VSFA\UGC'
    chunk_feature_folder = r'C:\vq_datasets\VSFA\UGC_CHUNKS'

    video_handler = get_video_handler()

    for index, row in ugc_mos.iterrows():
        vid = row['vid']
        video_path = get_video_path(vid)

        if video_path:
            video_meta = video_handler.get_video_meta(video_path)
            fps = round(video_meta[-2])
            mos_chunk_0 = row['MOS chunk00']
            mos_chunk_1 = row['MOS chunk05']
            mos_chunk_2 = row['MOS chunk10']

            chunk_mos = []
            chunk_mos.append(row['MOS full'])

            frame_features = np.load(os.path.join(frame_feature_folder, vid + '_resnet-50_res5c.npy'))
            if not np.isnan(mos_chunk_0):
                chunk_mos.append(mos_chunk_0)
                frame_features_chunk_0 = frame_features[0 : 10 * fps, :]
                np.save(os.path.join(chunk_feature_folder, vid + '_resnet-50_res5c_chunk_0.npy'), frame_features_chunk_0)

            if not np.isnan(mos_chunk_1):
                chunk_mos.append(mos_chunk_1)
                frame_features_chunk_1 = frame_features[5 * fps: 15 * fps, :]
                np.save(os.path.join(chunk_feature_folder, vid + '_resnet-50_res5c_chunk_1.npy'), frame_features_chunk_1)

            if not np.isnan(mos_chunk_2):
                chunk_mos.append(mos_chunk_2)
                frame_features_chunk_2 = frame_features[10 * fps:, :]
                np.save(os.path.join(chunk_feature_folder, vid + '_resnet-50_res5c_chunk_2.npy'), frame_features_chunk_2)

            chunk_mos_dict[vid] = chunk_mos

    return chunk_mos_dict


def get_chunk_features_mos():
    """
    Extract frame features of individual chunks and stored in Numpy npy files
    :return: Dictionary containing video id, full MOS, chunk1 MOS, chunk2 MOS, chunk3 MOS
    """
    chunk_mos_dict = dict()
    ugc_mos_file = r'..\\meta_data\ugc_mos_original.xlsx'
    ugc_mos = pd.read_excel(ugc_mos_file)

    # Frame feature files of YouTube-UGC videos must be specified
    frame_feature_folder = r'.\frame_features\ugc'

    # Target folder to store the frame features of chunks
    chunk_feature_folder = r'.\frame_features\ugc_chunks'

    video_handler = get_video_handler()

    for index, row in ugc_mos.iterrows():
        vid = row['vid']
        video_path = get_video_path(vid)

        if video_path:
            mos_chunk_0 = row['MOS chunk00']
            mos_chunk_1 = row['MOS chunk05']
            mos_chunk_2 = row['MOS chunk10']

            video_meta = video_handler.get_video_meta(video_path)
            fps = round(video_meta[-2])

            chunk_mos = []
            chunk_mos.append(row['MOS full'])

            frame_features = np.load(os.path.join(frame_feature_folder, vid + '.npy'))
            if not np.isnan(mos_chunk_0):
                chunk_mos.append(mos_chunk_0)
                frame_features_chunk_0 = frame_features[0 : 10 * fps, :, :, :]
                np.save(os.path.join(chunk_feature_folder, vid + '_chunk_0.npy'), frame_features_chunk_0)

            if not np.isnan(mos_chunk_1):
                chunk_mos.append(mos_chunk_1)
                frame_features_chunk_1 = frame_features[5 * fps: 15 * fps, :, :, :]
                np.save(os.path.join(chunk_feature_folder, vid + '_chunk_1.npy'), frame_features_chunk_1)

            if not np.isnan(mos_chunk_2):
                chunk_mos.append(mos_chunk_2)
                frame_features_chunk_2 = frame_features[10 * fps:, :, :, :]
                np.save(os.path.join(chunk_feature_folder, vid + '_chunk_2.npy'), frame_features_chunk_2)

            chunk_mos_dict[vid] = chunk_mos

    return chunk_mos_dict


if __name__ == '__main__':
    chunk_mos_dict = get_chunk_features_mos()
    chunk_mos_dict_resnet50s = get_chunk_mos_Resnet()

    # The chunk MOS values can be dumped
    # dump(chunk_mos_dict, open(r'..\\meta_data\ugc_chunks.pkl', 'wb'))
