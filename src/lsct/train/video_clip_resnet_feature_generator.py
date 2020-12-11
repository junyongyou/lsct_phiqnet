"""Generate data of features in all clips"""
from tensorflow.keras.utils import Sequence
import numpy as np
import os
from pickle import load
from tensorflow.keras.preprocessing.sequence import pad_sequences

from lsct.utils.gather_video_ids import gather_all_vids


class VideoClipResnetFeatureGenerator(Sequence):
    def __init__(self, video_mos, vids, batch_size, shuffle=True, clip_length=16, padding='post', ugc_chunk_pickle=None,
                 ugc_chunk_folder=None, ugc_chunk_folder_flipped=None, random_ratio=0., clip_or_frame='clip',
                 training=True, database=('live', 'konvid', 'ugc')):
        """
        Clip feature generator
        :param video_mos: meta file contains video path and MOS
        :param vids: specify video IDs in the generator, e.g., IDs for train set or test set
        :param batch_size: batch size
        :param shuffle: flag to shuffle or not
        :param clip_length: clip length
        :param padding: padding in the beginning ('pre') or end ('post')
        :param ugc_chunk_pickle: dump pickle file containing information about YouTube-UGC chunks
        :param ugc_chunk_folder: folder containing the feature files of YouTube-UGC chunks
        :param ugc_chunk_folder_flipped: folder containing the feature files of flipped frames in YouTube-UGC chunks
        :param random_ratio: ratio defining how many videos in a batch will be reversed
        :param clip_or_frame: choose 'clip' or 'frame' to decide features generated in clip unit or frame
        :param training: flag of training or testing
        :param database: which databases should be used
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ugc_chunk_pickle = ugc_chunk_pickle
        if ugc_chunk_pickle:
            self.ugc_chunk_info = load(open(ugc_chunk_pickle, 'rb'))
            if not ugc_chunk_folder or not ugc_chunk_folder_flipped:
                print('Error, please specify UGC chunk folder')
            else:
                self.ugc_chunk_folder = ugc_chunk_folder
                self.ugc_chunk_folder_flipped = ugc_chunk_folder_flipped
        self.clip_length = clip_length
        self.padding = padding
        self.random_ratio = random_ratio
        self.clip_or_frame = clip_or_frame
        self.training = training
        self.database = database
        self.get_feature_scores(video_mos, vids)
        self.on_epoch_end()
        self.mask_value = 0.

    def get_vid(self, video_file):
        vid = os.path.splitext(os.path.basename(video_file))[0]
        if 'live_vqc' in video_file.lower():
            return 'live_{}'.format(vid), 'live'
        if 'konvid' in video_file.lower():
            return 'konvid_{}'.format(vid), 'konvid'
        if 'ugc' in video_file.lower():
            return 'ugc_{}'.format(vid), 'ugc'

    def get_feature_scores(self, video_mos, vids):
        """
        Read in all feature files and scores
        :param video_mos: meta file contains video path and MOS
        :param vids: specify video IDs in the generator, e.g., IDs for train set or test set
        :return:
        """
        self.features_files = []
        self.scores = []

        with open(video_mos) as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(',')
                vid, dataset = self.get_vid(content[0])

                flag = True if dataset in self.database else False
                if flag:
                    if vid in vids:
                        self.features_files.append(content[0])
                        self.scores.append(float(content[1]))
                        if self.training:
                            self.features_files.append(content[0].replace('frame_features', 'frame_features_flipped'))
                            self.scores.append(float(content[1]))

                        if self.ugc_chunk_pickle:
                            if 'ugc' in vid:
                                ugc_vid = vid.replace('ugc_', '')
                                ugc_chunk = self.ugc_chunk_info[ugc_vid]
                                for i in range(len(ugc_chunk) - 2):
                                    self.features_files.append(
                                        os.path.join(self.ugc_chunk_folder, '{}_chunk_{}.npy'.format(ugc_vid, i)))
                                    self.scores.append(ugc_chunk[i + 2])
                                    if self.training:
                                        self.features_files.append(
                                            os.path.join(self.ugc_chunk_folder_flipped,
                                                         '{}_chunk_{}.npy'.format(ugc_vid, i)))
                                        self.scores.append(ugc_chunk[i + 2])

    def __len__(self):
        return len(self.scores) // self.batch_size

    def on_epoch_end(self):
        self.indices = np.arange(len(self.scores))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, item):
        indices_batch = self.indices[item * self.batch_size: (item + 1) * self.batch_size]
        frame_features_data = []
        y_scores = []
        if self.training:
            random_choice = np.random.choice(indices_batch, int(self.random_ratio * self.batch_size))
        for index in indices_batch:
            frame_features = np.load(self.features_files[index])
            frame_features = np.squeeze(frame_features, axis=1)

            # Reverse frames
            if self.training and index in random_choice:
                frame_features = frame_features[::-1, :, :]

            frame_features = np.reshape(frame_features,
                                        (frame_features.shape[0], frame_features.shape[1] * frame_features.shape[2]))

            frame_features_data.append(frame_features)
            y_scores.append(self.scores[index])

        frame_features_data = pad_sequences(frame_features_data, value=self.mask_value, dtype=np.float32,
                                            padding=self.padding)

        if self.clip_or_frame == 'frame':
            return np.array(frame_features_data), np.array(y_scores)

        clip_features_data = []
        for i in range(frame_features_data.shape[0]):
            clip_features = []
            for j in range(frame_features_data.shape[1] // self.clip_length):
                clip_features.append(frame_features_data[i, j * self.clip_length: (j + 1) * self.clip_length, :])
            clip_features_data.append(np.array(clip_features))

        return np.array(clip_features_data), np.array(y_scores)


if __name__ == '__main__':
    # video_mos_file = r'..\\examples\meta_data\all_video_mos_with_flipped.csv'
    video_mos_file = r'..\\examples\meta_data\all_video_mos.csv'
    ugc_chunk_pickle = r'..\\examples\meta_data\ugc_chunks.pkl'
    # ugc_chunk_pickle = None
    ugc_chunk_folder = r'..\frame_features\ugc_chunks'
    ugc_chunk_folder_flipped = r'..\frame_features_flipped\ugc_chunks'
    include_frame_scores = False
    clip_length = 16
    batch_size = 32
    database = ['live', 'konvid', 'ugc']
    train_vids, test_vids = gather_all_vids(all_vids_pkl=r'..\\examples\meta_data\all_vids.pkl')
    train_generator = VideoClipResnetFeatureGenerator(video_mos_file,
                                                train_vids,
                                                batch_size=batch_size,
                                                clip_length=clip_length,
                                                random_ratio=0.25,
                                                training=True,
                                                ugc_chunk_pickle=ugc_chunk_pickle,
                                                ugc_chunk_folder=ugc_chunk_folder,
                                                ugc_chunk_folder_flipped=ugc_chunk_folder_flipped,
                                                database=database)
    test_generator =  VideoClipResnetFeatureGenerator(video_mos_file,
                                                test_vids,
                                                batch_size=32,
                                                random_ratio=0.,
                                                training=False,
                                                ugc_chunk_pickle=ugc_chunk_pickle,
                                                ugc_chunk_folder=ugc_chunk_folder,
                                                ugc_chunk_folder_flipped=ugc_chunk_folder_flipped,
                                                database=database)

    for i in range(train_generator.__len__()):
        X, y = train_generator.__getitem__(i)

    for i in range(test_generator.__len__()):
        X, y = test_generator.__getitem__(i)
