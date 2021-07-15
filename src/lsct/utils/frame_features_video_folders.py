"""
This class is to calculate PHIQNet features on video frames in a list of video folders, FFMPEG is required
"""
import numpy as np
import subprocess as sp
import json
import os
import tensorflow as tf
from phiqnet.models.model_analysis import phiq_subnet


class CalculateFrameQualityFeatures():
    def __init__(self, model_weights, ffprobe_exe=None, ffmpeg_exe=None, process_frame_interval=0):
        """
        Frame PHIQNet feature computer
        :param model_weights: PHIQNet model_weights file
        :param ffprobe_exe: FFProbe exe file
        :param ffmpeg_exe: FFMPEG exe file
        :param process_frame_interval: parameter of frame processing interval, 0 means all frames will be used
        """
        self.ffmpeg = ffmpeg_exe
        self.ffprobe = ffprobe_exe
        self.process_frame_interval = process_frame_interval
        self.mos_scales = np.array([1, 2, 3, 4, 5])
        self.get_feature_model(model_weights)

    def get_feature_model(self, model_weights):
        self.feature_model = phiq_subnet(n_quality_levels=5, return_backbone_maps=False, return_feature_maps=True,
                                         return_features=True)
        self.feature_model.load_weights(model_weights, by_name=True)

    def get_video_meta(self, video_file):
        """Internal method to get video meta
        :return: a list containing [audio_exit, video_exit, duration, frame_count, height, width, fps]
        """
        cmd = [self.ffprobe, '-i', video_file, '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format']
        ffprobe_output = json.loads(sp.check_output(cmd).decode('utf-8'))

        # audio_exits = False
        video_exits = False
        duration = 0
        frame_count = 0
        height = 0
        width = 0
        fps = 0
        bitrate = 0

        stream_type = 'streams'
        codec_type = 'codec_type'
        if stream_type in ffprobe_output:
            for i in range(len(ffprobe_output[stream_type])):
                if codec_type in ffprobe_output[stream_type][i]:
                    # if ffprobe_output[stream_type][i][codec_type] == 'audio':
                    #     audio_exits = True
                    if ffprobe_output[stream_type][i][codec_type] == 'video':
                        video_exits = True
                        frame_rate = ffprobe_output[stream_type][i]['avg_frame_rate']
                        if '/' in frame_rate:
                            fps_temp = [float(item) for item in frame_rate.split('/')]
                            fps = fps_temp[0] / fps_temp[1]
                        else:
                            fps = float(frame_rate)
                        if 'duration' not in ffprobe_output[stream_type][i]:
                            if 'format' in ffprobe_output:
                                duration = float(ffprobe_output['format']['duration'])
                        else:
                            duration = float(ffprobe_output[stream_type][i]['duration'])
                        frame_count = int(duration * fps)
                        height = ffprobe_output[stream_type][i]['height']
                        width = ffprobe_output[stream_type][i]['width']
                        if 'bit_rate' not in ffprobe_output[stream_type][i]:
                            if 'format' in ffprobe_output:
                                bitrate = int(ffprobe_output['format']['bit_rate'])
                        else:
                            bitrate = int(ffprobe_output[stream_type][i]['bit_rate']) / 1000

        if not video_exits:
            return None
        return [video_exits, duration, frame_count, height, width, fps, bitrate]

    def video_features(self, video_folders, feature_folder):
        """
        :param video_folders: a list of folders of all video files
        :param feature_folder: target folder to store the features files in NPY format
        :return: None
        """
        for video_folder in video_folders:
            video_files = os.listdir(video_folder)
            for video_file in video_files:
                try:
                    if video_file.endswith(('.mkv', '.mp4')): # Only mkv and mps contained in KonViD-1k, LIVE-VQC and YouTube-UGC databases
                        video_path = os.path.join(video_folder, video_file)
                        video_name = os.path.splitext(os.path.basename(video_file))[0]

                        # Path to store the PHIQNet features of a frame and a flipped frame must be defined
                        npy_file_features = r''
                        npy_file_features_flipped = r''

                        if not os.path.exists(os.path.dirname(npy_file_features)):
                            os.makedirs(os.path.dirname(npy_file_features))
                        if not os.path.exists(os.path.dirname(npy_file_features_flipped)):
                            os.makedirs(os.path.dirname(npy_file_features_flipped))
                        frame_features, features_flipped = self.__ffmpeg_frames_features__(
                            os.path.join(video_folder, video_file), flip=True)
                        np.save(npy_file_features, np.asarray(frame_features, dtype=np.float16))
                        np.save(npy_file_features_flipped, np.asarray(features_flipped, dtype=np.float16))
                except Exception:
                    print('{} excep'.format(video_file))

    def __cal_features__(self, image):
        image /= 127.5
        image -= 1.
        return self.feature_model.predict(np.expand_dims(image, axis=0))

    def __ffmpeg_frames_features__(self, video_file, flip=True):
        meta = self.get_video_meta(video_file)
        video_height = meta[3]
        video_width = meta[4]
        video_size = video_height * video_width * 3
        # print('Start reading {}'.format(video_file))
        if self.process_frame_interval > 0:
            fps = 'fps=1/' + str(self.process_frame_interval)
            cmd = [self.ffmpeg, '-i', video_file, '-f', 'image2pipe', '-vf', fps, '-pix_fmt', 'rgb24', '-vcodec',
                   'rawvideo', '-']
        else:
            cmd = [self.ffmpeg, '-i', video_file, '-f', 'image2pipe', '-pix_fmt', 'rgb24', '-hide_banner', '-loglevel',
                   'panic', '-vcodec', 'rawvideo', '-']
        pipe = sp.Popen(cmd, stdout=sp.PIPE)

        features = []
        if flip:
            features_flipped = []
        try:
            while True:
                try:
                    raw_image = pipe.stdout.read(video_size)
                    if len(raw_image) != video_size:
                        break
                    image = np.fromstring(raw_image, dtype='uint8')
                    image = image.reshape((video_height, video_width, 3))
                    image = np.asarray(image, dtype=np.float32)
                    flipped_image = np.fliplr(image)
                    frame_feature = self.__cal_features__(image)
                    features.append(np.asarray(frame_feature))
                    if flip:
                        flipped_frame_features = self.__cal_features__(flipped_image)
                        features_flipped.append(np.array(flipped_frame_features))

                except Exception as e1:
                    print(e1)
                    continue
        except Exception as e2:
            print(e2)
        pipe.stdout.flush()

        if flip:
            return features, features_flipped
        else:
            return features


if __name__ == '__main__':
    ffmpeg_exe = r'...\\ffmpeg\ffmpeg.exe'
    ffprobe_exe = r'...\\ffmpeg\ffprobe.exe'
    model_weights_file = r'..\\model_weights\PHIQNet.h5'

    feature_folder = r'...\\model_weights\frame_features'
    video_frame_features = CalculateFrameQualityFeatures(model_weights=model_weights_file,
                                                         ffmpeg_exe=ffmpeg_exe,
                                                         ffprobe_exe=ffprobe_exe)
    video_folders = [
        r'.\live_vqc_video',
        r'.\ugc_test',
        r'.\ugc_train',
        r'.\ugc_validation',
        r'.\KoNViD_1k_videos'
    ]
    video_frame_features.video_features(video_folders, feature_folder)

