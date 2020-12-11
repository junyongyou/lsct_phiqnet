"""
This class is to calculate ResNet50 (ImageNet pretraied weights) features on video frames in a list of video folders, FFMPEG is required
"""
import numpy as np
import subprocess as sp
import json
import os
import cv2
import time
from scipy.special import gamma


class CalculateFrameQualityFeaturesResnet50():
    def __init__(self, model_weights=None, ffprobe_exe=None, ffmpeg_exe=None, process_frame_interval=0):
        """
        Frame Resnet50 feature computer
        :param model_weights: Resnet50 ImageNet weights file
        :param ffprobe_exe: FFProbe exe file
        :param ffmpeg_exe: FFMPEG exe file
        :param process_frame_interval: parameter of frame processing interval, 0 means all frames will be used
        """
        self.ffmpeg = ffmpeg_exe
        self.ffprobe = ffprobe_exe
        self.process_frame_interval = process_frame_interval

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

    def video_features(self, video_folders, feature_folder, feature_folder_flipped):
        """
        :param video_folders: a list of folders of all video files
        :param feature_folder: target folder to store the features files in NPY format
        :return: None
        """
        for video_folder in video_folders:
            video_files = os.listdir(video_folder)
            for video_file in video_files:
                try:
                    if video_file.endswith(('.mkv', '.mp4')): # Only mkv and mp4 contained in KonViD-1k, LIVE-VQC and YouTube-UGC databases
                        t_start = time.time()
                        video_path = os.path.join(video_folder, video_file)
                        video_file_name = os.path.splitext(os.path.basename(video_file))
                        video_name = video_file_name[0]
                        video_ext = video_file_name[1]

                        # Path to store the PHIQNet features of a frame and a flipped frame must be defined
                        if 'C:' in video_path:
                            npy_file_features = video_path.replace(r'C:\vq_datasets', feature_folder).replace(video_ext, '.npy')
                            npy_file_features_flipped = video_path.replace(r'C:\vq_datasets', feature_folder_flipped).replace(video_ext, '.npy')
                        else:
                            npy_file_features = video_path.replace(r'D:\VQ_datasets', feature_folder).replace(video_ext,
                                                                                                              '.npy')
                            npy_file_features_flipped = video_path.replace(r'D:\VQ_datasets', feature_folder_flipped).replace(
                                video_ext, '.npy')

                        if not os.path.exists(npy_file_features) or not os.path.exists(npy_file_features_flipped):
                            if not os.path.exists(os.path.dirname(npy_file_features)):
                                os.makedirs(os.path.dirname(npy_file_features))
                            if not os.path.exists(os.path.dirname(npy_file_features_flipped)):
                                os.makedirs(os.path.dirname(npy_file_features_flipped))
                            frame_features, features_flipped = self.__ffmpeg_frames_features__(video_path, flip=True)
                            np.save(npy_file_features, np.asarray(frame_features, dtype=np.float16))
                            np.save(npy_file_features_flipped, np.asarray(features_flipped, dtype=np.float16))

                        print('{} feature done!, time: {}'.format(video_file, time.time() - t_start))
                except Exception as e:
                    print('{} excep: {}'.format(video_file, e))

    def preprocess_image(self, img):
        if len(img.shape) == 2:
            image = img
        elif len(img.shape) == 3:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError('The image shape is not correct.')
        return image.astype(np.float64)

    def _estimate_ggd_param(self, vec):
        """Estimate GGD parameter.

        :param vec: The vector that we want to approximate its parameter.
        :type vec: np.ndarray
        """
        gam = np.arange(0.2, 10 + 0.001, 0.001)
        r_gam = (gamma(1.0 / gam) * gamma(3.0 / gam) / (gamma(2.0 / gam) ** 2))

        sigma_sq = np.mean(vec ** 2)
        sigma = np.sqrt(sigma_sq)
        E = np.mean(np.abs(vec))
        rho = sigma_sq / E ** 2

        differences = abs(rho - r_gam)
        array_position = np.argmin(differences)
        gamparam = gam[array_position]

        return gamparam, sigma

    def _estimate_aggd_param(self, vec):
        """Estimate AGGD parameter.

        :param vec: The vector that we want to approximate its parameter.
        :type vec: np.ndarray
        """
        gam = np.arange(0.2, 10 + 0.001, 0.001)
        r_gam = ((gamma(2.0 / gam)) ** 2) / (
                    gamma(1.0 / gam) * gamma(3.0 / gam))

        left_std = np.sqrt(np.mean((vec[vec < 0]) ** 2))
        right_std = np.sqrt(np.mean((vec[vec > 0]) ** 2))
        gamma_hat = left_std / right_std
        rhat = (np.mean(np.abs(vec))) ** 2 / np.mean((vec) ** 2)
        rhat_norm = (rhat * (gamma_hat ** 3 + 1) * (gamma_hat + 1)) / (
                (gamma_hat ** 2 + 1) ** 2)

        differences = (r_gam - rhat_norm) ** 2
        array_position = np.argmin(differences)
        alpha = gam[array_position]

        return alpha, left_std, right_std

    def get_feature(self, img):
        """Get brisque feature given an image.

        :param img: The path or array of the image.
        :type img: str, np.ndarray
        """
        imdist = self.preprocess_image(img)

        scale_num = 2
        feat = np.array([])

        for itr_scale in range(scale_num):
            mu = cv2.GaussianBlur(
                imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
            mu_sq = mu * mu
            sigma = cv2.GaussianBlur(
                imdist * imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
            sigma = np.sqrt(abs((sigma - mu_sq)))
            structdis = (imdist - mu) / (sigma + 1)

            alpha, overallstd = self._estimate_ggd_param(structdis)
            feat = np.append(feat, [alpha, overallstd ** 2])

            shifts = [[0, 1], [1, 0], [1, 1], [-1, 1]]
            for shift in shifts:
                shifted_structdis = np.roll(
                    np.roll(structdis, shift[0], axis=0), shift[1], axis=1)
                pair = np.ravel(structdis, order='F') * \
                       np.ravel(shifted_structdis, order='F')
                alpha, left_std, right_std = self._estimate_aggd_param(pair)

                const = np.sqrt(gamma(1 / alpha)) / np.sqrt(gamma(3 / alpha))
                mean_param = (right_std - left_std) * (
                        gamma(2 / alpha) / gamma(1 / alpha)) * const
                feat = np.append(
                    feat, [alpha, mean_param, left_std ** 2, right_std ** 2])

            imdist = cv2.resize(
                imdist,
                (0, 0),
                fx=0.5,
                fy=0.5,
                interpolation=cv2.INTER_NEAREST
            )
        return feat

    def __cal_features__(self, image):
        brisque_features = self.get_feature(image)
        return brisque_features

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
    ffmpeg_exe = r'C:\lsct_phiqnet\src\ffmpeg\ffmpeg.exe'
    ffprobe_exe = r'C:\lsct_phiqnet\src\ffmpeg\ffprobe.exe'

    feature_folder = r'C:\vq_datasets\BRISQUE_frame_features'
    feature_folder_flipped = r'C:\vq_datasets\BRISQUE_frame_features_flipped'

    # Use None that ResNet50 will download ImageNet Pretrained weights or specify the weight path
    video_frame_features = CalculateFrameQualityFeaturesResnet50(model_weights=r'C:\Users\junyong\Downloads\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                                                 ffmpeg_exe=ffmpeg_exe,
                                                                 ffprobe_exe=ffprobe_exe)
    video_folders = [
        # r'C:\vq_datasets\live_vqc\Video',
        # r'D:\VQ_datasets\ugc_test',
        r'D:\VQ_datasets\ugc_train',
        r'D:\VQ_datasets\ugc_validation',
        r'C:\vq_datasets\KonVid1k\KoNViD_1k_videos'
    ]
    video_frame_features.video_features(video_folders, feature_folder, feature_folder_flipped)

