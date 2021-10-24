import numpy as np
import os
import tensorflow as tf

from lsct.utils.frame_features_video_folders import CalculateFrameQualityFeatures
from lsct.ablations.frame_features_video_folders_resnet50 import CalculateFrameQualityFeaturesResnet50

FFMPEG = r'..\\ffmpeg\ffmpeg.exe'
FFPROBE = r'..\\ffmpeg\ffprobe.exe'


"""
This script shows how to calculate PHIQNet features on all video frames, FFMPEG and FFProbe are required 
"""
def video_frame_features_PHIQNet(phinqnet_weights_path, video_path, reture_clip_features=False):
    frame_features_extractor = CalculateFrameQualityFeatures(phinqnet_weights_path, FFPROBE, FFMPEG)
    features = frame_features_extractor.__ffmpeg_frames_features__(video_path, flip=False)
    features = np.squeeze(np.array(features), axis=2)
    features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

    if reture_clip_features:
        clip_features = []
        clip_length = 16
        for j in range(features.shape[0] // clip_length):
            clip_features.append(features[j * clip_length: (j + 1) * clip_length, :])
        clip_features = np.array(clip_features)
        return clip_features

    return np.array(features)


def video_frame_features_ResNet50(resnet50_weights_path, video_path, reture_clip_features=False):
    frame_features_extractor = CalculateFrameQualityFeaturesResnet50(resnet50_weights_path, FFPROBE, FFMPEG)
    features = frame_features_extractor.__ffmpeg_frames_features__(video_path, flip=False)
    features = np.squeeze(np.array(features), axis=1)

    if reture_clip_features:
        clip_features = []
        clip_length = 16
        for j in range(features.shape[0] // clip_length):
            clip_features.append(features[j * clip_length: (j + 1) * clip_length, :])
        clip_features = np.array(clip_features)
        return clip_features

    return np.array(features, np.float16)


def video_frame_features_ResNet50_folder(resnet50_weights_path, video_folder, target_folder):
    frame_features_extractor = CalculateFrameQualityFeaturesResnet50(resnet50_weights_path, FFPROBE, FFMPEG)

    video_types = ('.mp4', '.mpg')
    video_paths = [f for f in os.listdir(video_folder) if f.endswith(video_types)]
    video_paths = video_paths[:70000]
    numb_videos = len(video_paths)

    for i, video_path in enumerate(video_paths):
        ext = os.path.splitext(video_path)
        np_file = os.path.join(target_folder, '{}.npy'.format(ext[0]))
        if not os.path.exists(np_file):
            features = frame_features_extractor.__ffmpeg_frames_features__(os.path.join(video_folder, video_path), flip=False)
            features = np.squeeze(np.array(features), axis=1)
            features = np.array(features, dtype=np.float16)
            np.save(np_file, features)
            print('{} out of {}, {} done'.format(i, numb_videos, video_path))
        else:
            print('{} out of {}, {} already exists'.format(i, numb_videos, video_path))


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    # phiqnet_weights_path = r'..\\model_weights\PHIQNet.h5'
    # video_path = r'.\\sample_data\example_video (mos=3.24).mp4'
    video_folder = r'K:\Faglitteratur\VQA\k150ka'
    # features = video_frame_features_PHIQNet(phiqnet_weights_path, video_path)

    # Use None that ResNet50 will download ImageNet Pretrained weights or specify the weight path
    resnet50_imagenet_weights = r'C:\pretrained_weights_files\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # features_resnet50 = video_frame_features_ResNet50(resnet50_imagenet_weights, video_path)

    target_folder = r'F:\k150k_features'
    video_frame_features_ResNet50_folder(resnet50_imagenet_weights, video_folder, target_folder)
    t = 0