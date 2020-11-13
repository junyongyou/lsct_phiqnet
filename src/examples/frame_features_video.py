import numpy as np

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


if __name__ == '__main__':
    phiqnet_weights_path = r'..\\model_weights\PHIQNet.h5'
    video_path = r'.\\sample_data\example_video (mos=3.24).mp4'
    features = video_frame_features_PHIQNet(phiqnet_weights_path, video_path)

    # Use None that ResNet50 will download ImageNet Pretrained weights or specify the weight path
    resnet50_imagenet_weights = None#r'.\resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    features_resnet50 = video_frame_features_ResNet50(resnet50_imagenet_weights, video_path)