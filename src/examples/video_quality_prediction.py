import numpy as np

from lsct.utils.frame_features_video_folders import CalculateFrameQualityFeatures
from lsct.models.lsct_phiqnet_model import create_model

FFMPEG = r'..\\ffmpeg\ffmpeg.exe'
FFPROBE = r'..\\ffmpeg\ffprobe.exe'


def predict_video_quality(phinqnet_weights_path, lsct_weights_path, video_path):
    frame_features_extractor = CalculateFrameQualityFeatures(phinqnet_weights_path, FFPROBE, FFMPEG)
    features = frame_features_extractor.__ffmpeg_frames_features__(video_path, flip=False)
    features = np.squeeze(np.array(features), axis=2)
    features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2]))

    clip_features = []
    clip_length = 16
    for j in range(features.shape[0] // clip_length):
        clip_features.append(features[j * clip_length: (j + 1) * clip_length, :])
    clip_features = np.array(clip_features)

    transformer_params = [2, 64, 4, 64]
    dropout_rates = 0.1
    cnn_filters = [32, 64]

    feature_length = 5 * 256

    vq_model = create_model(clip_length,
                            feature_length=feature_length,
                            cnn_filters=cnn_filters,
                            transformer_params=transformer_params,
                            dropout_rate=dropout_rates)
    vq_model.summary()
    vq_model.load_weights(lsct_weights_path)
    predict_mos = vq_model.predict(np.expand_dims(clip_features, axis=0))
    return predict_mos[0][0]


if __name__ == '__main__':
    phiqnet_weights_path = r'..\\model_weights\PHIQNet.h5'
    lsct_weights_path = r'..\\model_weights\LSCT.h5'

    video_path = r'.\\sample_data\example_video (mos=3.24).mp4'
    predict_mos = predict_video_quality(phiqnet_weights_path, lsct_weights_path, video_path)
    print('Predicted MOS: {}'.format(predict_mos))
