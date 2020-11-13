from tensorflow.keras.callbacks import Callback
import numpy as np
import scipy.stats


class ModelEvaluationGeneratorVQ(Callback):
    """
    Evaluation for VQA, the main function is to calculate PLCC, SROCC, RMSE and MAD after each train epoch.
    """
    def __init__(self, val_generator, evaluation_generator=None):
        super(ModelEvaluationGeneratorVQ, self).__init__()
        self.val_generator = val_generator
        self.evaluation_generator = evaluation_generator

    def __evaluation__(self, vq_generator):
        predictions = []
        mos_scores = []

        for i in range(vq_generator.__len__()):
            features, score = vq_generator.__getitem__(i)
            mos_scores.extend(score)
            prediction = self.model.predict(features)
            predictions.extend(np.squeeze(prediction, 1))

        PLCC = scipy.stats.pearsonr(mos_scores, predictions)[0]
        SROCC = scipy.stats.spearmanr(mos_scores, predictions)[0]
        RMSE = np.sqrt(np.mean(np.subtract(predictions, mos_scores) ** 2))
        MAD = np.mean(np.abs(np.subtract(predictions, mos_scores)))
        return PLCC, SROCC, RMSE, MAD

    def on_epoch_end(self, epoch, logs=None):
        plcc, srcc, rmse, mad = self.__evaluation__(self.val_generator)
        print('\nPLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(plcc, srcc, rmse, mad))

        logs['plcc'] = plcc
        logs['srcc'] = srcc
        logs['rmse'] = rmse

        if self.evaluation_generator:
            if epoch % 10 == 0:
                plcc_10th, srcc_10th, rmse_10th, mad_10th = self.__evaluation__(self.evaluation_generator)
                print('\nEpoch {}: PLCC: {}, SRCC: {}, RMSE: {}, MAD: {}'.format(epoch, plcc_10th, srcc_10th, rmse_10th, mad_10th))
