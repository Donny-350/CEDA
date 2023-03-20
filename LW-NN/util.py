import numpy as np
from scipy import stats
import torch

def normalize(x, min_card_log, max_card_log):
    return np.maximum(np.minimum((np.log(x) - min_card_log) / (max_card_log - min_card_log), 1.0), 0.0)


def unnormalize(x, min_card_log, max_card_log):
    return np.exp(x * (max_card_log - min_card_log) + min_card_log)

def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("qerror的长度：", len(qerror))

    print("25th percentile: {}".format(np.percentile(qerror, 25)))
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))
    return qerror


def print_mse(preds_unnorm, labels_unnorm):
    print("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()))


def print_mape(preds_unnorm, labels_unnorm):
    print("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100))


def print_pearson_correlation(x, y):
    PCCs = stats.pearsonr(x, y)
    print("Pearson Correlation: {}".format(PCCs))

def estimate(model, test_data):
    test_data = torch.FloatTensor(test_data)
    logits = model(test_data)
    predicts = []
    predicts += logits.squeeze(1).tolist()
    return np.array(predicts)
