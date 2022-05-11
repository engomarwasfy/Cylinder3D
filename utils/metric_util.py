# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py 

import numpy as np
import torch


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist

def f1_score(prob, label):

    nb_classes = 10
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(label.view(-1), prob.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    confusion_matrix = confusion_matrix.cpu().detach().numpy()

    precision_scores = np.zeros(nb_classes)
    recall_scores = np.zeros(nb_classes)
    f1_scores = np.zeros(nb_classes)
    for i in range(nb_classes):
        precision_scores[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        recall_scores[i] = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        f1_scores[i] = 2 * precision_scores[i] * recall_scores[i] / (precision_scores[i] + recall_scores[i])

    mean_precision = np.nanmean(precision_scores)
    mean_recall = np.nanmean(recall_scores)
    mean_f1_score = np.nanmean(f1_scores)
    metrics = {'mean_precision': mean_precision,
               'mean_recall': mean_recall,
               'mean_f1_score': mean_f1_score,
               'precision_scores': precision_scores,
               'recall_scores': recall_scores,
               'f1_scores': f1_scores}

    return metrics

