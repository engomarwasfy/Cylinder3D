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

    # Set scores for unlabelled class to nan so it won't affect the average
    f1_scores[0] = None
    precision_scores[0] = None
    recall_scores[0] = None
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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)