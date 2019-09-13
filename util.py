import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os, random
from torch.utils.data import DataLoader, Dataset
from optparse import OptionParser
from codes_dcm_new.Knee_training_lib import get_index
from collections import Counter
from sklearn import metrics


def get_args():
    parser = OptionParser()
    parser.add_option('--rp', dest='RunPar', default=False)
    parser.add_option('--l1', dest='UseL1', default=False)
    parser.add_option('--l1v', dest='ValL1Factor', default=0.01, type='float')
    parser.add_option('--epochs', dest='NumEpochs', default=300, type='float')
    parser.add_option('--bs', dest='BatchSize', default=50, type='int')
    parser.add_option('--lr', dest='ValLr', default=0.0002, type='float')
    parser.add_option('--l2', dest='ValWeightDecay', default=0.00, type='float')
    parser.add_option('--gamma', dest='ValGamma', default=1.0, type='float')
    parser.add_option('--bstep', dest='BStep', default=10, type='int')

    (options, args) = parser.parse_args()
    return options


def cal_auc(y, pred):
    if pred.shape[1] == 2:
        fpr0, tpr0, thresholds = metrics.roc_curve(y, pred[:, 0], pos_label=0)
        fpr1, tpr1, thresholds = metrics.roc_curve(y, pred[:, 1], pos_label=1)
    else:
        fpr0, tpr0, thresholds = metrics.roc_curve(y, pred[:, 0], pos_label=1)
        fpr1 = fpr0
        tpr1 = tpr0

    return metrics.auc(fpr0, tpr0), metrics.auc(fpr1, tpr1)


def run_model(t, Manager, data_load, train_data, phase, use_gpu):

    test_loss = []
    test_out = []
    test_correct = 0
    test_total = 0

    if phase == 'training':
        Manager.net.train(True)
    else:
        Manager.net.train(False)

    """ data to collect """
    collect = dict()
    collect['idx'] = []
    collect['labels'] = []
    collect['attention_maps'] = []
    collect['x_cropped'] = []
    collect['features'] = []
    collect['out'] = []
    for ii in range(len(Manager.options['load_list'])):
        collect['attention_maps'].append([])
        collect['x_cropped'].append([])

    for x, labels, idx in data_load:
        labels = labels[:, 0]

        x = [y.cuda(use_gpu[0]) for y in x]  # (B X 1 X H X W X N)
        labels = labels.cuda(use_gpu[0])

        """ collect attention cropping """
        out, features, attention_map = Manager.net(x.copy())

        with torch.no_grad():
            collect['labels'].append(labels)
            collect['idx'].append(idx)
            collect['features'].append(features.cpu().numpy())
            collect['out'].append(out.cpu().numpy())

            for ii in range(len(x)):
                attention_maps = F.upsample_bilinear(attention_map[ii], size=(x[0].size(2), x[0].size(3)))  # (B X M X H X W)
                collect['attention_maps'][ii].append(attention_maps.cpu().numpy())

        """ collect cropped images"""
        if (t % 1 == 0):  # and phase == 'test':
            for ii in range(len(x)):
                collect['x_cropped'][ii].append(x[ii].cpu().numpy())

        if out.shape[1] == 1:
            preds = ((out[:,0]) >= 0.5)
            loss = Manager.criterion(out[:,0], labels.float()) * out.shape[0]
        else:
            _, preds = torch.max(out.data, 1)
            loss = Manager.criterion(out, labels) * out.shape[0]

        test_out.append(out.data.cpu().numpy())
        test_loss.append(loss.item())
        # Prediction.
        test_total += labels.shape[0]
        test_correct += np.sum(preds.cpu().numpy() == labels.cpu().numpy())

        """ Backward pass."""
        if phase == 'training':
            loss.backward()
            if test_total == len(train_data):
                Manager.optimizer.step()
                Manager.optimizer.zero_grad()

    """ collect attention maps and cropped images and sort by index"""
    if (t % 1 == 0): #and phase == 'test':
        with torch.no_grad():
            collect['idx'] = torch.cat(collect['idx'], 0)
            sort_id = torch.argsort(collect['idx'])[:]
            collect['labels'] = torch.cat(collect['labels'], 0).cpu().numpy()
            collect['labels'] = collect['labels'][sort_id]
            collect['features'] = np.concatenate(collect['features'], axis=0)
            collect['features'] = collect['features'][sort_id, ::]
            collect['out'] = np.concatenate(collect['out'], axis=0)
            collect['out'] = collect['out'][sort_id, ::]

            for ii in range(len(x)):
                collect['attention_maps'][ii] = np.concatenate(collect['attention_maps'][ii], axis=0)
                collect['attention_maps'][ii] = collect['attention_maps'][ii][sort_id, ::]
                collect['x_cropped'][ii] = np.concatenate(collect['x_cropped'][ii], axis=0)
                collect['x_cropped'][ii] = collect['x_cropped'][ii][sort_id, ::]

    """ final stats """
    test_out = np.concatenate(test_out, axis=0)
    test_acc = test_correct / test_total
    test_loss = sum(test_loss) / len(train_data)

    auc0, auc1 = cal_auc(y=train_data.labels[train_data.index_list], pred=test_out)

    return test_acc, test_loss, auc0, auc1, test_out, collect




