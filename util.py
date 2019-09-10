import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, os
import glob
from torch.utils.data import DataLoader, Dataset
from optparse import OptionParser
from codes_dcm_new.Knee_training_lib import get_index
from collections import Counter
from sklearn import metrics
import imageio


def dcm_n_att(all_x_cropped, all_attention_maps, cc, n1, n2, label):
    crop_img0 = []
    crop_img1 = []
    att_img0 = []
    att_img1 = []
    att_diff = []
    for zz in range(all_x_cropped[0].shape[4]):
        crop_img0.append(all_x_cropped[n1][cc, 0, :, :, zz])
        crop_img1.append(all_x_cropped[n2][cc, 0, :, :, zz])
        att_img0.append(all_attention_maps[n1][cc, zz, :, :])
        att_img1.append(all_attention_maps[n2][cc, zz, :, :])
        if label == 1:
            att_diff.append(att_img0[zz] - att_img1[zz])
        if label == 0:
            att_diff.append(att_img1[zz] - att_img0[zz])

    crop_img0 = npy_2_uint8(np.concatenate(crop_img0, axis=1))
    crop_img1 = npy_2_uint8(np.concatenate(crop_img1, axis=1))
    att_img0 = (np.concatenate(att_img0, axis=1))
    att_img1 = (np.concatenate(att_img1, axis=1))
    att_img = npy_2_uint8(np.concatenate([att_img0, att_img1], axis=0))
    att_diff = npy_2_uint8(np.concatenate(att_diff, axis=1))

    return np.concatenate([crop_img0, crop_img1, att_img, att_diff], axis=0)


def npy_2_uint8(x):
    xmin = x.min()
    xmax = x.max()
    x = (x - xmin) / (xmax - xmin)
    assert (x.max() == 1 and x.min() == 0)
    x = x * 255
    return x.astype(np.uint8)


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


def get_CAMs_maps(feature_maps, attentions, bsize, zlen, M, l2):
    # (B X N, C, l2, l2)
    feature_maps = feature_maps.view(bsize, zlen, feature_maps.shape[1], l2, l2)  # (B, N, C, l2, l2)
    attention_maps = []
    for ii in range(zlen):
        map = attentions(feature_maps[:, ii, :, :, :])[:, ii, :, :].unsqueeze(1)
        attention_maps.append(map)

    attention_maps = torch.cat(attention_maps, 1)

    # Normalize Attention Map
    attention_map = attention_maps.view(bsize, -1)  # (B, H * W)
    attention_map_max, _ = attention_map.max(dim=1, keepdim=True)  # (B, 1)
    attention_map_min, _ = attention_map.min(dim=1, keepdim=True)  # (B, 1)
    attention_map = (attention_map - attention_map_min) / (attention_map_max - attention_map_min)  # (B, H * W)
    attention_map = attention_map.view(bsize, zlen, l2, l2)  # (B, 1, H, W)

    return attention_map


def load_OAI_var():

    all_path = os.path.join(os.path.expanduser('~'), 'Dropbox/Z_DL/ID_and_Label/')

    all_var = glob.glob(all_path + '*.npy')
    all_var.sort()
    v = dict()
    for var in all_var:
        name = var.split('/')[-1].split('.')[0]
        v[name] = np.load(var, allow_pickle=True)

    return v


def get_label_KL_uni():
    Labels = dict()
    v = load_OAI_var()

    present = np.concatenate([v['ID_bil_fre_pain_C'], v['ID_bil_fre_pain_E'], v['ID_uni_fre_pain']], axis=0)
    exist = np.array([(x in present) for x in v['ID_main']])
    quality = np.ones(v['ID_main'].shape)

    pos = ((v['kl00_left'] - v['kl00_right']) >= 2)
    neg = ((v['kl00_right'] - v['kl00_left']) >= 2)

    select_condition = (pos | neg)
    pick = ((np.logical_and(quality == 1, exist)) & select_condition)

    Labels['label'] = (pos).astype(np.int64)[pick]
    Labels['ID_selected'] = v['ID_main'][pick]

    return Labels


def run_model(t, Manager, data_load, train_data, train, use_gpu):

    test_loss = []
    test_out = []
    test_correct = 0
    test_total = 0
    if train:
        Manager.net.train(True)
    else:
        Manager.net.train(False)

    """ data to collect """
    all_idx = []
    all_labels = []

    all_attention_maps = []
    all_x_cropped = []
    for ii in range(len(Manager.options['load_list'])):
        all_attention_maps.append([])
        all_x_cropped.append([])

    """ data to collect """
    for x, labels, idx in data_load:
        all_labels.append(labels)
        all_idx.append(idx)
        labels = labels[:, 0]

        x = [y.cuda(use_gpu[0]) for y in x]  # (B X 1 X H X W X N)
        labels = labels.cuda(use_gpu[0])

        """ collect attention cropping """
        out, attention_map = Manager.net(x)
        with torch.no_grad():
            for ii in range(len(x)):
                attention_maps = F.upsample_bilinear(attention_map[ii], size=(x[0].size(2), x[0].size(3)))  # (B X M X H X W)
                all_attention_maps[ii].append(attention_maps.cpu().numpy())

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
        if train:
            loss.backward()
            if test_total == len(train_data):
                Manager.optimizer.step()
                Manager.optimizer.zero_grad()

    """ collect CAM maps and cropped images and sort by id"""
    if not train and ((t+1) % 100 == 0):
        with torch.no_grad():
            all_idx = torch.cat(all_idx, 0)
            sort_id = torch.argsort(all_idx)[:20]
            all_labels = torch.cat(all_labels, 0)
            all_labels = all_labels[sort_id]

            for ii in range(len(x)):
                all_attention_maps[ii] = np.concatenate(all_attention_maps[ii], axis=0)
                all_attention_maps[ii] = all_attention_maps[ii][sort_id, ::]
                all_x_cropped[ii] = np.concatenate(all_x_cropped[ii], axis=0)
                all_x_cropped[ii] = all_x_cropped[ii][sort_id, ::]

            for cc in range(0, 20):
                out = dcm_n_att(all_x_cropped, all_attention_maps, cc, 0, 1, all_labels[cc])
                if len(x) == 4:
                    outB = dcm_n_att(all_x_cropped, all_attention_maps, cc, 2, 3)
                    out = np.concatenate([out, outB], axis=0)

                out = out[::2, ::2]

                if all_labels[cc] == 1:
                    imageio.imwrite('cam_print/' + str(cc) + '_' + str(t) + '_L.jpg', out)
                if all_labels[cc] == 0:
                    imageio.imwrite('cam_print/' + str(cc) + '_' + str(t) + '_R.jpg', out)

    """ final stats """
    test_out = np.concatenate(test_out, axis=0)
    test_acc = test_correct / test_total
    test_loss = sum(test_loss) / len(train_data)

    auc0, auc1 = cal_auc(y=train_data.labels[train_data.index_list], pred=test_out)

    return test_acc, test_loss, auc0, auc1, test_out









