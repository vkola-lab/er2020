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


class KneeManager(object):
    def __init__(self, options, train_args, labels, out_class):

        self.train_args = train_args
        self.options = options

        """ Define Case and Path """
        self._path = dict()
        self._path['save_path'] = os.path.join(os.getcwd(), 'Fusion_Results', self.options['trial_name'], 'Reg')
        self.labels = labels
        if out_class == 1:
            self.criterion = nn.BCELoss()
        if out_class == 2:
            self.criterion = nn.CrossEntropyLoss()

        """ Initialize model """
        self.net = None
        self.optimizer = None
        self.scheduler = None

    def init_model(self, model_ini):
        self.net = model_ini

        """ freeze certain parameters"""
        par_model = list(set(self.net.parameters()) - set(self.net.par_freeze))
        self.optimizer = torch.optim.SGD(par_model,
                                         lr=self.train_args['ValLr'], momentum=0.9,
                                         weight_decay=self.train_args['ValWeightDecay'])

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20,
                                                         gamma=self.train_args['ValGamma'])

    def prep_case(self, case_num):
        if not os.path.isdir(self._path['save_path'] + str(case_num)):
            os.mkdir(self._path['save_path'] + str(case_num))

        all_index, _ = get_index(self._path['save_path'] + str(case_num), case_num, self.options['sampling'],
                                 self.options['use_val'], self.labels['label'])

        return all_index


class KneeData(Dataset):

    def __init__(self, Manager, img_dir, load_list, index_list, id_list, labels):
        self.img_dir = img_dir
        self.load_list = load_list
        self.index_list = index_list
        assert len(id_list) == len(labels)
        self.id_list = id_list
        self.labels = labels

        print('Length:  ' + str(len(index_list)))
        print('Labels:  ' + str(Counter(labels[index_list])))

        self.slice_range = Manager.options['slice_range']

        self.LL = np.load('data/LL_pain3.npy')
        self.RR = np.load('data/RR_pain3.npy')

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):

        slice_range = self.slice_range
        index = self.index_list[idx]

        """ slice from existing matrices"""
        id_all = self.id_list[index]
        if type(idx) == int:
            id_all = [id_all]
        data = []
        for ii in range(len(self.load_list)):
            x = []
            for id in id_all:
                temp = np.load(os.path.join(self.img_dir, self.load_list[ii], id + '.npy'), mmap_mode='r')
                temp = temp[:, :, slice_range[ii]] # 224 X 224 X Slices
                temp = temp / temp.max()
                temp = np.expand_dims(temp, axis=0)  # 1 X 224 X 224 X Slices
                if type(idx) != int:
                    temp = np.expand_dims(temp, axis=1)  # 1 X 1 X 224 X 224 X Slices
                x.append(temp)
            if type(idx) == int:
                x = x[0]
            else:
                x = np.concatenate(x, 0)

            data.append(torch.FloatTensor(x))

        label = np.array([self.labels[index]])
        label = torch.from_numpy(label)

        return data, label, idx




