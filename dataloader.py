import os
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
import numpy as np
from collections import Counter

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

        print(self._path['save_path'])
        all_index = get_index(self._path['save_path'] + str(case_num), self.options['use_val'])

        return all_index


def get_index(path, use_val):
    """
    all_index: dict() with keys: {'train_index_0': training indices of class 0,
                                  'train_index_1': training indices of class 1,
                                  'val_index_0': validation indices of class 0,
                                  'val_index_1': validation indices of class 1,
                                  'test_index': test indices
    """
    all_index = {'train_index_0': np.load(os.path.join(path, 'allindex', 'train_index_0.npy')),
                 'train_index_1': np.load(os.path.join(path, 'allindex', 'train_index_1.npy')),
                 'test_index': np.load(os.path.join(path, 'allindex', 'test_index.npy'))}
    if use_val:
        all_index['val_index_0'] = np.load(os.path.join(path, 'allindex', 'val_index_0.npy'))
        all_index['val_index_1'] = np.load(os.path.join(path, 'allindex', 'val_index_1.npy'))

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

        self.LL = np.load('data/LL_pain3.npy')[:, :, :, :, self.slice_range[0]]
        self.RR = np.load('data/RR_pain3.npy')[:, :, :, :, self.slice_range[1]]

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):

        index = self.index_list[idx]

        """ slice from existing matrices"""
        slice_range = self.slice_range
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

