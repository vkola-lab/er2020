import torch
import torch.nn as nn
import numpy as np
import time, os
from torch.utils.data import Dataset
from codes_dcm_new.Knee_training_lib import get_index


class KneeManager(object):
    def __init__(self, options, train_args, labels):

        self.train_args = train_args
        self.options = options

        """ Define Case and Path """
        self._path = dict()
        self._path['save_path'] = os.path.join(os.getcwd(), 'Fusion_Results', self.options['trial_name'], 'Reg')
        self.labels = labels
        self.criterion = torch.nn.BCELoss()

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

        """ Labels """
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

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):

        slice_range = self.slice_range

        index = self.index_list[idx]
        id_all = self.id_list[index]
        if type(idx) == int:
            id_all = [id_all]

        """ slice from existing matrices"""
        data = []
        for ii in range(len(self.load_list)):
            x = []
            for id in id_all:
                temp = np.load(os.path.join(self.img_dir, self.load_list[ii], id + '.npy'), mmap_mode='r')
                temp = temp / temp.max()
                temp = temp[:, :, slice_range[ii]] # 224 X 224 X Slices
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