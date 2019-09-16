import os, time, imageio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from util import KneeData, run_model
from collections import Counter
from util import get_args, KneeManager
from dict_labels import get_label_pain_uni as get_label
#import cv2, glob

class KneeCAM:
    def __init__(self, out_class):
        torch.manual_seed(100)

        """ Training Parameters"""
        train_args = vars(get_args())
        train_args['NumEpochs'] = 300
        train_args['BatchSize'] = 16 * 4
        train_args['ValGamma'] = 0.9
        train_args['ValLr'] = 1e-6 * 5
        train_args['ValWeightDecay'] = 0.00
        train_args['RunPar'] = True
        train_args['use_gpu'] = [0]

        if train_args['registration']:
            print('Perform Linear Registration')
            registration(['SAG_IW_TSE_LEFT', 'SAG_IW_TSE_RIGHT'])

        """ Options """
        options = {'sampling': 'random7',
                   'use_val': True,
                   'img_dir': 'data/',
                   'load_list': ['SAG_IW_TSE_LEFT_UniA', 'SAG_IW_TSE_RIGHT_UniA'],
                   'slice_range': [None, None],
                   'network_choice': 'present',
                   'fusion_method': 'cat',
                   'trial_name': 'Pain_uni_shallow_2'}

        """ Create and save Manager"""
        self.Manager = KneeManager(options=options, train_args=train_args, labels=get_label(), out_class=out_class)
        print(self.Manager.options)
        print(self.Manager.train_args)

        if not os.path.isdir('Fusion_Results/' + self.Manager.options['trial_name']):
            os.mkdir('Fusion_Results/' + self.Manager.options['trial_name'])

        if not os.path.isdir('cam_print/'):
            os.mkdir('cam_print/')

    def prep_training(self, num_case, num_loc, text_name, training, cont):
        Manager = self.Manager
        options = Manager.options

        """""""""
        all_index: dict() with keys: {'train_index_0': training indices of class 0,
                                      'train_index_1': training indices of class 1,
                                      'val_index_0': validation indices of class 0,
                                      'val_index_1': validation indices of class 1,
                                      'test_index': test indices
        """""""""
        all_index = Manager.prep_case(num_case)
        dirname = os.path.join(Manager._path['save_path'] + str(num_case), str(num_loc))

        self.Manager.net.cuda()
        """ load trained model """
        if training == 'testing' or cont:
            print('load check pts:')
            pretrained_dict = torch.load(os.path.join(dirname, 'PRE0'), map_location="cuda:0")
            Manager.net.load_state_dict(pretrained_dict)

        """ Prepare  Model"""
        if Manager.train_args['RunPar']:
            Manager.net = nn.DataParallel(Manager.net)

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        text_file = os.path.join(Manager._path['save_path'] + str(num_case), text_name)

        print('Overall dataset:')
        print('Length:  ' + str(len(Manager.labels['ID_selected'])))
        print('Labels:  ' + str(Counter(Manager.labels['label'])))

        print('Prep training set:')
        train_data = KneeData(Manager=Manager, img_dir=options['img_dir'], load_list=options['load_list'],
                              index_list=np.concatenate([all_index['train_index_0'], all_index['train_index_1']], axis=0),
                              id_list=Manager.labels['ID_selected'], labels=Manager.labels['label'])

        if options['use_val']:
            print('Prep validation set:')
            val_data = KneeData(Manager=Manager, img_dir=options['img_dir'], load_list=options['load_list'],
                                index_list=np.concatenate([all_index['val_index_0'], all_index['val_index_1']], axis=0),
                                id_list=Manager.labels['ID_selected'], labels=Manager.labels['label'])
        else:
            val_data = ''

        print('Prep testing set:')
        test_data = KneeData(Manager=Manager, img_dir=options['img_dir'], load_list=options['load_list'],
                             index_list=all_index['test_index'],
                             id_list=Manager.labels['ID_selected'], labels=Manager.labels['label'])

        train_load = DataLoader(train_data, batch_size=Manager.train_args['BatchSize'], shuffle=True, num_workers=4)
        val_load = DataLoader(val_data, batch_size=Manager.train_args['BatchSize'], shuffle=False, num_workers=4)
        test_load = DataLoader(test_data, batch_size=Manager.train_args['BatchSize'], shuffle=False, num_workers=4)

        """ train model """
        train_out, val_out, test_out = self.eval_model(Manager, text_file, dirname, train_load, val_load, test_load, train_data, val_data, test_data,
                                                       training=training)

        """ outputs """
        if training == 'testing':
            outdir = os.path.join(os.path.expanduser('~'), 'Dropbox/Z_DL/training_outputs/', options['trial_name'])
            try:
                os.mkdir(outdir)
            except:
                print('outdir exist')
            np.save(os.path.join(outdir, 'train_out_' + str(num_case) + '.npy'), train_out)
            if options['use_val']:
                np.save(os.path.join(outdir, 'val_out_' + str(num_case) + '.npy'), val_out)
            np.save(os.path.join(outdir, 'test_out_' + str(num_case) + '.npy'), test_out)
            np.save(os.path.join(outdir, 'test_index_' + str(num_case) + '.npy'), test_data.index_list)
            np.save(os.path.join(outdir, 'test_label_' + str(num_case) + '.npy'), Manager.labels['label'][test_data.index_list])

    def eval_model(self, Manager, text_file, dirname, train_load, val_load, test_load,
                   train_data, val_data, test_data, training):

        options = Manager.options
        train_args = Manager.train_args

        """Preparing for training"""
        best_val_loss = np.inf
        best_val_auc = 0

        if training == 'testing':
            train_args['NumEpochs'] = 1
        else:
            text_file = open(text_file, "w")

        for t in range(Manager.train_args['NumEpochs']):
            tini = time.time()

            train_acc, train_loss, train_auc, _, train_out, train_collect = run_model(t, Manager=Manager,
                                                                                      data_load=train_load,
                                                                                      train_data=train_data,
                                                                                      phase=training,
                                                                                      use_gpu=train_args['use_gpu'])

            if options['use_val']:
                val_acc, val_loss, val_auc, _, val_out, val_collect = run_model(t, Manager=Manager,
                                                                                data_load=val_load,
                                                                                train_data=val_data,
                                                                                phase='val',
                                                                                use_gpu=train_args['use_gpu'])
            else:
                val_acc = 0
                val_auc = 0
                val_loss = 0

            test_acc, test_loss, test_auc, _, test_out, test_collect = run_model(t, Manager=Manager,
                                                                                 data_load=test_load,
                                                                                 train_data=test_data,
                                                                                 phase='test',
                                                                                 use_gpu=train_args['use_gpu'])

            if t % 10 == 0:
                print_cams(test_collect, t, 2)

            Manager.scheduler.step()

            """Save Model"""
            if (val_loss <= best_val_loss) and training == 'training':
                best_val_loss = val_loss
                if Manager.train_args['RunPar']:
                    torch.save(Manager.net.module.state_dict(), os.path.join(dirname, 'PRE0'))
                else:
                    torch.save(Manager.net.state_dict(), os.path.join(dirname, 'PRE0'))

                np.save(os.path.join(dirname, 'train_features'), train_collect['features'])
                np.save(os.path.join(dirname, 'val_features'), val_collect['features'])
                np.save(os.path.join(dirname, 'test_features'), test_collect['features'])

                np.save(os.path.join(dirname, 'train_out'), train_collect['out'])
                np.save(os.path.join(dirname, 'val_out'), val_collect['out'])
                np.save(os.path.join(dirname, 'test_out'), test_collect['out'])

                np.save(os.path.join(dirname, 'train_labels'), train_collect['labels'])
                np.save(os.path.join(dirname, 'val_labels'), val_collect['labels'])
                np.save(os.path.join(dirname, 'test_labels'), test_collect['labels'])

            if (val_auc >= best_val_auc) and training == 'training':
                best_val_auc = val_auc
                if Manager.train_args['RunPar']:
                    torch.save(Manager.net.module.state_dict(), os.path.join(dirname, 'PRE1'))
                else:
                    torch.save(Manager.net.state_dict(), os.path.join(dirname, 'PRE1'))

            """Print Stats"""
            print('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.1f}'.format(t, train_acc, val_acc, test_acc, train_loss,
                                                                                             val_loss, test_loss, val_auc, test_auc, time.time() - tini))

            if training == 'training':
                text_file.write('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.1f} \n'
                      .format(t, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, val_auc, test_auc, time.time()-tini))
                text_file.flush()

        return train_out, val_out, test_out


def print_cams(collect, t, lenx):
    for cc in range(40, 41):
        out = dcm_n_att(collect, cc, 0, 1, collect['labels'][cc])
        if lenx == 4:
            outB = dcm_n_att(collect, cc, 2, 3, collect['labels'][cc])
            out = np.concatenate([out, outB], axis=0)

        out = out[::2, ::2]

        if collect['labels'][cc] == 1:
            imageio.imwrite('cam_print/' + str(cc) + '_' + str(t) + '_L.jpg', out)
        if collect['labels'][cc] == 0:
            imageio.imwrite('cam_print/' + str(cc) + '_' + str(t) + '_R.jpg', out)


def dcm_n_att(collect, cc, n1, n2, label):
    crop_img0 = []
    crop_img1 = []
    att_img0 = []
    att_img1 = []
    att_diff = []

    for zz in range(collect['x_cropped'][0].shape[4]):
        crop_img0.append(collect['x_cropped'][n1][cc, 0, :, :, zz])
        crop_img1.append(collect['x_cropped'][n2][cc, 0, :, :, zz])
        att_img0.append(collect['attention_maps'][n1][cc, zz, :, :])
        att_img1.append(collect['attention_maps'][n2][cc, zz, :, :])
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


def image_regis(ratio, template, target, warp_mode, iters, eps):

    sz_ori = template.shape
    trans_template = cv2.resize(template,(sz_ori[0]//ratio,sz_ori[1]//ratio))
    trans_target = cv2.resize(target, (sz_ori[0]//ratio, sz_ori[1]//ratio))
    warp_matrix = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    (cc, warp_matrix) = cv2.findTransformECC(trans_template, trans_target, warp_matrix, warp_mode, criteria)
    warp_matrix[0, 2] = warp_matrix[0, 2] * ratio
    warp_matrix[1, 2] = warp_matrix[1, 2] * ratio

    return warp_matrix


def uint16_to_8(x):
    return (x.astype(np.float16) / x.max() * 255).astype(np.uint8)


def registration(sequences):
    for seq in sequences:
        print('Registrating: ' + seq)
        cases = glob.glob('data/raw/' + seq + '/*')
        template = uint16_to_8(np.load(cases[0])[:, :, 18])

        for x in cases:
            target = uint16_to_8(np.load(x)[:, :, 18])
            name = x.split('/')[-1]

            warp_matrix = image_regis(ratio=2, template=template, target=target,
                                      warp_mode=cv2.MOTION_TRANSLATION, iters=500, eps=1e-10)

            transformed = cv2.warpAffine(uint16_to_8(np.load(x)), warp_matrix, target.shape[:2],
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

            np.save('data/registered/' + seq + '/' + name, transformed)

