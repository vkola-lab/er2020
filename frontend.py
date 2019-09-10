from model import DefModel
from dataloader import KneeManager, KneeData
from util import get_args, run_model
from util import get_label_KL_uni as get_label
import os
from collections import Counter


class KneeCAM:
    def __init__(self):
        self.Train_args = vars(get_args())
        self.Options = {'CopyChannel':True,
                        'sampling':'10fold',
                        'use_val':False,
                        'network_choice':'shallow',
                        'fusion_method':'cat',
                        'trial_name':'KL_cor_alex_cat_attention',
                        'load_list':['SAG_IW_TSE_LEFT_resize2', 'SAG_IW_TSE_RIGHT_resize2']}
        self.img_dir = 'data/'
        self.Manager = KneeManager(options=self.Options, train_args=self.Train_args, labels=get_label())
        self.text_name = 'out_' + Options['network_choice'] + '_' + Options['fusion_method'] + '_' + '.txt'
        self.use_gpu = [None]
        if not os.path.isdir('Fusion_Results/' + Options['trial_name']):
            os.mkdir('Fusion_Results/' + Options['trial_name'])
        if not os.path.isdir('cam_print/'):
            os.mkdir('cam_print/')

    def train(self, case_num, slice_name, dataset, testing, cont):
        """ Prepare training """
        model_ini = DefModel(Manager=self.Manager, zlen=len(self.Manager.options['slice_range'][0]))
        self.Manager.init_model(model_ini)

        all_index = self.Manager.prep_case(self.case_num)
        dirname = os.path.join(self.Manager._path['save_path'] + str(self.case_num), slice_name)

        self.Manager.net.cuda()
        """ load trained model """
        if testing or cont:
            print('load check pts:')
            pretrained_dict = torch.load(os.path.join(dirname, 'checkpts'), map_location="cuda:0")
            self.Manager.net.load_state_dict(pretrained_dict)

        """ Prepare  Model"""
        if self.Manager.train_args['RunPar']:
            self.Manager.net = nn.DataParallel(self.Manager.net, device_ids=[0, 1])

        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        text_file = os.path.join(self.Manager._path['save_path'] + str(self.case_num), self.text_name)

        print('Overall dataset:')
        print('Length:  ' + str(len(self.Manager.labels['ID_selected'])))
        print('Labels:  ' + str(Counter(self.Manager.labels['label'])))

        print('Prep training set:')
        self.train_data = dataset(Manager=self.Manager, img_dir=self.img_dir, load_list=self.Manager.options['load_list'],
                             index_list=np.concatenate([all_index['train_index_1'], all_index['train_index_0']],
                                                       axis=0),
                             id_list=self.Manager.labels['ID_selected'], labels=self.Manager.labels['label'])

        if self.Manager.options['use_val']:
            print('Prep validation set:')
            self.val_data = dataset(Manager=self.Manager, img_dir=self.img_dir, load_list=self.Manager.options['load_list'],
                               index_list=np.concatenate([all_index['val_index_0'], all_index['val_index_1']], axis=0),
                               id_list=self.Manager.labels['ID_selected'], labels=self.Manager.labels['label'])
        else:
            val_data = ''

        print('Prep testing set:')
        self.test_data = dataset(Manager=self.Manager, img_dir=self.img_dir, load_list=self.Manager.options['load_list'],
                            index_list=all_index['test_index'],
                            id_list=self.Manager.labels['ID_selected'], labels=self.Manager.labels['label'])

        self.train_load = DataLoader(self.train_data, batch_size=self.Manager.train_args['BatchSize'], shuffle=not testing, num_workers=4)
        self.val_load = DataLoader(self.val_data, batch_size=self.Manager.train_args['BatchSize'], shuffle=False, num_workers=4)
        self.test_load = DataLoader(self.test_data, batch_size=self.Manager.train_args['BatchSize'], shuffle=False, num_workers=4)

        """ train model """
        train_out, val_out, test_out = self._train(text_file, dirname, testing=testing, use_gpu=self.use_gpu)

        """ outputs """
        if testing:
            outdir = os.path.join(os.path.expanduser('~'), 'Dropbox/Z_DL/training_outputs/',
                                  self.Manager.options['trial_name'])
            try:
                os.mkdir(outdir)
            except:
                print('outdir exist')
            np.save(os.path.join(outdir, 'train_out_' + str(case_num) + '.npy'), train_out)
            if Manager.options['use_val']:
                np.save(os.path.join(outdir, 'val_out_' + str(case_num) + '.npy'), val_out)
            np.save(os.path.join(outdir, 'test_out_' + str(case_num) + '.npy'), test_out)
            np.save(os.path.join(outdir, 'test_index_' + str(case_num) + '.npy'), test_data.index_list)
            np.save(os.path.join(outdir, 'test_label_' + str(case_num) + '.npy'), self.Manager.labels['label'][test_data.index_list])

    def _train(self, text_file, dirname, testing, use_gpu):

        """Preparing for training"""
        best_val_loss = np.inf
        best_val_auc = 0

        if testing:
            self.Manager.train_args['NumEpochs'] = 1
        else:
            text_file = open(text_file, "w")

        for t in range(self.Manager.train_args['NumEpochs']):
            tini = time.time()

            train_acc, train_loss, train_auc, _, train_out = run_model(t, Manager=self.Manager,
                                                                        data_load=self.train_load,
                                                                        train_data=self.train_data,
                                                                        train=not testing, use_gpu=use_gpu)

            if Manager.options['use_val']:
                val_acc, val_loss, val_auc, _, val_out = run_model(t, Manager=self.Manager,
                                                                    data_load=self.val_load,
                                                                    train_data=self.val_data,
                                                                    train=False, use_gpu=use_gpu)
            else:
                val_acc = 0
                val_auc = 0
                val_loss = 0
                val_out = None

            test_acc, test_loss, test_auc, _, test_out = run_model(t, Manager=self.Manager,
                                                                    data_load=test_load,
                                                                    train_data=test_data,
                                                                    train=False, use_gpu=use_gpu)

            self.Manager.scheduler.step()

            """Save Model"""
            if (val_loss <= best_val_loss) and not testing:
                best_val_loss = val_loss
                if self.Manager.train_args['RunPar']:
                    torch.save(self.Manager.net.module.state_dict(), os.path.join(dirname, 'checkpts'))
                else:
                    torch.save(self.Manager.net.state_dict(), os.path.join(dirname, 'checkpts'))

            """Print Stats"""
            print('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.1f}'.format(t, train_acc, val_acc, test_acc, train_loss,
                                                                                             val_loss, test_loss, val_auc, test_auc, time.time() - tini))

            if not testing:
                text_file.write('{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.1f} \n'
                      .format(t, train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, val_auc, test_auc, time.time()-tini))
                text_file.flush()

        return train_out, val_out, test_out

