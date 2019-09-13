import torch
import torch.nn as nn
import random, os
import numpy as np
from util import cal_auc


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.Sigmoid()(x)
        return x


def fusion(Manager, case_num, n_features):

    path = os.path.join(Manager._path['save_path'] + str(case_num))
    all_index = Manager.prep_case(case_num)

    all_train_features = []
    all_val_features = []
    all_test_features = []

    for loc in range(0, 23):
        all_train_features.append(np.load(os.path.join(path, str(loc), 'train_features.npy')))
        all_val_features.append(np.load(os.path.join(path, str(loc), 'val_features.npy')))
        all_test_features.append(np.load(os.path.join(path, str(loc), 'test_features.npy')))

    all_train_features = np.concatenate(all_train_features, axis=1)
    all_val_features = np.concatenate(all_val_features, axis=1)
    all_test_features = np.concatenate(all_test_features, axis=1)

    all_train_labels = Manager.labels['label'][np.concatenate([all_index['train_index_0'], all_index['train_index_1']])]
    all_val_labels = Manager.labels['label'][np.concatenate([all_index['val_index_0'], all_index['val_index_1']])]
    all_test_labels = Manager.labels['label'][all_index['test_index']]

    train_index = np.array([x for x in range(len(all_train_labels))])
    val_index = np.array([x for x in range(len(all_val_labels))])
    test_index = np.array([x for x in range(len(all_test_labels))])

    NetA = Net(n_features * 23)
    par_model = list(NetA.parameters())
    criterion = nn.BCELoss()  # .cuda()
    optimizer = torch.optim.SGD(par_model, lr=1e-6, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1.0)

    num_epochs = 1000
    batch_size = 20

    for epoch in range(num_epochs):
        random.shuffle(train_index)

        scheduler.step()
        stages = ['train', 'val', 'test']

        for phase in stages:

            running_loss = 0.0
            running_corrects = 0.0
            AllOut = []
            AllLabel = []

            if phase == 'train':
                NetA.train(True)  # Set model to training mode
                running_index = train_index
                running_batch = batch_size

            if phase == 'val':
                NetA.train(False)  # Set model to training mode
                running_index = val_index
                running_batch = batch_size

            if phase == 'test':
                NetA.train(False)  # Set model to training mode
                running_index = test_index
                running_batch = batch_size

            for i in range(len(running_index) // running_batch):
                if i + 1 == (len(running_index) // running_batch):
                    running_index_batch = running_index[i * running_batch:]
                else:
                    running_index_batch = running_index[i * running_batch:(i + 1) * running_batch]

                """ Batch data and label"""
                if phase == 'train1':
                    inputs = torch.from_numpy(all_train_features[running_index_batch, :].astype(
                        np.float32))
                    labels = torch.from_numpy(all_train_labels[running_index_batch].astype(np.float32))

                if phase == 'val':
                    inputs = torch.from_numpy(all_val_features[running_index_batch, :].astype(
                        np.float32))
                    labels = torch.from_numpy(all_val_labels[running_index_batch].astype(np.float32))

                if phase == 'test':
                    inputs = torch.from_numpy(all_test_features[running_index_batch, :].astype(
                        np.float32))
                    labels = torch.from_numpy(all_test_labels[running_index_batch].astype(np.float32))

                optimizer.zero_grad()

                """ Forward """
                outputs = NetA(inputs)

                """ Output """
                preds = ((outputs[:, 0]) >= 0.5)
                loss = criterion(outputs, labels)

                if phase == 'train1':
                    loss.backward()
                    optimizer.step()

                AllOut.append(outputs[:, 0].detach().cpu().numpy())
                AllLabel.append(labels.cpu().numpy())

                """ statistics """
                running_loss += loss.item() * inputs.size(0)
                running_corrects += np.sum(
                    preds.cpu().numpy() == labels.data.cpu().numpy())  # torch.sum(preds == labels.data)

            AllOut = np.concatenate(AllOut, axis=0)
            AllLabel = np.concatenate(AllLabel, axis=0)
            auc, _ = cal_auc(y=AllLabel, pred=np.expand_dims(AllOut, 1))

            epoch_loss = running_loss / len(running_index)
            epoch_acc = running_corrects / len(running_index)

            if phase == 'train1':
                TrainLoss = epoch_loss
                TrainAcc = epoch_acc
            if phase == 'val':
                ValLoss = epoch_loss
                ValAcc = epoch_acc
                auc0 = auc
            if phase == 'test':
                TestLoss = epoch_loss
                TestAcc = epoch_acc
                auc1 = auc

        print(
            '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(epoch, TrainAcc, ValAcc, TestAcc, TrainLoss,
                                                                                ValLoss, TestLoss, auc0,
                                                                                auc1))  # /, TrainSen, TestSen))

