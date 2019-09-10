import torch.nn as nn
import torchvision
import torch
from torch.nn import BatchNorm2d

class KneeNet(nn.Module):
    def __init__(self, FilNum):
        super(KneeNet, self).__init__()
        self.features = nn.Sequential(
        nn.Conv2d(1, FilNum[0], 7, stride=2, padding=3, bias=True),
        BatchNorm2d(FilNum[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
        nn.Conv2d(FilNum[0],FilNum[1], 3, stride=1, padding=1, bias=True),
        BatchNorm2d(FilNum[1]),
        nn.ReLU(),
        nn.Conv2d(FilNum[1],FilNum[2], 3, stride=1, padding=1, bias=True),
        BatchNorm2d(FilNum[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(FilNum[2],FilNum[3], 3, stride=1, padding=1, bias=True),
        BatchNorm2d(FilNum[3]),
        nn.ReLU(),
        nn.Conv2d(FilNum[3],FilNum[4], 3, stride=1, padding=1, bias=True),
        BatchNorm2d(FilNum[4]),
        nn.ReLU(),
        nn.Conv2d(FilNum[4],FilNum[5], 3, stride=1, padding=1, bias=True),
        BatchNorm2d(FilNum[5]),
        nn.ReLU(),
        nn.AvgPool2d(28),
        )
        self.classifier = nn.Sequential(
            nn.Linear(FilNum[5], 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)


class DefModel(nn.Module):

    def __init__(self, Manager, zlen):
        super(DefModel).__init__()

        xlen = len(Manager.options['load_list'])
        self.zlen = zlen
        self.fusion_method = Manager.options['fusion_method']
        use_gpu = [None]

        if Manager.options['network_choice'] == 'alex':
            self.features0 = torchvision.models.alexnet(pretrained=True).features[:-1].cuda(use_gpu[0])
            self.l1 = 256
            self.l2 = 13
            self.copy_channel = True

        if Manager.options['network_choice'] == 'shallow':
            self.features0 = KneeNet([64, 128, 256, 256, 512, 512]).features[:-1].cuda(use_gpu[0])
            self.l1 = 512
            self.l2 = 28
            self.copy_channel = False

        """ CAM Heatmaps """
        self.M = 1
        self.CAMs = (nn.Conv2d(self.l1, self.zlen, kernel_size=1, stride=1, bias=False).cuda(use_gpu[0]))

        if self.fusion_method == 'cat':
            self.classifier = nn.Linear(self.l1 * self.zlen * (xlen // 2), 1)
            all_weight = [self.classifier.weight[:, :self.l1]] * self.zlen * (xlen // 2)
            all_weight = torch.cat(all_weight, 1)
            self.classifier.weight.data = all_weight
            self.classifier.cuda(use_gpu[0])

        """ freeze parameters """
        self.par_freeze = []
        for param in self.par_freeze:
            param.requires_grad = False

    def forward(self, x):  # B 1 224 224 N
        bsize = x[0].shape[0]
        xlen = len(x)

        if self.fusion_method == 'cat':
            self.CAMs.weight.data = \
                1 * self.classifier.weight.data[0, :self.zlen * self.l1].view(self.zlen, self.l1, 1, 1)

        """ inputs """
        for ii in range(xlen):
            if self.copy_channel:
                x[ii] = torch.cat([x[ii]] * 3, 1)

            x[ii] = x[ii].permute(0, 4, 1, 2, 3)
            x[ii] = x[ii].contiguous().view(x[ii].shape[0] * x[ii].shape[1],
                                            x[ii].shape[2], x[ii].shape[3], x[ii].shape[4])  # (B X N, 3, H, W)

        """ featujres """
        xx = [self.features0(x[0]), self.features0(x[1])]  # (B X N, C, l2, l2)

        """ attention maps """
        CAMs = []
        for ii in range(2):
            CAMs.append(get_CAMs_maps(xx[ii], self.CAMs, bsize, self.zlen, self.M, self.l2))

        """ compare left and right """
        for ii in range(xlen):
            xx[ii] = nn.AvgPool2d(self.l2)(xx[ii])
            xx[ii] = xx[ii].view(bsize, self.zlen, xx[ii].shape[1])  # (B, N, C)

        x = xx[0] - xx[1]  # nn.ReLU()(xx[1]) - nn.ReLU()(xx[0])

        """ flatten the axial direction """
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])  # (B, N X C)

        """ classification """
        x = self.classifier(x)
        x = nn.Sigmoid()(x)

        return x, CAMs




