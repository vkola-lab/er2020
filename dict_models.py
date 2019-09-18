import torch.nn as nn
import torchvision
import torch
#from encoding.nn import BatchNorm2d
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

    def __init__(self, Manager, zlen, out_class):
        super().__init__()

        xlen = len(Manager.options['load_list'])
        self.zlen = zlen
        self.fusion_method = Manager.options['fusion_method']
        self.out_class = out_class
        use_gpu = [None]

        if Manager.options['network_choice'] == 'alex':
            self.features0 = nn.Sequential(torchvision.models.alexnet(pretrained=True).features[:-2],
                                           nn.Conv2d(256, 1024, 3, stride=1, padding=1)).cuda(use_gpu[0])
            if xlen == 4:
                self.features1 = nn.Sequential(torchvision.models.alexnet(pretrained=True).features[:-2],
                                               nn.Conv2d(256, 1024, 3, stride=1, padding=1)).cuda(use_gpu[0])
            self.l1 = 1024
            self.l2 = 13
            self.copy_channel = True

        if Manager.options['network_choice'] == 'vgg':
            self.features0 = nn.Sequential(torchvision.models.vgg16_bn(pretrained=True).features[:-1],
                                           nn.Conv2d(512, 1024, 3, stride=1, padding=1)).cuda(use_gpu[0])
            if xlen == 4:
                self.features1 = nn.Sequential(torchvision.models.vgg16(pretrained=True).features[:-1],
                                               nn.Conv2d(512, 1024, 3, stride=1, padding=1)).cuda(use_gpu[0])
            self.l1 = 1024
            self.l2 = 14
            self.copy_channel = True

        if Manager.options['network_choice'] == 'present':
            self.features0 = KneeNet([64, 128, 256, 256, 512, 512]).features[:-1].cuda(use_gpu[0])
            if xlen == 4:
                self.features1 = KneeNet([64, 128, 256, 256, 512, 512]).features[:-1].cuda(use_gpu[0])
            self.l1 = 512
            self.l2 = 28
            self.copy_channel = False

        """ Attention Maps """
        self.M = 1
        self.attentions0 = nn.Conv2d(self.l1, self.zlen, kernel_size=1, stride=1, bias=False).cuda(use_gpu[0])
        self.attentions1 = nn.Conv2d(self.l1, self.zlen, kernel_size=1, stride=1, bias=False).cuda(use_gpu[0])

        if self.fusion_method == 'cat':
            self.classifier = nn.Linear(self.l1 * self.zlen * (xlen // 2), self.out_class)
            all_weight = [self.classifier.weight[:, :self.l1]] * self.zlen * (xlen // 2)
            all_weight = torch.cat(all_weight, 1)
            self.classifier.weight.data = all_weight
            self.classifier.cuda(use_gpu[0])

        """ freeze parameters """
        self.par_freeze = []#list(self.features0.parameters()) + list(self.features1.parameters())
        for param in self.par_freeze:
            param.requires_grad = False

    def forward(self, x):  # B 1 224 224 N
        bsize = x[0].shape[0]
        xlen = len(x)

        if self.fusion_method == 'cat':
            self.attentions0.weight.data = \
                1 * self.classifier.weight.data[0, :self.zlen * self.l1].view(self.zlen, self.l1, 1, 1)
            if xlen == 4:
                self.attentions1.weight.data = \
                    1 * self.classifier.weight.data[0, (self.zlen * self.l1):].view(self.zlen, self.l1, 1, 1)

        """ inputs """
        for ii in range(xlen):
            if self.copy_channel:
                x[ii] = torch.cat([x[ii]] * 3, 1)

            x[ii] = x[ii].permute(0, 4, 1, 2, 3)
            x[ii] = x[ii].contiguous().view(x[ii].shape[0] * x[ii].shape[1],
                                            x[ii].shape[2], x[ii].shape[3], x[ii].shape[4])  # (B X N, 3, H, W)

        """ featujre maps """
        if xlen == 4:
            xx = [self.features0(x[0]), self.features0(x[1]), self.features1(x[2]), self.features1(x[3])]
        else:
            xx = [self.features0(x[0]), self.features0(x[1])]  # (B X N, C, l2, l2)

        """ attention maps """
        attentions = []
        for ii in range(2):
            attentions.append(get_attention_maps(xx[ii], self.attentions0, bsize, self.zlen, self.M, self.l2))
        if xlen == 4:
            for ii in range(2, 4):
                attentions.append(get_attention_maps(xx[ii], self.attentions1, bsize, self.zlen, self.M, self.l2))

        """ compare left and right """
        for ii in range(xlen):
            xx[ii] = nn.AvgPool2d(self.l2)(xx[ii])
            xx[ii] = xx[ii].view(bsize, self.zlen, xx[ii].shape[1])  # (B, N, C)

        x = xx[0] - xx[1]  # nn.ReLU()(xx[1]) - nn.ReLU()(xx[0])
        if xlen == 4:
            y = xx[2] - xx[3]

        """ flatten the axial direction """
        x = x.view(x.shape[0], x.shape[1] * x.shape[2])  # (B, N X C)
        features = x

        if xlen == 4:
            y = y.view(x.shape[0], y.shape[1] * y.shape[2])
            x = torch.cat((x, y), 1)

        """ classification """
        x = nn.ReLU()(x)
        x = self.classifier(x)
        if self.out_class == 1:
            x = nn.Sigmoid()(x)

        return x, features, attentions


def get_attention_maps(feature_maps, attentions, bsize, zlen, M, l2):
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

