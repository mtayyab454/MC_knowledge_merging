'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
from collections import OrderedDict
from torch.hub import load_state_dict_from_url


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, tasks=None, l=512):
        super(VGG, self).__init__()
        self.features = features
        self.classifiers = nn.ModuleList()

        for t in tasks:
            self.classifiers.append(nn.Linear(l, t))
            # self.classifiers.append( nn.Linear(l, t) )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = []

        for i in range(len(self.classifiers)):
            out.append(self.classifiers[i](x))

        return out

    # def cuda(self):
    #     super(VGG, self).cuda()
    #     for c in self.classifiers:
    #         c.cuda()

    # def cpu(self):
    #     super(VGG, self).cpu()
    #     for c in self.classifiers:
    #         c.cpu()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


x = 1
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M'],
    'D2': [40, 40, 'M', 80, 80, 'M', 160, 160, 160, 'M', 160, 160, 160, 'M', 320, 320, 320, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}

def vgg16_multitask(tasks): # VGG16 for CIFAR10 initilized with CIFAR100 weights

    model = VGG(make_layers(cfg['D']), tasks)

    return model

# model = vgg16_multitask([90, 5, 5])
# print(model(torch.randn(1, 3, 32, 32)))