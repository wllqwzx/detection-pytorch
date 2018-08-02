""" Implementation of backbone network: Darknet """

import torch
import torch.nn as nn
import math

class BasicBlock(nn.Module):
    def __init__(self, inplane, planes):
        '''
        inplace: in channel
        planes: output channels of first and second conv
        '''
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplane, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = out + x
        return out


class Darknet(nn.Module):
    def __init__(self, n_blocks):
        super(Darknet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        self.inplane = 32

        self.layer1 = self._make_layer([32,    64], n_blocks[0])
        self.layer2 = self._make_layer([64,   128], n_blocks[1])
        self.layer3 = self._make_layer([128,  256], n_blocks[2])
        self.layer4 = self._make_layer([256,  512], n_blocks[3])
        self.layer5 = self._make_layer([512, 1024], n_blocks[4])

        # feature map dimantions of layer3,4,5
        self.layers_out_filters = [256,512,1024]

        # parameters init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # init scale
                m.bias.data.zero_()     # init shift


    def _make_layer(self, planes, n_block):
        layers = []
        
        # downsample layer
        layers.append(nn.Conv2d(self.inplane, planes[1], kernel_size=3, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(planes[1]))
        layers.append(nn.LeakyReLU(0.1))

        # n_block layers
        self.inplane = planes[1]
        for _ in range(n_block):
            layers.append(BasicBlock(self.inplane, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out0 = self.relu1(self.bn1(self.conv1(x)))
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet21():
    ''' without last avgpool(global), FC(1000), softmax layers. Actually 20 layers'''
    return Darknet([1,1,2,2,1])

def darknet53():
    ''' without last avgpool(global), FC(1000), softmax layers. Actually 52 layers'''
    return Darknet([1,2,8,8,4])
