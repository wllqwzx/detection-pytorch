""" Implementation of yolo """

import torch
import torch.nn as nn
import detection._backbone.darknet as darknet

class YOLO(nn.Module):
    def __init__(self, config=None):
        ''' backbone = darknet53 or darknet21 '''
        super(YOLO, self).__init__()
        if config is None:
            config = _DEFAULT_CONFIG
        self.backbone = darknet.__dict__[config['backbone']]()

        output_dim0 = len(config['anchors'][0])*(5+config['classes'])  # B*(5+C)
        self.feature_map0 = self._make_feature_map(in_filter=self.backbone.layers_out_filters[-1], mid_filter=1024, out_filter=512)
        self.predict0 = self._make_predict(in_filter=512, mid_filter=1024, out_filter=output_dim0)
        self.upsampling0 = nn.Sequential(
            self._make_cbr(512, 256, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        output_dim1 = len(config['anchors'][1])*(5+config['classes'])  # B*(5+C)
        self.feature_map1 = self._make_feature_map(in_filter=self.backbone.layers_out_filters[-2]+256, mid_filter=512, out_filter=256)
        self.predict1 = self._make_predict(in_filter=256, mid_filter=512, out_filter=output_dim1)
        self.upsampling1 = nn.Sequential(
            self._make_cbr(256, 128, 1, 1, 0),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        output_dim2 =  len(config['anchors'][2])*(5+config['classes'])  # B*(5+C)
        self.feature_map2 = self._make_feature_map(in_filter=self.backbone.layers_out_filters[-3]+128, mid_filter=256, out_filter=128)
        self.predict2 = self._make_predict(in_filter=128, mid_filter=256, out_filter=output_dim2)


    def _make_feature_map(self, in_filter, mid_filter, out_filter):
        layers = []
        layers.append(self._make_cbr(in_filter, out_filter, 1, 1, 0))
        layers.append(self._make_cbr(out_filter, mid_filter, 3, 1, 1))
        layers.append(self._make_cbr(mid_filter, out_filter, 1, 1, 0))
        layers.append(self._make_cbr(out_filter, mid_filter, 3, 1, 1))
        layers.append(self._make_cbr(mid_filter, out_filter, 1, 1, 0))
        return nn.Sequential(*layers)

    def _make_predict(self, in_filter, mid_filter, out_filter):
        return nn.Sequential(
            self._make_cbr(in_filter, mid_filter, 3, 1, 1),
            nn.Conv2d(mid_filter, out_filter, 1, 1, 0, bias=True)
        )

    def _make_cbr(self, in_filter, out_filter, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_filter, out_filter, kernel_size, stride, padding),
            nn.BatchNorm2d(out_filter),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)

        feature_map0 = self.feature_map0(x0)
        predict0 = self.predict0(feature_map0)

        x1_aux = self.upsampling0(feature_map0)
        x1_all = torch.cat([x1_aux, x1], 1)
        feature_map1 = self.feature_map1(x1_all)
        predict1 = self.predict1(feature_map1)

        x2_aux = self.upsampling1(feature_map1)
        x2_all = torch.cat([x2_aux, x2], 1)
        feature_map2 = self.feature_map2(x2_all)
        predict2 = self.predict2(feature_map2)

        return predict0, predict1, predict2

_DEFAULT_CONFIG = {
    'backbone': 'darknet53',
    'classes': 80,
    'image_size': 416,
    'anchors': [[[116, 90], [156, 198], [373, 326]],    # feature map 0
                [[30, 61], [62, 45], [59, 119]],        # feature map 1
                [[10, 13], [16, 30], [33, 23]]],        # feature map 2
    'pretrained_path': None
}

if __name__ == '__main__':
    net = YOLO()
    img = torch.randn((1,3,416,416))
    pred0, pred1, pred2 = net(img)
    print(pred0.shape)
    print(pred1.shape)
    print(pred2.shape)
    '''
    torch.Size([1, 255, 13, 13])
    torch.Size([1, 255, 26, 26])
    torch.Size([1, 255, 52, 52])
    '''
