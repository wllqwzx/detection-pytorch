import torch
import torch.nn as nn
import numpy as np
import math

from detection.yolov3.yolo_utils import bbox_iou

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        ''' img_size: (H,W)
            anchors: [[w,h],[w,h],...]
        '''
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + len(num_classes)
        self.img_size = img_size

        self.ignore_threshold = 0.5 #TODO
        self.lambda_xy = 2.5    #TODO
        self.lambda_hw = 2.5    #TODO
        self.lambda_conf = 1.0  # loss weoght for confidence
        self.lambda_cls = 1.0   #TODO

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, prediction, target):
        assert isinstance(prediction, torch.cuda.FloatTensor)
        assert isinstance(target, torch.cuda.FloatTensor)
        assert len(target.shape) == 2 and target.shape[1] == 5

        batch_size = prediction.size(0)
        H = prediction.size(2)
        W = prediction.size(3)

        stride_H = self.img_size[0] / H
        stride_W = self.img_size[1] / W

        # scale anchor from image scale to predicted_cell scale
        scaled_anchors = [(aw/stride_W, ah/stride_H) for aw,ah in self.anchors]

        prediction = prediction.view(batch_size, self.num_anchors, self.bbox_attrs, H, W)
        prediction = prediction.permute(0, 3, 4, 1, 2).contiguous()

        # interpreate yolo output
        cx = torch.sigmoid(prediction[:,:,:,:,0])
        cy = torch.sigmoid(prediction[:,:,:,:,1])
        w = prediction[:,:,:,:,2]
        h = prediction[:,:,:,:,3]
        conf = torch.sigmoid(prediction[:,:,:,:,4])
        clas = torch.sigmoid(prediction[:,:,:,:,5:])

        # build target
        obj_mask, noobj_mask, tx, ty, tw, th, tconf, tclas = self._get_anchor_target(target, scaled_anchors, 
                                                                                     W, H, self.ignore_threshold)
        # create losses
        loss_x = self.bce_loss(cx*obj_mask, tx*obj_mask)
        loss_y = self.bce_loss(cy*obj_mask, ty*obj_mask)
        loss_w = self.mse_loss(w*obj_mask, tw*obj_mask)
        loss_h = self.mse_loss(h*obj_mask, th*obj_mask)

        loss_conf = self.bce_loss(conf*obj_mask, obj_mask) +\
                    0.5*self.bce_loss(conf*noobj_mask, noobj_mask*0)
        loss_clas = self.bce_loss(clas[obj_mask==0], tclas[obj_mask==0])

        # compute total loss
        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy +\
               loss_w * self.lambda_hw + loss_h * self.lambda_hw +\
               loss_conf * self.lambda_conf + loss_clas * self.lambda_conf
        loss_aux = [loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_clas.item()]
        return loss, loss_aux

    def _get_anchor_target(self, target, anchors, W, H, ignore_threshold):
        '''
            target: torch.Tensor: (N, MAX_OBJ, 5)
                    a target contains: [cls, cx, cy, w, h]
                    cx,xy,w,h are between [0,1] (ratio w,r,t img's W and H)
            anchors: List: (B, 2)
        '''
        batch_size = target.shape[0]

        obj_mask = torch.zeros(batch_size, H, W, self.num_anchors).cuda()
        noobj_mask = torch.ones(batch_size, H, W, self.num_anchors).cuda()
        tx = torch.zeros(batch_size, H, W, self.num_anchors).cuda()
        ty = torch.zeros(batch_size, H, W, self.num_anchors).cuda()
        th = torch.zeros(batch_size, H, W, self.num_anchors).cuda()
        tw = torch.zeros(batch_size, H, W, self.num_anchors).cuda()
        tconf = torch.zeros(batch_size, H, W, self.num_anchors).cuda()
        tclas = torch.zeros(batch_size, H, W, self.num_anchors, self.num_classes).cuda()

        for b in range(batch_size):
            for t in range(target.shape[1]):
                if target[b,t].sum() == 0:  # no target(filled with 0)
                    continue
                # get pixel position of gt bbox on predicted_cell
                gt_cx = target[b, t, 1] * W
                gt_cy = target[b, t, 2] * H
                gt_w = target[b, t, 3] * W
                gt_h = target[b, t, 4] * H

                gt_cx_idx = int(gt_cx)  # floor
                gt_cy_idx = int(gt_cy)  # floor

                gt_bbox = torch.FloatTensor(np.array([[0,0,gt_w, gt_h]])).cuda()
                np_anchor = np.concatenate((np.zeros((self.num_anchors, 2)), np.array(self.anchors)), 1)
                anchor_bbox = torch.FloatTensor(np_anchor).cuda()

                ious = bbox_iou(gt_bbox, anchor_bbox)

                noobj_mask[b, gt_cx_idx, gt_cy_idx, ious, ignore_threshold] = 0

                best_n = torch.argmax(ious)
                
                obj_mask[b,gt_cx_idx, gt_cy_idx, best_n] = 1
                tx[b, gt_cx_idx, gt_cy_idx, best_n] = gt_cx - gt_cx_idx
                ty[b, gt_cx_idx, gt_cy_idx, best_n] = gt_cy - gt_cy_idx
                tw[b, gt_cx_idx, gt_cy_idx, best_n] = math.log(gt_w/anchors[best_n][0] + 1e-16)
                th[b, gt_cx_idx, gt_cy_idx, best_n] = math.log(gt_h/anchors[best_n][1] + 1e-16)

                tconf[b, gt_cx_idx, gt_cy_idx, best_n] = 1
                tclas[b, gt_cx_idx, gt_cy_idx, best_n, int(target[b,t,0])] = 1
        return obj_mask, noobj_mask, tx, ty, tw, th, tconf, tclas

