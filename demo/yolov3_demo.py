import torch
import numpy as np
import os
import sys
import glob
import cv2

LIB_PATH = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(LIB_PATH)

import detection.yolov3.yolo as yolo
import detection.yolov3.yolo_utils as yolo_utils

IMG_PATH = LIB_PATH + '/demo/images'
images = glob.glob(IMG_PATH+'/*.jpg')
CONFIG = yolo._DEFAULT_CONFIG

model = yolo.YOLO(CONFIG)
model = model.cuda()

for file in images:
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    if img is None:
        print("image:", file, "read failed!")
        continue
    print("image:", file, "read succeed!")
    origin_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CONFIG['image_size'], CONFIG['image_size']), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(img, (2, 0, 1))  #(C, H, W)
    img = img.astype(np.float32)
    img /= 255.0

    img = torch.from_numpy(img).cuda()
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    pred0, pred1, pred2 = model(img)
    out0 = yolo_utils.prediction2bbox(pred0, CONFIG['anchors'][0], CONFIG['classes'], (CONFIG['image_size'], CONFIG['image_size']))
    out1 = yolo_utils.prediction2bbox(pred1, CONFIG['anchors'][1], CONFIG['classes'], (CONFIG['image_size'], CONFIG['image_size']))
    out2 = yolo_utils.prediction2bbox(pred2, CONFIG['anchors'][2], CONFIG['classes'], (CONFIG['image_size'], CONFIG['image_size']))

    #===== test
    # print(out2.shape)
    for bbox in out2[0]:
        if bbox[4].item() > 0.7:
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
            clas = torch.argmax(bbox[5:]).item()
            ori_h, ori_w = origin_img.shape[0], origin_img.shape[1]
            post_h, post_w = img.shape[2], img.shape[3]
            
            ori_x1 = (x1 / post_w) * ori_w
            ori_y1 = (y1 / post_h) * ori_h
            ori_x2 = (x2 / post_w) * ori_w
            ori_y2 = (y2 / post_h) * ori_h
            # print("bbox: ", ori_x1, ori_y1, ori_x2, ori_y2)
            cv2.rectangle(origin_img, (ori_x1, ori_y1), (ori_x2, ori_y2), (0,0,255), 3)
            cv2.putText(origin_img, str(clas), (ori_x1, ori_y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
            cv2.imwrite(LIB_PATH +'/demo/output/'+'yolov3_'+file[-9:], origin_img)