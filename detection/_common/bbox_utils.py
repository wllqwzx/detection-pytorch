import torch

def CCWH2XYXY(bbox):
    """ transfrom bbox from (Cx Cy W H) format to (X1 Y1 X2 Y2) format
    params:
        bbox: cuda.FloatTensor: (N, 4)
    return:
        bbox1: cuda.FloatTensor: (N, 4)
    """
    bbox1 = torch.stack((bbox[:,0] - bbox[:,2]/2,
                         bbox[:,1] - bbox[:,3]/2,
                         bbox[:,0] + bbox[:,2]/2,
                         bbox[:,1] + bbox[:,3]/2), dim=1)
    return bbox1

def XYXY2CCWH(bbox):
    """transfrom bbox from (X1 Y1 X2 Y2) format to (Cx Cy W H) format
    params:
        bbox: cuda.FloatTensor: (N, 4)
    return:
        bbox1: cuda.FloatTensor: (N, 4)
    """
    bbox1 = torch.stack(((bbox[:,0] + bbox[:,2])/2,
                         (bbox[:,1] + bbox[:,3])/2,
                         bbox[:,2] - bbox[:,0],
                         bbox[:,3] - bbox[:,1]), dim=1)
    return bbox1


def bbox_iou(bbox1, bbox2, mode):
    """
    params:
        bbox1: cuda.FloatTensor: (N1, 4)
        bbox2: cuda.FloatTensor: (N2, 4)
    return:
        ious: cuda.FloatTensor: (N1, N2)
    """
    if mode == "XYXY":
        pass
    elif mode == "CCWH":
        bbox1 = CCWH2XYXY(bbox1)
        bbox2 = CCWH2XYXY(bbox2)
    else:
        print("mode can only be XYXY or CCWH!")
        exit(1)
    top_left = torch.max(bbox1[:,None,:2], bbox2[:,:2])     # (N1,N2,2)
    bottom_right = torch.min(bbox1[:,None,2:], bbox2[:,2:]) # (N1,N2,2)
    area_inter = torch.prod(bottom_right - top_left, dim=2) * (top_left < bottom_right).prod(dim=2).type(torch.cuda.FloatTensor)  # (N1,N2)
    area_1 = torch.prod(bbox1[:,2:]-bbox1[:,:2], dim=1)   # (N1,)
    area_2 = torch.prod(bbox2[:,2:]-bbox2[:,:2], dim=1)   # (N2,)
    ious = area_inter / (area_1[:,None] + area_2 - area_inter)   # (N1, N2)
    return ious


def nms():
    pass



if __name__ == '__main__':
    bbox1 = torch.FloatTensor([[0,0,50,50],[50,0,100,50]]).cuda()
    bbox2 = torch.FloatTensor([[25,25,50,50]]).cuda()
    ious = bbox_iou(bbox1,bbox2,mode="XYXY")
    # test bbox_iou
    assert ious[0].item() == 0.25 and ious[1].item() == 0 
    # test CCWH2XYXY & XYXYCCWH
    assert (CCWH2XYXY(XYXY2CCWH(bbox2)) == bbox2).all().item() == 1
    
