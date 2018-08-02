import torch

def prediction2bbox(prediction, anchors, num_classes, img_size):
    assert isinstance(prediction, torch.cuda.FloatTensor)
    num_anchors = len(anchors)
    bbox_attrs = 5 + num_classes

    batch_size = prediction.size(0)
    B = len(anchors)
    H = prediction.size(2)
    W = prediction.size(3)

    stride_H = img_size[0] / H
    stride_W = img_size[1] / W

    # scale anchor from image scale to predicted_cell scale
    scaled_anchors = [(aw/stride_W, ah/stride_H) for aw,ah in anchors]

    prediction = prediction.view(batch_size, B, bbox_attrs, H, W)
    prediction = prediction.permute(0, 1, 3, 4, 2).contiguous() #(N, B, H, W, 5+Cls) 

    # interpreate yolo output
    cx = torch.sigmoid(prediction[:,:,:,:,0])   #(N, B, H, W)
    cy = torch.sigmoid(prediction[:,:,:,:,1])   #(N, B, H, W)
    w = prediction[:,:,:,:,2]   #(N, B, H, W)
    h = prediction[:,:,:,:,3]   #(N, B, H, W)
    conf = torch.sigmoid(prediction[:,:,:,:,4])   #(N, B, H, W)
    clas = torch.sigmoid(prediction[:,:,:,:,5:])  #(N, B, H, W, Cls)

    # Calculate offsets for each grid (prediction_map scale)
    grid_x = torch.linspace(0,W-1,W).repeat(H, 1).repeat(B, 1, 1).repeat(batch_size, 1,1,1).type(torch.FloatTensor).cuda()
    grid_y = torch.linspace(0,H-1,H).repeat(W, 1).t().repeat(B,1,1).repeat(batch_size,1,1,1).type(torch.FloatTensor).cuda()

    # Calculate anchor w, h (prediction_map scale)
    anchor_w = torch.FloatTensor(scaled_anchors)[:,0].repeat(batch_size, H, W, 1).permute(0,3,1,2).contiguous().cuda()
    anchor_h = torch.FloatTensor(scaled_anchors)[:,1].repeat(batch_size, H, W, 1).permute(0,3,1,2).contiguous().cuda()

    # Add offset to x,y and scale to w,h
    pred = torch.randn(batch_size, B, H, W, 4).cuda()
    pred[:,:,:,:,0] = cx.data + grid_x
    pred[:,:,:,:,1] = cy.data + grid_y
    pred[:,:,:,:,2] = torch.exp(w) * anchor_w 
    pred[:,:,:,:,3] = torch.exp(h) * anchor_h

    img_scale = torch.FloatTensor([stride_W, stride_H, stride_W, stride_H]).cuda()

    output = torch.cat((pred.view(batch_size, -1, 4) * img_scale,   # to image scale
                        conf.view(batch_size, -1, 1),
                        clas.view(batch_size, -1, num_classes)), 2)
    return output.data


if __name__ == '__main__':
    prediction = torch.randn(16, 255, 13, 13).cuda()   # (N, 3*85, H, W)
    anchors = [[116, 90], [156, 198], [373, 326]]
    num_classes = 80
    img_size = [416, 416]
    out = prediction2bbox(prediction, anchors, num_classes, img_size)
    print(out.shape)    #=> (16, 507, 85)
