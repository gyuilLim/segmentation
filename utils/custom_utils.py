import time
import torch
from tqdm import tqdm
import json
import os
import numpy as np

# from utils.train_utils import MetricLogger
from utils.coco_utils import get_coco_api_from_dataset, CocoEvaluator


import torchvision
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def build_model(num_classes=8+1):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

def mAP(model, data_loader) :

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    coco = get_coco_api_from_dataset(data_loader.dataset)
    
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)

    mAP_list = []

    
    for images, targets in tqdm(data_loader):
        images = list(img.to('cuda') for img in images)

        if 'cuda' != torch.device("cpu"):
            torch.cuda.synchronize('cuda')

        model_time = time.time()
        # outputs = model(image, mask=masks)
        outputs = model(images=images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    print_txt = coco_evaluator.coco_eval[iou_types[0]].stats
    AP = print_txt[0]
    mAP = print_txt[1]

    if isinstance(mAP_list, list):
        mAP_list.append(AP)

    return AP, mAP

def calculate_dice_iou(preds, masks) :
    
    smooth = 1

    preds[preds!=0] = 1
    masks[masks!=0] = 1

    intersection = np.sum(preds * masks, axis=(1,2))
    union = np.logical_or(preds, masks)

    dice = (2. * intersection + smooth) / (np.sum(preds, axis=(1,2)) + np.sum(masks, axis=(1,2)) + smooth)
    iou = (intersection + smooth) / (np.sum(union, axis=(1,2))+smooth)

    return dice.sum() / len(dice), iou.sum() / len(iou)




def model_save(model, path, optimizer, scheduler, epoch) :
    save_files = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': scheduler.state_dict(),
        'epoch': epoch}
    
    torch.save(save_files, os.path.join(path, "{}-model-{}.pth".format(cfg.backbone, epoch)))

def result_save(loss_info, mAP_info, AP_info, path) :

        
    with open(path + '/loss_info.json', 'w') as json_file:
        json.dump(loss_info, json_file)

    json_file.close()

    with open(path + '/mAP_info.json', 'w') as json_file:
        json.dump(mAP_info, json_file)

    json_file.close()
    
    with open(path + '/AP_info.json', 'w') as json_file:
        json.dump(AP_info, json_file)
    json_file.close()

