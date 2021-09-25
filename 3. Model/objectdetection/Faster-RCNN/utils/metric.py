import torch
from tqdm import tqdm

def get_bbox(image_id, target, device='cpu', pred=False):
    """
        image_id (int): image id from annotation.
        target (dict): target data from annotation or result of model inference.
        device (string): use CUDA of CPU.
        pred (bool): if True, result include confidence score of bbox.
            if False, result include only bboxes and labels.
            
        
    """
    
    box_num = target['boxes'].shape[0]
    
    image_id = torch.tensor(image_id).broadcast_to(box_num, 1)
    boxes = target['boxes'].reshape(box_num, 4).to(device)
    labels = target['labels'].reshape(box_num, 1).to(device)
    
    if pred:
        scores = target['scores'].reshape(box_num, 1).to(device)
        bbox_list = torch.cat([image_id, labels, scores, boxes], dim=1)
        return bbox_list
    
    bbox_list = torch.cat([image_id, labels, boxes], dim=1)
    
    return bbox_list


def mean_average_precision(
    pred_bboxes, true_bboxes, iou_threshold=0.5, box_format='corner', num_classes=91
):
    """
    inputs:
        pred_bboxes (tensor): tensor of predicted bboxes[img_id, class_id, confidence_score, bbox coordinate(4)]
        true_bboxes (tensor): tensor of ground-thruth bboxes[img_id, class_id, bbox coordinate(4)]
    
    outputs:
        mAP (float) : Mean of average precision about all classes.
        AP_per_class (tensor) : Average precision about each classes.
    """
    
    AP_per_class = torch.zeros((num_classes))
    epsilon = 1e-6
    with tqdm(range(num_classes), unit='class') as t:
        for cls in t:
            cls_mask_pred = pred_bboxes[:, 1] == cls
            cls_mask_true = true_bboxes[:, 1] == cls
            
            
            detections = pred_bboxes[cls_mask_pred, :]
            ground_truths = true_bboxes[cls_mask_true, :]
            
            # bboxes_per_image = {0:tensor(0,0,0), 1:tensor(0,0,0,0), ...}
            objects_per_image = ground_truths[:, 0].long().bincount()
            bboxes_per_image = {k:torch.zeros(v) for k, v in enumerate(objects_per_image) if v > 0}
            
            detections = sorted(detections, key=lambda x: x[2], reverse=True)
            TP = torch.zeros(len(detections))
            FP = torch.zeros(len(detections))
            total_gt_bboxes = len(ground_truths)
            
            # gt_bboxes per 1 predicted bbox
            for detection_idx, detection in enumerate(detections):
                gt_mask = ground_truths[:, 0] == detection[0]
                gt_bboxes = ground_truths[gt_mask, :]
                
                best_iou = 0
                for idx, gt_bbox in enumerate(gt_bboxes):
                    iou = intersection_over_union(
                        detection[3:],
                        gt_bbox[2:],
                        box_format=box_format
                    )
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_bbox_idx = idx
                        
                if best_iou >= iou_threshold:
                    if bboxes_per_image[detection[0].item()][best_gt_bbox_idx] == 0:
                        TP[detection_idx] = 1
                        bboxes_per_image[detection[0].item()][best_gt_bbox_idx] = 1
                    else:
                        FP[detection_idx] = 1    
                else:
                    FP[detection_idx] = 1
            
            TP_cumsum = TP.cumsum(dim=0)
            FP_cumsum = FP.cumsum(dim=0)
            
            # Recall
            recalls = TP_cumsum / (total_gt_bboxes + epsilon)
            recalls = torch.cat([torch.tensor([0]), recalls])
            
            # Precision
            precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
            precisions = torch.cat([torch.tensor([1]), precisions])
            
            # AP
            ap = torch.trapz(precisions, recalls)
            AP_per_class[cls] = ap

    return sum(AP_per_class) / len(AP_per_class), AP_per_class


def intersection_over_union(boxes_preds, boxes_labels, box_format='corners'):
    # boxes_preds의 shape은 (N, 4), N은 예측한 객체의 개수
    # boxes_labels의 shape은 (N, 4)

    if box_format == 'corners': # YOLO dataset
        preds_x1 = boxes_preds[..., 0:1]
        preds_y1 = boxes_preds[..., 1:2]
        preds_x2 = boxes_preds[..., 2:3]
        preds_y2 = boxes_preds[..., 3:4]
        labels_x1 = boxes_labels[..., 0:1]
        labels_y1 = boxes_labels[..., 1:2]
        labels_x2 = boxes_labels[..., 2:3]
        labels_y2 = boxes_labels[..., 3:4]

    elif box_format == 'midpoint': # VOC-PASCAL dataset
        preds_x1 = bboxes_preds[..., 0:1] - bboxes_preds[..., 2:3] / 2
        preds_y1 = bboxes_preds[..., 1:2] - bboxes_preds[..., 3:4] / 2
        preds_x2 = bboxes_preds[..., 0:1] + bboxes_preds[..., 2:3] / 2
        preds_y2 = bboxes_preds[..., 1:2] + bboxes_preds[..., 3:4] / 2
        labels_x1 = bboxes_labels[..., 0:1] - bboxes_labels[..., 2:3] / 2
        labels_y1 = bboxes_labels[..., 1:2] - bboxes_labels[..., 3:4] / 2
        labels_x2 = bboxes_labels[..., 0:1] + bboxes_labels[..., 2:3] / 2
        labels_y2 = bboxes_labels[..., 1:2] + bboxes_labels[..., 3:4] / 2
        
    else: # COCO dataset
        preds_x1 = boxes_preds[..., 0:1]
        preds_y1 = boxes_preds[..., 1:2]
        preds_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3]
        preds_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4]
        labels_x1 = boxes_labels[..., 0:1]
        labels_y1 = boxes_labels[..., 1:2]
        labels_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3]
        labels_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4]

    # Intersection Area
    x1 = torch.max(preds_x1, labels_x1)
    y1 = torch.max(preds_y1, labels_y1)
    x2 = torch.min(preds_x2, labels_x2)
    y2 = torch.min(preds_y2, labels_y2)
    
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    preds_area = abs((preds_x2 - preds_x1) * (preds_y2 - preds_y1))
    labels_area = abs((labels_x2 - labels_x1) * (labels_y2 - labels_y1))
    
    # print(f"bbox1 Area : {preds_area.item()} \nbbox2 Area : {labels_area.item()} \nIntersection Area : {intersection.item()}")
    return (intersection / (preds_area + labels_area - intersection + 1e-6)).item()