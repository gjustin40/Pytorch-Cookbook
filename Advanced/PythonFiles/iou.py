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