import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(
    pred_bboxes, true_bboxes, iou_threshold=0.5, box_format='corner', num_classes=91
):
    AP_per_class = torch.zeros((num_classes))
    epsilon = 1e-6
    
    for cls in range(num_classes):
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



def mean_average_precision2(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format='corner', num_classes=20
):
    # train_idx = 각 이미지 index(해당 bbox가 어느 image에 속해있는지)
    # pred boxes (list) : [[train_idx, class, confidence, boxes(4)], ...]
    # true boxes (list) : [[train_idx, class, boxes(4)], ...]
    
    # Average Precision for each class
    average_precisions = []
    epsilon = 1e-6
    
    # 각 class별로 AP 계산
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        # 예측 bbox 중 class가 c인 bbox
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        # 정답 bbox 중 class가 c인 bbox
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
                
        # 각 이미지별(train_idx) 정답 bbox의 개수
        # 예를 들어 img0에 3개의 정답 bbox, img1에 5개의 정답 bbox
        # ammount_bboxes = {0: 3, 1: 5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
             # {0:torch.Tensor([0,0,0]), 1:torch.Tensor([0,0,0,0,0])}
            amount_bboxes[key] = torch.zeros(val)
        
        # 예측 bbox들을 confidence를 기준으로 내림차순 정렬    
        detections.sort(key=lambda x: x[2], reversed=True)
        TP = torch.zeros(len(detections)) # 각각의 예측 bbox들이 TP인지 아닌지(0 or 1)
        FP = torch.zeros(len(detections)) # 각각의 예측 bbox들이 FP인지 아닌지(0 or 1)
        total_true_bboxes = len(ground_truths) # 정답 bbox들의 개수
        
        # IoU를 통해 각각의 예측 bbox에 대해 대응되는 정답 bbox 찾기
        for detection_idx, detection in enumerate(detections):
            
            # 1개의 예측 bbox에 대해 같은 image에 있는 정답 bbox들 모두 찾기
            # 같은 image에 있는 예측 bbox와 정답 bbox의 쌍은 알지만
            # 각각의 bbox가 서로 대응되는지 모르기 때문에 이런 방법으로 접근
            # 해당 예측 bbox가 속해있는 image에 있는 정답 bbox들을 모두 찾겠지?
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            
            num_gts = len(ground_truth_img)
            best_iou = 0 
            
            for idx, gt in enumerate(ground_truth_img):
                # 1개의 예측 bbox와 같은 image에 있는 모든 정답 bbox들의 IoU 계산
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[2:]),
                    box_format=box_format,
                )
                
                # 예측 bbox와 IoU가 가장 큰 정답 bbox
                if iou > best_iou:
                    best_iou = iou # 가장 큰 IoU값
                    best_gt_idx = idx # 예측 bbox와 매칭되는 정답 bbox의 idx
            
            # image에 있는 정답 bbox 중 예측 bbox와의 IoU값이 Threshold를 넘겼을 때
            if best_iou > iou_threshold:
                # 만약 본 적 없는 정답 bbox일 경우
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1 # Threshold를 넘었기에 해당 예측 bbox는 Ture Positive
                    amount_bboxes[detection[0]][best_gt_idx] = 1 # 해당 정답 bbox는 본 것으로 처리
                    
                # 만약 본 적 있는 정답 bbox일 경우
                else:
                    # 이미 해당 예측 bbox는 정답 bbox와 짝을 이루었기에
                    # 이미 짝을 이룬 정답 bbox와의 IoU가 최대인 예측 bbox는 False Positive
                    FP[detection_idx] = 1 
                
            # 만약 예측 bbox가 특정 IoU가 넘는 정답 bbox가 없을 때    
            else:
                FP[detection_idx] = 1
        
        # TP = [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        # 이것은 마치 confidence순으로 정렬된 예측 bbox에 대해
        # 점점 예측 bbox들을 추가하면서 precision과 Recall을 구해주는 작업과 동일
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        
        # Recall 계산
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        recalls = torch.cat((torch.tensor([0]), recalls)) # Recall 시작은 x=0, 따라서 처음 0값 추가
        
        # Precision 계산
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions)) # Precision 시작은 y=1, 따라서 처음 1값 추가
        
        # y value와 x value를 이용해 그래프 넓이 계산
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)


