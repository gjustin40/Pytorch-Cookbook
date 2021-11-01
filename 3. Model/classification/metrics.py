import torchmetrics

if __name__ == '__main__':
    import torch
    # import our library
    import torchmetrics

    # initialize metric
    metric = torchmetrics.Accuracy(num_classes=5).cuda()
    recall = torchmetrics.Recall(num_classes=5, multiclass=True, average='samples').cuda()
    precision = torchmetrics.Precision(num_classes=5, multiclass=True).cuda()

    n_batches = 100
    for i in range(n_batches):
        # simulate a classification problem
        preds = torch.randn(10, 5).cuda()
        target = torch.randint(5, (10,)).cuda()
        
        # metric on current batch
        metric(preds, target)
        recall(preds, target)
        precision(preds, target)
        result(preds, target)
        #print(f"Accuracy on batch {i}: {acc:0.4f}")
    # metric on all batches using custom accumulation
    acc = metric.compute()
    recall = metric.compute()
    precision = metric.compute()

    print(f"Accuracy on all data: {acc:0.4}")
    print(f"Recall: {recall:0.4}")
    print(f"Precision: {precision:0.4}")
    # print(f'result : {result.compute()}')
    acc = metric.compute()
    # Reseting internal state such that metric ready for new data
    metric.reset()