import torch
import torchmetrics


config = {
    'num_classes': 2
}


def train(model, dataloader, optimizer, loss_func, device, scheduler, writer, epoch, e):
    print(f'EPOCH[{e+1}/{epoch}] Training....')
    # print start train
    model.train()
    model.to(device)
    
    iter_losses = []
    iter_accuracies = []
    iter_data_size = 0
    
    train_acc = torchmetrics.Accuracy(num_classes=config['num_classes']).to(device)
    step = 0
    
    for i, (images, labels) in enumerate(dataloader):
        # Time
        
        images, labels = images.to(device), labels.to(device)
        iter_data_size += images.shape[0]
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_acc(outputs, labels)
        iter_losses.append(loss.item())
        
        ###### Tensorboard
        writer.add_scalar('Training Accuracy', train_acc.compute().cpu(), global_step=step)
        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1
        
        if ((i+1) % int(len(dataloader)/5) == 0) or ((i+1) == len(dataloader)):
            print(
                f'Iter[{i+1}/{len(dataloader)}]'\
                f' --- Loss: {sum(iter_losses)/iter_data_size:0.4f}'\
                f' --- Accuracy: {train_acc.compute():0.2f}'\
                f'--- LR: {scheduler.get_lr()[0]:0.4f}'\
            )
    
    epoch_acc = train_acc.compute().cpu()
    epoch_loss = sum(iter_losses) / iter_data_size
    
    return epoch_acc, epoch_loss


@torch.no_grad()
def val(model, dataloader, loss_func, device, writer, e):
    # print start validation
    model.eval()
    model.to(device)
    
    iter_losses = []
    iter_accuracies = []
    iter_data_size = 0
    
    val_acc = torchmetrics.Accuracy(num_classes=config['num_classes']).to(device)
    step = 0
    
    for i, (images, labels) in enumerate(dataloader):
        # Time
        
        images, labels = images.to(device), labels.to(device)
        iter_data_size += images.shape[0]
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        val_acc(outputs, labels)
        iter_losses.append(loss.item())
        
        ###### Tensorboard
        writer.add_scalar('Validation Accuracy', val_acc.compute().cpu())
        writer.add_scalar('Validation loss', loss, global_step=step)
        step += 1
        
        if ((i+1) % int(len(dataloader)/5) == 0) or ((i+1) == len(dataloader)):
            print(
                f'Iter[{i+1}/{len(dataloader)}]'\
                f' --- Loss: {sum(iter_losses)/iter_data_size:0.4f}'\
                f' --- Accuracy: {val_acc.compute():0.2f}'\
            )
           
    epoch_acc = val_acc.compute().cpu()
    epoch_loss = sum(iter_losses) / iter_data_size
    
    return epoch_acc, epoch_loss


@torch.no_grad()
def test(model, dataloader, loss_func, device, writer, e):
    # print start validation
    model.eval()
    model.to(device)
    
    iter_losses = []
    iter_accuracies = []
    iter_data_size = 0
    
    test_acc = torchmetrics.Accuracy(num_classes=config['num_classes']).to(device)
    step = 0
    
    for i, (images, labels) in enumerate(dataloader):
        # Time
        
        images, labels = images.to(device), labels.to(device)
        iter_data_size += images.shape[0]
        
        outputs = model(images)
        loss = loss_func(outputs, labels)
        
        test_acc(outputs, labels)
        iter_losses.append(loss.item())
        
        ###### Tensorboard
        writer.add_scalar('Test Accuracy', test_acc.compute().cpu())
        writer.add_scalar('Test loss', loss, global_step=step)
        step += 1
        
        if ((i+1) % int(len(dataloader)/5) == 0) or ((i+1) == len(dataloader)):
            print(
                f'Iter[{i+1}/{len(dataloader)}]'\
                f' --- Loss: {sum(iter_losses)/iter_data_size:0.4f}'\
                f' --- Accuracy: {test_acc.compute():0.2f}'\
            )
           
    epoch_acc = test_acc.compute().cpu()
    epoch_loss = sum(iter_losses) / iter_data_size
    
    return epoch_acc, epoch_loss
