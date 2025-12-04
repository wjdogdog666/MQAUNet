import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from torch.cuda.amp import autocast 
torch.cuda.empty_cache() 

# 自动检测设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    model.train() 
    model=model.to(device)
    loss_list = []

    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        images, targets = data
        images, targets = images.float().to(device), targets.float().to(device)  # 传输到 GPU
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            logger.info(log_info)
    scheduler.step() 
    return np.mean(loss_list)

def val_one_epoch(test_loader,
                    model,
                    criterion, 
                    epoch, 
                    logger,
                    config):
    # switch to evaluate mode
    model.eval()
    model=model.to(device)  # 确保模型在 GPU 上
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img, msk = img.float().to(device), msk.float().to(device)  # 传输到 GPU
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if isinstance(out, tuple):
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 

    if epoch % config.val_interval == 0:
        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        # 更新后的评价指标计算
        Acc = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        Pre = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0  # 精确率
        Rec = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0  # 召回率
        DSC = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0  # Dice系数
        IoU = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0  # 交并比

        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}, Acc: {Acc:.4f}, DSC: {DSC:.4f}, Pre: {Pre:.4f}, Rec: {Rec:.4f}, IoU: {IoU:.4f}'
        print(log_info)
        logger.info(log_info)

    else:
        log_info = f'val epoch: {epoch}, loss: {np.mean(loss_list):.4f}'
        print(log_info)
        logger.info(log_info)
    
    return np.mean(loss_list)

def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.float().to(device), msk.float().to(device)
           
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            if i % config.save_interval == 0:
                save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        # 更新后的评价指标计算
        Acc = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        Pre = float(TP) / float(TP + FP) if float(TP + FP) != 0 else 0
        Rec = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        DSC = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        IoU = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f}, Acc: {Acc:.4f}, DSC: {DSC:.4f}, Pre: {Pre:.4f}, Rec: {Rec:.4f}, IoU: {IoU:.4f}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)