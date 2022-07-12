import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast

from src.dataset.create_dataset import create_dataset
from src.loss_functions.losses import create_loss
from src.utils import mAP, ModelEma, add_weight_decay, cal_confounder, model_transfer
from src.models import CCD

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', help='(coco,voc,nuswide)', default='coco')
parser.add_argument('--data_path', help='path to dataset', default='')
parser.add_argument('--transforms', help='data transform style (asl or mlgacn)', default='asl')
parser.add_argument('--pretrain_path', default='', type=str)
parser.add_argument('--lr', default='1e-4', type=float)
parser.add_argument('--num-classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--save_path', default='')
parser.add_argument('--loss', default='mlsm', type=str, help='(mlsm,bce,focal,asl,halfasl)')
parser.add_argument('--use_tde', default=False, type=bool)
parser.add_argument('--feat_fuse', default='selector', type=str,help='none,mlp,selector')
parser.add_argument('--use_intervention', default=False, type=bool)
parser.add_argument('--stop_epoch', default=5, type=int)
parser.add_argument('--backbone', default='resnet101', type=str, help='(resnet101,vit,swim_transformer,swim_transformer_large)')


def main():
    args = parser.parse_args(sys.argv[1:])
    
    # Setup model
    print('creating model...')
    model = CCD(backbone=args.backbone, num_classes=args.num_classes, pretrain=args.pretrain_path, use_intervention=args.use_intervention, use_tde=args.use_tde, feat_fuse=args.feat_fuse)
    args.feat_dim = model.feat_dim
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.cuda()
    print('done\n')
    
    #Data loading
    train_dataset, val_dataset = create_dataset(args)
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    train_loader_cfer = torch.utils.data.DataLoader(train_dataset, 
        batch_size=args.batch_size//8, shuffle=False, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # Actuall Training
    args.stage = 1
    model = train_multi_label_coco(model, train_loader, val_loader, args)
    if args.use_tde:
        print ('building confounder......')
        confounder = cal_confounder(train_loader_cfer,model,args)
        tde_model = CCD(backbone=args.backbone, num_classes=args.num_classes, pretrain=args.pretrain_path, use_intervention=args.use_intervention, use_tde=args.use_tde, feat_fuse=args.feat_fuse)
        tde_model = model_transfer(model,tde_model,confounder,args)
        args.stage = 2
        
        torch.cuda.empty_cache()
        train_multi_label_coco(tde_model, train_loader, val_loader, args)

def train_multi_label_coco(model, train_loader, val_loader, args):
    base_lr = args.lr
    save_name = args.save_path
    ema = ModelEma(model, 0.9997)  
    # set optimizer
    Epochs = 80
    Stop_epoch = 80
    weight_decay = 1e-4

    criterion = create_loss(args.loss)    
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=base_lr, weight_decay=0)
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr, steps_per_epoch=steps_per_epoch, epochs=Epochs, pct_start=0.2)

    #!
    if not os.path.exists(save_name):
        os.mkdir(save_name)

    highest_mAP = 0

    scaler = GradScaler()

    for epoch in range(Epochs):
        if epoch > Stop_epoch:
            break
        for i, (inputData, target) in enumerate(train_loader):
            inputData = inputData.cuda()
            target = target.cuda()  
          
            with autocast():  # mixed precision
                feat,output = model(inputData)  # sigmoid will be done in loss 
                output = output.float()
                loss = criterion(output, target)

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            # store information
            if i % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))
           
        model.eval()
        mAP_score = validate_multi(val_loader, model, ema)
        model.train()
        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            try:
                torch.save(ema.module.state_dict(), os.path.join(save_name, 'model-highest.ckpt'))
            except:
                pass
        
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(mAP_score, highest_mAP))
        if args.stage==1 and epoch==args.stop_epoch and args.use_tde:
            return model

def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
    
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input.cuda())[1]).cpu()
                output_ema = Sig(ema_model.module(input.cuda())[1]).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema)



if __name__ == '__main__':
    main()
