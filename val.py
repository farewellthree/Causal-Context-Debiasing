import time
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from src.dataset.create_dataset import create_dataset
from src.loss_functions.losses import create_loss
from src.utils import mAP, AverageMeter
from src.models import CCD

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--dataset', help='(coco,voc,nuswide)', default='coco')
parser.add_argument('--data_path', help='path to dataset', default='')
parser.add_argument('--transforms', help='data transform style (asl or mlgacn)', default='asl')
parser.add_argument('--model_path', default='', type=str)
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
parser.add_argument('--thre', default=0.5, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency (default: 64)')
parser.add_argument('--all', default=True, type=bool, help='top3 or all')


def main():
    args = parser.parse_args()
    

    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    model = CCD(backbone=args.backbone, num_classes=args.num_classes, use_intervention=args.use_intervention, use_tde=args.use_tde, feat_fuse=args.feat_fuse)
    if args.use_tde:
        model.clf.stagetwo = True
    
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(state)
    print('done\n')
    model = model.eval()
    # Data loading code

    _, val_dataset = create_dataset(args)

    print("len(val_dataset)): ", len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(val_loader, model, args)


def validate_multi(val_loader, model, args):
    print("starting actuall validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    mAP_meter = AverageMeter()

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())[1]).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        output = output.data
        if args.all:
            idx = torch.sort(-output)[1]
            idx_after3 = idx[:,3:]
            output.scatter_(1,idx_after3,0.) 
        # measure accuracy and record loss
        pred = output.gt(args.thre).long()
        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        r_c = [float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0 if tp[
                                                                             i] > 0 else 0.0
               for i in range(len(tp))]
        f_c = [2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0 for
               i in range(len(tp))]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                i, len(val_loader), batch_time=batch_time,
                prec=prec, rec=rec))
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    print(
        '--------------------------------------------------------------------')
    print(' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
          .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("mAP score:", mAP_score)

    return

def validate_multi_ori(val_loader, model, args):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    targets = []
    for i, (input, target) in enumerate(val_loader):
        target = target
        target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            
            output_regular = Sig(model(input.cuda())[1]).cpu()
            

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    print("mAP score regular {:.2f}".format(mAP_score_regular))
    return mAP_score_regular

if __name__ == '__main__':
    main()
