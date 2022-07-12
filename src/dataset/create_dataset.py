import os
from .coco import CocoDetection
from .voc import Voc2007Classification
from .nuswide import NuswideClf
from ..utils import CutoutPIL
import torchvision.transforms as transforms
from randaugment import RandAugment


def create_dataset(args):
    path, dataset = args.data_path, args.dataset
    if 'coco' in dataset:
        instances_path_val = os.path.join(path, 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(path, 'annotations/instances_train2014.json')
        
        data_path_val   = f'{path}/val2014'    
        data_path_train = f'{path}/train2014'  
        val_dataset = CocoDetection(data_path_val,
                                    instances_path_val,
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ]))
        if args.transforms == 'asl':
            train_dataset = CocoDetection(data_path_train,
                                          instances_path_train,
                                          transforms.Compose([
                                              transforms.Resize((args.image_size, args.image_size)),
                                              CutoutPIL(cutout_factor=0.5),
                                              RandAugment(),
                                              transforms.ToTensor(),
                                              # normalize,
                                          ]))
        elif args.transforms == 'mlgcn':
            train_dataset = CocoDetection(data_path_train,
                                          instances_path_train,
                                          transforms.Compose([
                                              transforms.Resize((args.image_size, args.image_size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop((args.image_size, args.image_size), scale=(0.7, 1.0)),
                                              transforms.ToTensor(),
                                              # normalize,
                                          ]))
        else:
            raise ValueError('data transform not implemented')

    
    elif 'voc' in dataset:
        if 2007 in dataset:
            path += 'voc2007'
        elif 2012 in dataset:
            path += 'voc2012'
        else:
            raise ValueError('dataset not implemented')

        val_dataset = Voc2007Classification(path,
                                    'val',
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                        # normalize, # no need, toTensor does normalization
                                    ]))
        if args.transforms == 'asl':
            train_dataset = Voc2007Classification(path,
                                          'train',
                                          transforms.Compose([
                                              transforms.Resize((args.image_size, args.image_size)),
                                              CutoutPIL(cutout_factor=0.5),
                                              RandAugment(),
                                              transforms.ToTensor(),
                                              # normalize,
                                          ]))
        elif args.transforms == 'mlgcn':
            train_dataset = Voc2007Classification(path,
                                          'train',
                                          transforms.Compose([
                                              transforms.Resize((args.image_size, args.image_size)),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop((args.image_size, args.image_size), scale=(0.7, 1.0)),
                                              transforms.ToTensor(),
                                              # normalize,
                                          ]))
        else:
            raise ValueError('data transform not implemented')    
        

    elif 'nus' in dataset:
        val_dataset = NuswideClf(path,
                                    'nus_wid_data.csv',
                                    'val',
                                    transforms.Compose([
                                        transforms.Resize((args.image_size, args.image_size)),
                                        transforms.ToTensor(),
                                       # normalize, # no need, toTensor does normalization
                                    ]))
        if args.transforms == 'asl':                           
            train_dataset = NuswideClf(path,
                                        'nus_wid_data.csv',
                                        'train',
                                        transforms.Compose([
                                            transforms.Resize((args.image_size, args.image_size)),
                                            CutoutPIL(cutout_factor=0.5),
                                            RandAugment(),
                                            transforms.ToTensor(),
                                           # normalize,
                                        ]))
        elif args.transforms == 'mlgcn':
            train_dataset = NuswideClf(path,
                                        'nus_wid_data.csv',
                                        'train',
                                        transforms.Compose([
                                            transforms.Resize((args.image_size, args.image_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomResizedCrop((args.image_size, args.image_size), scale=(0.7, 1.0)),
                                            transforms.ToTensor(),
                                           # normalize,
                                        ]))

    else:
        raise ValueError('dataset not implemented')
    
    return train_dataset, val_dataset




