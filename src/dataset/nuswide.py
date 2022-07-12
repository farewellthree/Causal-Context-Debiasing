import os
from PIL import Image
from torchvision import datasets as datasets
import torch
import torch.utils.data as data

class NuswideClf(data.Dataset):
    def __init__(self, root, csv_path, phrase='train', transform=None, target_transform=None):
        self.root = root
        self.csv_path = csv_path
        self.class_path = os.path.join(self.root,'Concepts81.txt')
        self.classes = {}
        self.phrase = phrase
        self.transform = transform
        self.target_transform = target_transform
        with open(self.class_path,'r') as cp:
            for i,c in enumerate(cp.readlines()):
                c = c.strip()
                self.classes[c] = i
        
        self.images = []
        self.targets = []
         
        with open(os.path.join(self.root,self.csv_path),'r') as csp:
            lines = csp.readlines()
            lines = lines[1:]
            for line in lines:
                line = line.strip()
                if '\"' in line:
                    img_path, labels_phrase = line.split(',\"')
                    labels, phrase = labels_phrase.split('\",')
                else:
                    img_path, labels, phrase = line.split(',')
                
                if phrase!=self.phrase:
                    continue
                self.images.append(img_path)
                labels = (labels[2:-2]).split('\', \'')
                
                label_ids = [self.classes[k] for k in labels]        
                self.targets.append(label_ids)
     
    def __getitem__(self, index):
        path, targets = self.images[index], self.targets[index]
        output = torch.zeros((81), dtype=torch.long)

        for target in targets:
            output[target] = 1
        target = output
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)