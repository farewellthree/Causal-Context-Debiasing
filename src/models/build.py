import torch
import torch.nn as nn
from torch.nn import Module as Module
from collections import OrderedDict
import torchvision
import math
import numpy as np
from .classifier import Interventional_Classifier, CosNorm_Classifier
from .swin_transformer import SwinTransformer
from .vision_transformer import VisionTransformer

class CCD(Module):
    def __init__(self,backbone="resnet101",num_classes=80,pretrain=None,use_intervention=False,use_tde=False,feat_fuse='none'):
        super(CCD,self).__init__()
        if backbone=="resnet101":
            self.backbone = resnet101_backbone(pretrain)
        elif backbone=="swim_transformer":
            self.backbone = swimtrans_backbone(num_classes,pretrain)
        elif backbone=="swim_transformer_large":
            self.backbone = swimtrans_backbone(num_classes,pretrain,large = True)
        elif backbone=="vit":
            self.backbone = Vit_backbone(num_classes,pretrain)

        self.feat_dim = self.backbone.feat_dim
        
        if use_tde:
            self.clf = tde_classifier(num_classes,self.feat_dim,use_intervention,feat_fuse=feat_fuse)
        else:
            if not use_intervention:
                self.clf = nn.Linear(self.feat_dim,num_classes)
                
            else:
                self.clf = Interventional_Classifier(num_classes=num_classes, feat_dim=self.feat_dim, num_head=4, tau=32.0, beta=0.03125)
    
    def forward(self,x):
        feats = self.backbone(x)
        
        logits = self.clf(feats)
        return feats, logits

 
class resnet101_backbone(Module):
    def __init__(self, pretrain):
        super(resnet101_backbone,self).__init__()
        res101 = torchvision.models.resnet101(pretrained=False)
        if pretrain:
            path = pretrain
            state = torch.load(path, map_location='cpu')
            if type(state)==dict and "state_dict" in state:
                res101 = nn.DataParallel(res101)
                res101.load_state_dict(state["state_dict"])
                res101 = res101.module
            else:
                res101.load_state_dict(state)
        numFit = res101.fc.in_features
        self.resnet_layer = nn.Sequential(*list(res101.children())[:-2])
   
        self.feat_dim = numFit

    def forward(self,x):
        feats = self.resnet_layer(x)
       
        return feats

class swimtrans_backbone(Module):
    def __init__(self,num_classes,pretrain,large=False):
        super(swimtrans_backbone,self).__init__()
        if large:
            self.model = SwinTransformer(img_size=384,patch_size=4,num_classes=num_classes,embed_dim=192,depths=(2, 2, 18, 2),num_heads=(6, 12, 24, 48),window_size=12)
        else:
            self.model = SwinTransformer(img_size=384,patch_size=4,num_classes=num_classes,embed_dim=128,depths=(2, 2, 18, 2),num_heads=(4, 8, 16, 32),window_size=12)
        if pretrain:
            path = pretrain
            state = torch.load(path, map_location='cpu')['model']
            filtered_dict = {k: v for k, v in state.items() if(k in self.model.state_dict() and 'head' not in k)}
            self.model.load_state_dict(filtered_dict,strict=False)
        numFit = self.model.num_features
        self.feat_dim = numFit
        del self.model.head

    def forward(self,x):
        feats = self.model.forward_features(x)
        return feats

class Vit_backbone(Module):
    def __init__(self,num_classes,pretrain):
        super(Vit_backbone,self).__init__()
        
        self.model = VisionTransformer(img_size=224, patch_size=16,num_classes=num_classes,embed_dim=1024,depth=24,num_heads=16,representation_size=1024)

        if pretrain:
            path = pretrain
            self.model.load_pretrained(path)
        numFit = self.model.num_features
        self.feat_dim = numFit
        del self.model.head

    def forward(self,x):
        feats = self.model.forward_features(x)
        return feats

class tde_classifier(Module):
    def __init__(self,num_classes,feat_dim,use_intervention,stagetwo=False,feat_fuse='selector'):
        super(tde_classifier,self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.memory = nn.Parameter(torch.zeros((num_classes,feat_dim)).cuda(), requires_grad=False)
        self.feat_fuse = feat_fuse
        if feat_fuse=='selector':
            self.selector = nn.Linear(self.feat_dim,self.feat_dim)
        elif feat_fuse=='mlp':
            self.selector = twolayers_MLP(self.feat_dim)
        
        self.stagetwo = stagetwo
        if use_intervention:
            self.context_clf = Interventional_Classifier(num_classes=num_classes, feat_dim=feat_dim, num_head=4, tau=32.0, beta=0.03125)
            self.logit_clf = Interventional_Classifier(num_classes=num_classes, feat_dim=feat_dim, num_head=4, tau=32.0, beta=0.03125)
        else:
            self.context_clf = nn.Linear(feat_dim,num_classes)
            self.logit_clf = nn.Linear(feat_dim,num_classes)
        self.softmax = nn.Softmax(dim=1) 
        
    def forward(self,feats):
        if len(list(feats.size())) == 2:
            global_feat =  feats
            memory = self.memory
        else:
            global_feat = feats.flatten(2).mean(dim=-1)
            feats = feats.flatten(2).max(dim=-1)[0]  
            #memory = self.memory.flatten(2).mean(dim=-1) 
            memory = self.memory
        if self.stagetwo:
            pre_logits = self.softmax(self.context_clf(global_feat))
            memory_feature = torch.mm(pre_logits,memory)
            
            if self.feat_fuse == 'none':
                combine_feature = feats - memory_feature
            elif self.feat_fuse == 'selector':
                selector = self.selector(feats.clone())
                selector = selector.tanh()
                combine_feature = feats - selector * memory_feature
            elif self.feat_fuse == 'mlp':
                combine_feature = feats - self.selector(memory_feature.clone())

            logits = self.logit_clf(combine_feature)
        else:
            logits = self.context_clf(global_feat)
        return logits




























































