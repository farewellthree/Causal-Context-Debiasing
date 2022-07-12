import torch
import torch.nn as nn
import math

class Interventional_Classifier(nn.Module):
    def __init__(self, num_classes=80, feat_dim=2048, num_head=4, tau=32.0, beta=0.03125, *args):
        super(Interventional_Classifier, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_classes, feat_dim).cuda(), requires_grad=True)
        self.scale = tau / num_head   # 32.0 / num_head
        self.norm_scale = beta       # 1.0 / 32.0      
        self.num_head = num_head
        self.head_dim = feat_dim // num_head
        self.reset_parameters(self.weight)
        self.feat_dim = feat_dim
        
    def reset_parameters(self, weight):
        stdv = 1. / math.sqrt(weight.size(1))
        weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x_list = torch.split(x, self.head_dim, dim=1)
        w_list = torch.split(self.weight, self.head_dim, dim=1)
        y_list = []
        for x_, w_ in zip(x_list, w_list):
            normed_x = x_ / torch.norm(x_, 2, 1, keepdim=True)
            normed_w = w_ / (torch.norm(w_, 2, 1, keepdim=True) + self.norm_scale)
            y_ = torch.mm(normed_x * self.scale, normed_w.t())   
            y_list.append(y_)
        y = sum(y_list)
        return y


class CosNorm_Classifier(nn.Module):
    def __init__(self, in_dims, out_dims, scale=16):
        super(CosNorm_Classifier, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.scale = scale
        self.weight = nn.Parameter(torch.Tensor(out_dims, in_dims).cuda())
        self.reset_parameters() 

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, *args):
        norm_x = torch.norm(input.clone(), 2, 1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (input / norm_x)
        ew = self.weight / torch.norm(self.weight, 2, 1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())

