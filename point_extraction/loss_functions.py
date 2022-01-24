import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential


a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059], [-0.3108, -2.4423], [-0.4821,  1.059]])

b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702], [-0.3108, -2.4423], [-0.4821,  1.059], [-0.3108, -2.4423]])

c = (torch.cdist(a, b, p=2))

d = torch.diagonal(c, 0)
print(d.mean())


class EuclideanLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self,x,y):
        #print(x.shape, 'x')
        #print(y.shape, 'y')
        #print(y.is_contiguous())
        y = torch.transpose(y, 1, 2).contiguous()
        #print(y.is_contiguous())
        #print(y.shape, 'y')
        dist = torch.cdist(x, y, p=2)
        dist = torch.mean(dist, 0)

        dist_diag = torch.diagonal(dist)

        loss = dist_diag.mean()
        #print(loss,'loss')
        return loss
