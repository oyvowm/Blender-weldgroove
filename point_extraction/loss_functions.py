from dis import dis
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.container import Sequential


#a = torch.tensor([[0.9041,  0.0196], [-0.3108, -2.4423], [-0.4821,  1.059], [-0.3108, -2.4423], [-0.4821,  1.059]])
#print(a)
#b = torch.tensor([[-2.1763, -0.4713], [-0.6986,  1.3702], [-0.3108, -2.4423], [-0.4821,  1.059], [-0.3108, -2.4423]])
#print(b)
#c = (torch.cdist(a, b, p=2))
#print(c)
#print(torch.mean(c, 0))
#
#print(torch.diagonal(torch.mean(c, 0)))


class EuclideanLoss(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):

        # finn nærmest x punkter, hvis avstand > en grense -> inkluder i loss.

        y = torch.transpose(y, 1, 2).contiguous()
        #print(y.is_contiguous())
        #print(y.shape, 'y')
        dist = torch.cdist(x, y, p=2)
        dist = torch.mean(dist, 0) # averaging acroos batch dimension

        dist_diag = torch.diagonal(dist) # diagonal entries correspond to distance between pairs of row vectors.
        #print(dist_diag)
        #print(dist_diag[1:3])
        dist_diag[1:3] = dist_diag[1:3] * 1.5 # scaling up the loss for the root corner points.
        #print(dist_diag)
        loss = dist_diag.mean()
        #print(loss,'loss')
        return loss

    
class EuclideanLoss2(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y, alt):

        # finn nærmest x punkter, hvis avstand > en grense -> inkluder i loss.
        
       
        alt = torch.transpose(alt, 1, 2).contiguous()
        y = torch.transpose(y, 1, 2).contiguous()
        d = torch.cdist(y[:,1:4,:], alt, p=2)
        #print(y.is_contiguous())
        #print(y.shape, 'y')
        dist = torch.cdist(x, y, p=2)
        dist = torch.mean(dist, 0) # averaging acroos batch dimension

        dist_diag = torch.diagonal(dist) # diagonal entries correspond to distance between pairs of row vectors.
        #print(dist_diag)
        #print(dist_diag[1:3])
        dist_diag[1:3] = dist_diag[1:3] * 1.5 # scaling up the loss for the root corner points.
        #print(dist_diag)
        loss = dist_diag.mean()
        #print(loss,'loss')

        




        return loss

if __name__ == "__main__":
    y = torch.randn(5, 2, 5)
    x = torch.randn(5, 5, 2)
    alt = torch.randn(5, 2, 3)


    criterion = EuclideanLoss2()

    ut = criterion(x, y, alt)
    print(ut)