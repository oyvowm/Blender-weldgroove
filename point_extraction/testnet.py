import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import model
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from conv2d import Conv2DNetwork


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#net = model.SimpleNetwork2()
net = model.ResidualNetwork()
#net = Conv2DNetwork()
#et = model.FeedForwardNet()

state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_lessLRDecay.pth')



net.load_state_dict(state_dict['model_state_dict'])

pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

print('num parameters:',pytorch_total_params)

hei = np.load('/home/oyvind/Downloads/3mmRoot_noNoise.npy')
#hei = np.load('/home/oyvind/Downloads/3mmRoot_VLstubNoise.npy')
#hei = np.load('/home/oyvind/Downloads/3mmRoot_HstubNoise.npy')
hei2 = np.load('/home/oyvind/Blender-weldgroove/render/27/processed_images/points_0002/0002_EST_fixed.npy')
chei = hei
chei2 = hei2[1:] * 1000

hei = np.flip(hei, axis=1)
#chei = hei
#chei=chei*1000

hei2 = hei2[1:]
hei2 = dataset.normalize(hei2)
#hei2 = dataset.cloud_normalization(hei2)
#hei2 = hei2 * 1000

#hei2 = torch.from_numpy(hei2)
#hei2 = hei2.type(torch.float32)
hei2=hei2.unsqueeze(0)






hei = hei/1000
hei = torch.from_numpy(np.array(hei))
hei = hei.type(torch.float32)
#t = transforms.Normalize((0.0016, 0.2542), (0.0318, 0.0260))
t = transforms.Normalize((0.0011, 0.2079), (0.0253, 0.0180))
hei = hei.unsqueeze(0)
hei = hei.permute(1, 0, 2)
hei = t(hei)
hei = hei.squeeze()


#hei = hei/1000
#hei = dataset.cloud_normalization(hei)
#hei = torch.from_numpy(hei)
#hei = hei.type(torch.float32)

hei = hei.unsqueeze(0)

net.eval()
with torch.no_grad():
    start = time.time()
    ut = net(hei)
    print(f'inference time: {time.time() - start}')
    ut2 = net(hei2)

hmm = ut.detach().numpy()
hmm = hmm*1000
hmm = hmm.squeeze()

hmm2 = ut2.detach().numpy()
hmm2 = hmm2*1000
hmm2 = hmm2.squeeze()
#print(chei[0])
#ut = ut * 1000

#print(chei)
plt.scatter(chei[0], chei[1],s=1)
plt.scatter(hmm[0], hmm[1], s=20)
plt.show()

plt.scatter(chei2[0], chei2[1],s=1)
plt.scatter(hmm2[0], hmm2[1], s=20)
plt.show()

#print(ut)


