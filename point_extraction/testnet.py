import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import dataset
import model
import resnet
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from conv2d import Conv2DNetwork
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

net = model.SimpleNetwork23b()
#print(net)
#net = resnet.ResidualNetwork()
#net = Conv2DNetwork()
#et = model.FeedForwardNet()

state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_NewSet123.pth')
net.load_state_dict(state_dict['model_state_dict'])
net.to(device)
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

print('num parameters:',pytorch_total_params)

grooves = os.listdir('/home/oyvind/Downloads/grooves')

for groove in grooves:
    path = '/home/oyvind/Downloads/grooves/' + groove
    print(path)
    hei = np.load(path)


    #hei = np.load('/home/oyvind/Downloads/3mmRoot_noNoise.npy')
    #hei = np.load('/home/oyvind/Downloads/3mmRoot_VLstubNoise.npy')
    #hei = np.load('/home/oyvind/Downloads/3mmRoot_HstubNoise.npy')
    hei2 = np.load('/home/oyvind/Blender-weldgroove/render/80/processed_images/points_0009/0009_EST_fixed.npy')
    data = dataset.LaserPointDataset('/home/oyvind/Blender-weldgroove/render', noise=False, return_gt=True, corrected=True, normalization='', test=False, shuffle_gt_and_est=False)
    num = np.random.randint(0, len(data))
    print(num)
    e, g, p = data[num]
    idx = np.round(np.linspace(0, len(g[0]) - 1, 640)).astype(int)
    g = g[:, idx]
    g = g[1:]
    hei2 = g

    chei = hei
    chei2 = hei2 * 1000

    hei = np.flip(hei, axis=1)
    #chei = hei
    #chei=chei*1000

    #hei2 = hei2[1:]
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
    #t = transforms.Normalize((0.0012, 0.2145), (0.0263, 0.0183))
    #t = transforms.Normalize((0.0013, 0.2211), (0.0281, 0.0194))
    #t = transforms.Normalize((0.0015, 0.2292), (0.0305, 0.0192))
    #t = transforms.Normalize((0.0017, 0.2310), (0.0308, 0.0195)) #
    #t = transforms.Normalize((0.0017, 0.2328), (0.0310, 0.0198))
    #t = transforms.Normalize((0.0061, 0.2474), (0.0307, 0.0298)) # with noise
    t = transforms.Normalize((0.0018, 0.2333), (0.0310, 0.0205))
    
    hei = hei.unsqueeze(0)
    hei = hei.permute(1, 0, 2)
    hei = t(hei)
    hei = hei.squeeze()


    #hei = hei/1000
    #hei = dataset.cloud_normalization(hei)
    #hei = torch.from_numpy(hei)
    #hei = hei.type(torch.float32)

    hei = hei.unsqueeze(0)
    hei = hei.to(device)
    hei2 = hei2.to(device)
    net.eval()
    with torch.no_grad():
        start = time.time()
        ut = net(hei)
        print(f'inference time: {time.time() - start}')
        ut2 = net(hei2)

    
    hmm = ut.cpu().detach().numpy()
    hmm = hmm*1000
    hmm = hmm.squeeze()

    hmm2 = ut2.cpu().detach().numpy()
    hmm2 = hmm2*1000
    hmm2 = hmm2.squeeze()
    #print(chei[0])
    #ut = ut * 1000

    #print(chei)
    plt.scatter(chei[0], chei[1],s=1)
    
    plt.scatter(hmm[:,0], hmm[:,1], s=20) # når output (B, 5, 2)
    
    #plt.scatter(hmm[0], hmm[1], s=2) # når output (B, 2, 5)
    #plt.show()

    #plt.scatter(chei2[0], chei2[1],s=1)
    #plt.scatter(hmm2[:,0], hmm2[:,1], s=20) # når output (B, 5, 2)

    p = p * 1000
    #plt.scatter(p[0], p[1], s = 20)


    #plt.scatter(hmm2[0], hmm2[1], s=20)
    path = groove + '.png'
    plt.savefig('results/' + path)
    plt.show()

    #print(ut)


