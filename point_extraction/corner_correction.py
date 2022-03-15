from turtle import update
import numpy as np
import torch
import matplotlib.pyplot as plt
import model
import os
import time
import copy
from torchvision import transforms


net = model.SimpleNetwork23b()
state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_NewSet62.pth')
net.load_state_dict(state_dict['model_state_dict'])


grooves = os.listdir('/home/oyvind/Downloads/grooves')

for groove in grooves:

    path = '/home/oyvind/Downloads/grooves/' + groove
    print(path)
    hei = np.load(path)
    chei = hei
    hei = np.flip(hei, axis=1)


    hei = hei/1000
    hei = torch.from_numpy(np.array(hei))
    hei = hei.type(torch.float32)
    t = transforms.Normalize((0.0017, 0.2328), (0.0310, 0.0198))
    hei = hei.unsqueeze(0)
    hei = hei.permute(1, 0, 2)
    hei = t(hei)
    hei = hei.squeeze()


    hei = hei.unsqueeze(0)

    net.eval()
    with torch.no_grad():
        start = time.time()
        ut = net(hei)

    hmm = ut.detach().numpy()
    hmm = hmm*1000
    hmm = hmm.squeeze()
    

    cheiflip = np.flip(chei, axis=1)
    idx = np.searchsorted(cheiflip[0], hmm.T[0], side="left") 
    for i in range(len(idx)-1):
        segment1 = cheiflip[:,idx[i]:idx[i+1]-1]
        segment1 = np.c_[hmm[i],segment1, hmm[i+1]]
        print(i)
        #if len(segment1[0]) > 175:
        #    print(len(segment1[0]))
        #    #x = segment1[0, int(len(segment1[0]) / 1.5):]
        #    #y = segment1[1, int(len(segment1[0]) / 1.5):]
        #    x = segment1[0, -100:]
        #    y = segment1[1, -100:]
        #else:
        x = segment1[0]
        y = segment1[1]
        slope, intercept = np.polyfit(x, y, 1)
        slope_est, intercept_est = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
        line_values = [slope * i + intercept for i in x]
        line_values_est = [slope_est * i + intercept_est for i in [x[0], x[-1]]]
        xs = [x[0], x[-1]]
        
        
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        center_x = 0.5 * (x[0] + x[-1])
        center_y = 0.5 * (y[0] + y[-1])
        diff_x = mean_x - center_x
        diff_y = mean_y - center_y

        corners = np.vstack((np.array([x[0], x[-1]]), np.array([y[0], y[-1]])))
        old_corners = copy.deepcopy(corners)
        corners[0] = corners[0] + diff_x
        corners[1] = corners[1] + diff_y
        
        corners[0] = corners[0] - mean_x
        corners[1] = corners[1] - mean_y

        angle = np.arctan((slope - slope_est) / (1 + slope * slope_est))

        rot = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])

        corners = rot @ corners
        
        corners[0] = corners[0] + mean_x
        corners[1] = corners[1] + mean_y
        plt.scatter(x, y, s = 5)
        plt.plot(x, line_values, c='r')
        plt.plot(xs, line_values_est)
        plt.scatter(old_corners[0], old_corners[1], c='g')
        plt.scatter(corners[0], corners[1], c='m')
        #plt.show()

        print('f')
        plt.scatter(chei[0], chei[1],s=1)
        plt.scatter(hmm[:,0], hmm[:,1], s=20) # når output (B, 5, 2)
        #plt.show()
        if i == 0:
            updated_corners = corners
        else:
            updated_corners = np.c_[updated_corners, corners]
    plt.show()
    print(updated_corners)    

    c1 = np.mean(updated_corners[:,1:3], axis=1)    
    c2 = np.mean(updated_corners[:,3:5], axis=1)  
    c3 = np.mean(updated_corners[:,5:7], axis=1)
    c = np.c_[c1,c2,c3]

    plt.scatter(chei[0], chei[1], s=1)
    plt.scatter(c[0], c[1], s=20, c='g')
    plt.scatter(hmm[:,0], hmm[:,1], s=20, c='r')
    plt.show()
    break