from audioop import avg
from tkinter import E
from turtle import update
from types import new_class
from cv2 import projectPoints
from matplotlib import lines
import numpy as np
import torch
import matplotlib.pyplot as plt
import model
import resnet
import os
import time
import copy
from torchvision import transforms
import loss_functions


# Uses the neural network + corner correction algorithm for an alternative way of extracting ground truth labels.
# necessary as the finite-differences method struggled when using wider root welds.



if __name__ == '__main__':
    
    # loads net
    net = model.SimpleNetwork23c()
    #net = resnet.ResidualNetwork()
    state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_NewSet72.pth')
    net.load_state_dict(state_dict['model_state_dict'])



    grooves = os.listdir('/home/oyvind/Downloads/grooves')
    print(grooves)
    print(grooves[1:2])

    for groove in grooves[2:3]:
        groove_start_time = time.time()

        path = '/home/oyvind/Downloads/grooves/' + groove
        #print(path)
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
            inference_start = time.time()
            ut = net(hei)
            print(f'inference time: {time.time() - inference_start}')

        hmm = ut.detach().numpy()
        hmm = hmm*1000
        hmm = hmm.squeeze()
        

        cheiflip = np.flip(chei, axis=1)

        # definer objekt
        line = LineSegment(cheiflip, hmm, 0)
        print(f'\n groove iterative correction time = {time.time() - groove_start_time} \n')

    
        


    