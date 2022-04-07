import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import model
import time
from corner_correction_updated import LineSegment 


root = '/home/oyvind/Downloads/grooves_second_dataset/4 RotPredOn/5 steps 100mm RPon'
root = '/home/oyvind/Downloads/grooves_second_dataset/1 RotPredOff DirCorrOff/56 steps 10mm RPoff DCoff'



class UnpickleGrooves():
    def __init__(self):
        self.grooves = []

    def unpickle_groove(self, path):
        #files = os.listdir(path)
        infile = open(os.path.join(path, 'groPoints_2'), 'rb')
        groPoints = pickle.load(infile)
        infile.close()

        infile = open(os.path.join(path, 'groCorners_2'), 'rb')
        groCorners = pickle.load(infile)
        infile.close()

        self.grooves = [groPoints, groCorners]


ob = UnpickleGrooves()

ob.unpickle_groove(root)

print(len(ob.grooves[0][:]))
print(ob.grooves[0][5].shape)

net = model.SimpleNetwork23d()
#net = resnet.ResidualNetwork()
state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_NewSet76.pth')
net.load_state_dict(state_dict['model_state_dict'])


for i in range(len(ob.grooves[0])):
    groove_start_time = time.time()
    hei = ob.grooves[0][i][:,::2]
    labels = ob.grooves[1][i][:,::2]
    hei = hei.T
    
    hei = np.flip(hei, axis=1)

    chei = hei

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

    line = LineSegment(chei, hmm, 0, labels=labels)
    print(f'\n groove iterative correction time = {time.time() - groove_start_time} \n')
print()

