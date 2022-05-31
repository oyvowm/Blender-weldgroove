import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import model
import time
from corner_correction_updated import LineSegment 

#root = 'C:/Users/oyvin/Desktop/grooves/4 RotPredOn/9 steps 60mm RPon'
#root = 'C:/Users/oyvin/Desktop/grooves/3 DirCorrOn/8 steps 70mm DCon'
root = '/home/oyvind/Downloads/grooves_second_dataset/3 DirCorrOn/8 steps 70mm DCon'
#root = '/home/oyvind/Downloads/grooves_second_dataset/1 RotPredOff DirCorrOff/56 steps 10mm RPoff DCoff'



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


net = model.SimpleNetwork23b()
#net = resnet.ResidualNetwork()    
#state_dict = torch.load('C:/Users/oyvin/Desktop/dfsdf/ResNet_NewSet123.pth')
state_dict = torch.load('/home/oyvind/Blender-weldgroove/ResNet_NewSet123.pth')
net.load_state_dict(state_dict['model_state_dict'])

ob = UnpickleGrooves()
#path = 'C:/Users/oyvin/Desktop/grooves/' 
path = '/home/oyvind/Downloads/grooves_second_dataset/'
groove_folders = os.listdir(path)


for folder in groove_folders:
    new_path = os.path.join(path, folder)
    segments = os.listdir(new_path)

    print(new_path)
    for segment in segments:
        segment_path = os.path.join(new_path, segment)
        print('segment path:', segment_path)

        ob.unpickle_groove(segment_path)

        print(len(ob.grooves[0][:]))
        print(ob.grooves[0][5].shape)




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
            #t = transforms.Normalize((0.0017, 0.2328), (0.0310, 0.0198))
            t = transforms.Normalize((0.0018, 0.2333), (0.0310, 0.0205))
            hei = hei.unsqueeze(0)
            hei = hei.permute(1, 0, 2)
            hei = t(hei)
            hei = hei.squeeze()

            hei = hei.unsqueeze(0)

            net.eval()
            with torch.no_grad():
                inference_start = time.time()
                ut = net(hei)
                #print(f'inference time: {time.time() - inference_start}')

            hmm = ut.detach().numpy()
            hmm = hmm*1000
            hmm = hmm.squeeze()

            line = LineSegment('asdfasdf', chei, hmm, 0, labels=labels, display_line_segments=False)
            #print(f'\n groove iterative correction time = {time.time() - groove_start_time} \n')
        print()
