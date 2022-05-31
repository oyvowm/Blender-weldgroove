import pickle
import os
from shutil import copy2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import model
import time
from corner_correction_copy_2 import LineSegment 

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

        #print(len(ob.grooves[0][:]))
        #print(ob.grooves[0][5].shape)
        block_length = int(len(ob.grooves[0][:]) / 6)
        block_length_list = [block_length] * 6
        rest =  len(ob.grooves[0][:]) % 6
        if rest < 3:
            block_length_list[1:rest+1] = [i + 1 for i in block_length_list[1:rest+1]]
        else:
            block_length_list[:rest] = [i + 1 for i in block_length_list[:rest]]

        blocks_grooves = {}
        blocks_labels = {}
        current_idx = 0
        for i in range(6):
            idx = current_idx + block_length_list[i]
            print()
            if i < 3:
                blocks_grooves[str(i)] = ob.grooves[0][current_idx:idx]
                blocks_labels[str(i)] = ob.grooves[1][current_idx:idx]
            else:
                if i == 3:
                    blocks_grooves['2'].extend(ob.grooves[0][current_idx:idx])
                    blocks_labels['2'].extend(ob.grooves[1][current_idx:idx])
                elif i == 4:
                    blocks_grooves['1'].extend(ob.grooves[0][current_idx:idx])
                    blocks_labels['1'].extend(ob.grooves[1][current_idx:idx])
                else:
                    blocks_grooves['0'].extend(ob.grooves[0][current_idx:])
                    blocks_labels['0'].extend(ob.grooves[1][current_idx:])
            current_idx = idx





        #print('len block 0',len(blocks_grooves['0']))
        #print('len block 1',len(blocks_grooves['1']))
        #print('len block 2',len(blocks_grooves['2']))

        print()

        for section in blocks_grooves:
            method_deviation = 0
            for i in range(len(blocks_grooves[section])):
                print(i)
                groove = blocks_grooves[section][i][:,::2].T
                labels = blocks_labels[section][i][:,::2]

                groove = np.flip(groove, axis=1)
                labels = np.flip(labels, axis=0)
                chei = groove

                groove = groove / 1000
                groove = torch.from_numpy(np.array(groove))
                groove = groove.type(torch.float32)
                t = transforms.Normalize((0.0018, 0.2333), (0.0310, 0.0205))
                groove = groove.unsqueeze(0)
                groove = groove.permute(1, 0, 2)
                groove = t(groove)
                groove = groove.squeeze()
                groove = groove.unsqueeze(0)

                net.eval()
                with torch.no_grad():
                    inference_start = time.time()
                    output = net(groove)
                    #print(f'inference time: {time.time() - inference_start}')

                output = output.detach().numpy()
                output = output*1000
                output = output.squeeze()

                line = LineSegment('asdfasdf', chei, output, 0, labels=labels, display_line_segments=False)

                #print(line.corner_points)
                corner_diff = np.linalg.norm(labels[1:4]-line.corner_points[1:4])
                method_deviation += corner_diff
            method_deviation = method_deviation / len(blocks_grooves[section])
            print(f'The average deviation for section {section} is: {method_deviation}')
                #print(len(blocks_grooves[section]))
    """

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

"""